"""
Punjab GeoGuardian - interactive web app (Lahore, 2019 vs 2025)

Run (from a terminal, inside your .venv):
    pip install streamlit streamlit-folium folium requests
    streamlit run geoguardian_app.py

How to use
  1. Search a place name / paste "lat, lon" / click the map / jump to a model-flagged hotspot.
  2. Two synchronised maps open (left = 2019, right = 2025, zoom up to level 22). Switch between
     dated high-resolution photos, the Sentinel-2 images the classifier used, and the class overlay.
  3. Read the change summary, the evidence check, and the 2019-vs-2025 percentage table.

The "prediction" was made earlier in your notebook (Random Forest and Dynamic World, 3 classes).
This app only looks those saved rasters up for the chosen window - it does not retrain anything.
"""
import math
import os
import re
from datetime import date
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import rasterio
import requests
import streamlit as st
import streamlit.components.v1 as components
from folium.plugins import DualMap
from rasterio.windows import Window
from rasterio.windows import bounds as win_bounds
from streamlit_folium import st_folium

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
PROJECT_DIR = Path(os.environ.get("GEOGUARDIAN_DIR", r"E:\Punjab GeoGuardian"))
OUT = PROJECT_DIR / "outputs"
RAW = PROJECT_DIR / "data" / "raw"
BOUNDARY_PATH = PROJECT_DIR / "data" / "boundaries" / "Lahore_District.geojson"
YEARS = (2019, 2025)

PATHS = {
    "RF3": {y: OUT / "classifications" / f"RF3_{y}.tif" for y in YEARS},
    "DW3": {y: OUT / "classifications" / f"DW3_{y}.tif" for y in YEARS},
    "feat": {y: RAW / f"Sentinel2_{y}_Features.tif" for y in YEARS},
    "change": OUT / "change_detection" / "change_strata_2019_2025.tif",
}
METHODS = {
    "Random Forest (this project's model)": "RF3",
    "Dynamic World (Google labels)": "DW3",
}

SHORT = {0: "Vegetation / Agriculture", 1: "Built-up / Urban", 2: "Other (bare, water, mixed)"}
COLORS = {0: "#2e8b57", 1: "#d73027", 2: "#e0c068"}
NODATA = 255
CONV = "Vegetation -> Built-up (predicted conversion)"
STAYV = "Stayed vegetation (control)"
STAYB = "Stayed built-up (control)"
VIEW_PHOTO = "High-res photo (dated)"
VIEW_S2 = "Sentinel-2 true colour (what the model saw)"
VIEW_CLASSES = "Land-cover classes (over the photo)"
MAX_ZOOM = 22

LEFT, BOTTOM, RIGHT, TOP = 74.0034, 31.2559, 74.6531, 31.7166   # only used to bias place search

ESRI_SAT = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_LABELS = ("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/"
               "World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}")
ESRI_ATTR = "Esri, Maxar, Earthstar Geographics"
ESRI_GRAY = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
WAYBACK_CONFIG = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"

TRANSITION_MEANING = {
    (0, 1): "Possible agriculture/vegetation -> urban conversion",
    (0, 2): "Vegetation -> bare/other (cleared land, fallow field, seasonal change)",
    (1, 0): "Built-up -> vegetation (rare in reality; usually classification noise)",
    (1, 2): "Built-up -> other (usually noise, or cleared/bare land)",
    (2, 0): "Other -> vegetation (crops on previously bare land, seasonal change)",
    (2, 1): "Other -> built-up (bare land that was developed)",
}


# ----------------------------------------------------------------------
# Location input helpers
# ----------------------------------------------------------------------
def parse_coords(text):
    """Accepts '31.4674, 74.1928' (lat, lon). Also fixes 'lon, lat' for Pakistan."""
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*[,;\s]\s*(-?\d+(?:\.\d+)?)\s*$", text)
    if not m:
        return None
    a, b = float(m.group(1)), float(m.group(2))
    if 60 <= a <= 78 and 23 <= b <= 37:
        a, b = b, a
    if not (-90 <= a <= 90 and -180 <= b <= 180):
        return None
    return a, b


@st.cache_data(show_spinner=False, ttl=3600)
def geocode(query):
    """Place-name search through OpenStreetMap Nominatim (needs internet)."""
    params = {"q": query, "format": "jsonv2", "limit": 6, "countrycodes": "pk",
              "viewbox": f"{LEFT},{TOP},{RIGHT},{BOTTOM}", "bounded": 0}
    headers = {"User-Agent": "PunjabGeoGuardian-student-project/1.0"}
    r = requests.get("https://nominatim.openstreetmap.org/search", params=params, headers=headers, timeout=10)
    r.raise_for_status()
    return [(d["display_name"], float(d["lat"]), float(d["lon"])) for d in r.json()]


@st.cache_data(show_spinner=False, ttl=86400)
def wayback_releases():
    """Optional sharper dated imagery: list of (release_date, leaflet tile url) from Esri Wayback."""
    r = requests.get(WAYBACK_CONFIG, timeout=15)
    r.raise_for_status()
    out = []
    for val in r.json().values():
        if not isinstance(val, dict):
            continue
        m = re.search(r"(\d{4}-\d{2}-\d{2})", val.get("itemTitle", ""))
        url = val.get("itemURL", "")
        if m and url:
            url = url.replace("default028mm", "GoogleMapsCompatible").replace("{level}/{row}/{col}", "{z}/{y}/{x}")
            out.append((m.group(1), url))
    return sorted(set(out))


def nearest_release_index(options, year):
    target = date(year, 6, 30)
    gaps = [abs((date.fromisoformat(d) - target).days) for d in options]
    return gaps.index(min(gaps))


# ----------------------------------------------------------------------
# Raster analysis
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Reading satellite rasters...")
def analyze(lat, lon, radius_m, method):
    with rasterio.open(PATHS[method][2019]) as ref:
        r, c = ref.index(lon, lat)
        H, W = ref.height, ref.width
        res_x, res_y = ref.res
        tf = ref.transform
    if not (0 <= r < H and 0 <= c < W):
        raise ValueError("This location is outside the exported Lahore raster. "
                         "Pick a point inside the cyan Lahore boundary.")

    ny = res_y * 110574.0
    nx = res_x * 111320.0 * math.cos(math.radians(lat))
    px_km2 = ny * nx / 1e6
    hh, hw = max(2, int(radius_m / ny)), max(2, int(radius_m / nx))
    r0, r1 = max(0, r - hh), min(H, r + hh + 1)
    c0, c1 = max(0, c - hw), min(W, c + hw + 1)
    win = Window(c0, r0, c1 - c0, r1 - r0)

    def read(path, bands):
        with rasterio.open(path) as s:
            return s.read(bands, window=win)

    cls = {y: read(PATHS[method][y], 1) for y in YEARS}
    feat = {y: read(PATHS["feat"][y], [3, 2, 1, 7, 8]).astype("float32") for y in YEARS}  # R,G,B,NDVI,NDBI
    strata = read(PATHS["change"], 1)

    valid = (cls[2019] != NODATA) & (cls[2025] != NODATA)
    if not valid.any():
        raise ValueError("This window contains no Lahore pixels (outside the district boundary).")

    codes = cls[2019].astype(np.int16) * 3 + cls[2025].astype(np.int16)
    mat = np.bincount(codes[valid], minlength=9).reshape(3, 3) * px_km2     # rows=2019, cols=2025

    # ONE colour stretch for BOTH years, so differences you see are real, not display artefacts
    rgb_all = np.concatenate([feat[y][:3][:, valid].ravel() for y in YEARS])
    rgb_all = rgb_all[np.isfinite(rgb_all)]
    lo, hi = (float(v) for v in np.percentile(rgb_all, [2, 98])) if rgb_all.size else (0.02, 0.25)
    if hi <= lo:
        hi = lo + 0.01

    # evidence groups: do NDVI/NDBI behave the way real change should?
    f19, f25 = feat[2019], feat[2025]
    finite = np.isfinite(f19[3]) & np.isfinite(f25[3]) & np.isfinite(f19[4]) & np.isfinite(f25[4])

    def group(mask):
        m = mask & finite
        n = int(m.sum())
        if n == 0:
            return {"n": 0}
        return {"n": n, "km2": n * px_km2,
                "ndvi19": float(f19[3][m].mean()), "ndvi25": float(f25[3][m].mean()),
                "ndbi19": float(f19[4][m].mean()), "ndbi25": float(f25[4][m].mean())}

    groups = {
        CONV: group(valid & (cls[2019] == 0) & (cls[2025] == 1)),
        STAYV: group(valid & (cls[2019] == 0) & (cls[2025] == 0)),
        STAYB: group(valid & (cls[2019] == 1) & (cls[2025] == 1)),
    }
    cg = groups[CONV]
    vb_ndvi = (cg["ndvi25"] - cg["ndvi19"]) if cg["n"] else None
    vb_ndbi = (cg["ndbi25"] - cg["ndbi19"]) if cg["n"] else None

    return {
        "lat": lat, "lon": lon, "radius_m": radius_m, "method": method,
        "cls": cls, "feat": feat, "strata": strata, "valid": valid,
        "mat": mat, "px_km2": px_km2, "window_km2": float(valid.sum()) * px_km2,
        "bounds": win_bounds(win, tf),                       # left, bottom, right, top
        "stretch": (lo, hi), "groups": groups,
        "vb_dndvi": vb_ndvi, "vb_dndbi": vb_ndbi,
        "conf_km2": float(((strata >= 1) & (strata <= 3)).sum()) * px_km2,
        "conf_both_km2": float((strata == 1).sum()) * px_km2,
    }


@st.cache_data(show_spinner="Scanning the change layer for hotspots...")
def find_hotspots(top_n=10, cell_px=150):
    """Grid cells (~1.4 km) with the most area where BOTH methods say vegetation -> built-up."""
    with rasterio.open(PATHS["change"]) as s:
        arr, tf = s.read(1), s.transform
    H, W = arr.shape
    h2, w2 = H // cell_px * cell_px, W // cell_px * cell_px
    blocks = (arr[:h2, :w2] == 1).reshape(h2 // cell_px, cell_px, w2 // cell_px, cell_px).sum(axis=(1, 3))
    order = np.dstack(np.unravel_index(np.argsort(blocks, axis=None)[::-1], blocks.shape))[0]
    picked, out = [], []
    for i, j in order:
        if blocks[i, j] == 0 or len(out) >= top_n:
            break
        if any(max(abs(i - a), abs(j - b)) <= 1 for a, b in picked):
            continue
        picked.append((i, j))
        lon = tf.c + tf.a * (j * cell_px + cell_px / 2)
        lat = tf.f + tf.e * (i * cell_px + cell_px / 2)
        px = abs(tf.a) * 111320.0 * math.cos(math.radians(lat)) * abs(tf.e) * 110574.0 / 1e6
        out.append((float(lat), float(lon), float(blocks[i, j]) * px))
    return out


def evidence_check(res):
    """Do NDVI/NDBI on predicted-conversion pixels behave like real change? Returns (level, text)."""
    g = res["groups"]
    conv, sv = g[CONV], g[STAYV]
    if conv["n"] < 30:
        return "unknown", ("Fewer than 30 pixels are predicted as vegetation -> built-up in this window, so there "
                           "is too little to judge. Try a bigger radius or a model-flagged hotspot.")
    dn, db = conv["ndvi25"] - conv["ndvi19"], conv["ndbi25"] - conv["ndbi19"]
    if sv["n"] >= 30:
        dn0, db0 = sv["ndvi25"] - sv["ndvi19"], sv["ndbi25"] - sv["ndbi19"]
    else:
        dn0 = db0 = 0.0
    gap_n, gap_b = dn - dn0, db - db0
    detail = (f"On predicted-conversion pixels NDVI changed by {dn:+.2f} and NDBI by {db:+.2f}; on land that "
              f"stayed vegetation NDVI changed by {dn0:+.2f} and NDBI by {db0:+.2f} (that is the season/noise level).")
    if gap_n < -0.15 and gap_b > 0.05:
        return "strong", "Evidence is STRONG: greenness dropped and the built-up signal rose far more than seasonal noise. " + detail
    if gap_n < -0.08 and gap_b > 0.02:
        return "moderate", "Evidence is MODERATE: the direction is right but the size is not large. " + detail
    return "weak", ("Evidence is WEAK: the indices barely differ from normal seasonal noise, so a good part of the "
                    "predicted conversion here may be false. " + detail)


def groups_table(res):
    rows = []
    for name, g in res["groups"].items():
        if not g["n"]:
            rows.append({"Group": name, "Area (km2)": 0.0})
            continue
        rows.append({
            "Group": name, "Area (km2)": round(g["km2"], 3),
            "NDVI 2019": round(g["ndvi19"], 3), "NDVI 2025": round(g["ndvi25"], 3),
            "NDVI change": round(g["ndvi25"] - g["ndvi19"], 3),
            "NDBI 2019": round(g["ndbi19"], 3), "NDBI 2025": round(g["ndbi25"], 3),
            "NDBI change": round(g["ndbi25"] - g["ndbi19"], 3),
        })
    return pd.DataFrame(rows)


def category_table(res):
    mat, win = res["mat"], res["window_km2"]
    a19, a25 = mat.sum(axis=1), mat.sum(axis=0)
    rows = []
    for k in (0, 1, 2):
        d = a25[k] - a19[k]
        rows.append({
            "Category": SHORT[k],
            "2019 (km2)": round(a19[k], 3), "2025 (km2)": round(a25[k], 3), "Change (km2)": round(d, 3),
            "2019 (% of window)": round(100 * a19[k] / win, 1),
            "2025 (% of window)": round(100 * a25[k] / win, 1),
            "Change (pct points)": round(100 * d / win, 1),
        })
    return pd.DataFrame(rows)


def final_table(res_rf, res_dw):
    """2019 vs 2025 percentages for the selected area, both methods side by side."""
    rows = []
    for k in (0, 1, 2):
        row = {"Category": SHORT[k]}
        for tag, res in (("Random Forest", res_rf), ("Dynamic World", res_dw)):
            a19, a25 = res["mat"].sum(axis=1)[k], res["mat"].sum(axis=0)[k]
            w = res["window_km2"]
            row[f"{tag} 2019 (%)"] = round(100 * a19 / w, 1)
            row[f"{tag} 2025 (%)"] = round(100 * a25 / w, 1)
            row[f"{tag} change (pts)"] = round(100 * (a25 - a19) / w, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def transition_table(res):
    mat, win = res["mat"], res["window_km2"]
    rows = []
    for a in (0, 1, 2):
        for b in (0, 1, 2):
            if a != b:
                rows.append({"From (2019)": SHORT[a], "To (2025)": SHORT[b],
                             "Area (km2)": round(mat[a, b], 3), "% of window": round(100 * mat[a, b] / win, 2),
                             "Meaning": TRANSITION_MEANING[(a, b)]})
    return pd.DataFrame(rows).sort_values("Area (km2)", ascending=False).reset_index(drop=True)


def summarize(res):
    mat, win = res["mat"], res["window_km2"]
    a19, a25 = mat.sum(axis=1), mat.sum(axis=0)
    pct = lambda x: 100 * x / win
    side = 2 * res["radius_m"] / 1000
    L = [f"Inside the selected window (about {side:.1f} km across, {win:.2f} km2 of Lahore), estimated land cover "
         f"changed from {pct(a19[0]):.0f}% vegetation / {pct(a19[1]):.0f}% built-up / {pct(a19[2]):.0f}% other in 2019 "
         f"to {pct(a25[0]):.0f}% / {pct(a25[1]):.0f}% / {pct(a25[2]):.0f}% in 2025."]

    def move(k, word):
        d = a25[k] - a19[k]
        return f"{word} {'rose' if d >= 0 else 'fell'} by {abs(d):.2f} km2 ({abs(pct(d)):.1f} points)"
    L.append("Net change: " + "; ".join([move(1, "built-up area"), move(0, "vegetation/agriculture"),
                                          move(2, "other land")]) + ".")

    off = sorted(((mat[a, b], a, b) for a in range(3) for b in range(3) if a != b and mat[a, b] >= 0.01), reverse=True)
    changed = mat.sum() - np.trace(mat)
    nm = lambda k: SHORT[k].split(" /")[0].split(" (")[0]
    if pct(changed) < 1:
        L.append("Very little change was detected here (under 1% of the window changed class).")
    elif off:
        L.append("Largest estimated shifts: " + "; ".join(
            f"{nm(a)} -> {nm(b)}: {v:.2f} km2 ({pct(v):.1f}%)" for v, a, b in off[:3]) + ".")

    vb = mat[0, 1]
    if vb >= 0.01:
        L.append(f"Possible agricultural-to-urban conversion: about {vb:.2f} km2 ({pct(vb):.1f}% of the window) "
                 f"was vegetation in 2019 and is classed built-up in 2025 (see the evidence check below).")
    else:
        L.append("No meaningful vegetation -> built-up conversion was detected in this window.")
    if vb > 0 and mat[1, 0] > 0.3 * vb:
        L.append(f"Caution: the maps also show {mat[1, 0]:.2f} km2 of built-up -> vegetation, which is rarely real over "
                 f"six years. It signals classification noise of a similar order, so treat the conversion figure as an upper estimate.")
    L.append(f"A stricter layer (NDVI must drop and NDBI must rise) flags {res['conf_km2']:.2f} km2 here, of which "
             f"{res['conf_both_km2']:.2f} km2 is agreed by both Random Forest and Dynamic World.")
    L.append("All figures are estimated/possible change from 10 m Sentinel-2 data and have not been field-verified.")
    return L


# ----------------------------------------------------------------------
# Map layers
# ----------------------------------------------------------------------
def _hex(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


def class_rgba(arr):
    out = np.zeros(arr.shape + (4,), np.uint8)
    for k, h in COLORS.items():
        m = arr == k
        out[m, :3] = _hex(h)
        out[m, 3] = 255
    return out


def rgb_rgba(feat, valid, stretch):
    lo, hi = stretch
    img = np.moveaxis(feat[:3], 0, -1)
    finite = np.isfinite(img).all(-1)
    img = np.clip((np.nan_to_num(img) - lo) / (hi - lo), 0, 1) ** 0.9
    out = np.zeros(img.shape[:2] + (4,), np.uint8)
    out[..., :3] = (img * 255).astype(np.uint8)
    out[..., 3] = np.where(valid & finite, 255, 0)
    return out


def change_rgba(strata):
    out = np.zeros(strata.shape + (4,), np.uint8)
    out[strata == 1] = [255, 0, 0, 255]
    out[(strata == 2) | (strata == 3)] = [255, 140, 0, 255]
    return out


def zoom_for(radius_m, lat):
    z = math.log2(156543.03 * math.cos(math.radians(lat)) * 500 / (2 * max(radius_m, 100)))
    return int(max(10, min(18, round(z))))


def build_dual(res, view, opacity, show_change, labels, label, photos, photo_dates, smooth):
    """Left map = 2019, right map = 2025. Zoom goes to level 22.

    photos: {year: tile url}  - high-resolution photo base layer for each map (dated Wayback or current Esri)
    """
    lat, lon = res["lat"], res["lon"]
    left, bottom, right, top = res["bounds"]
    bnds = [[bottom, left], [top, right]]
    dual = DualMap(location=[lat, lon], zoom_start=zoom_for(res["radius_m"], lat), tiles=None, layout="horizontal")
    for m, y in ((dual.m1, 2019), (dual.m2, 2025)):
        dated = photo_dates.get(y)
        folium.TileLayer(
            photos.get(y, ESRI_SAT), attr="Esri World Imagery" + (" Wayback" if dated else ""),
            max_zoom=MAX_ZOOM, max_native_zoom=18 if dated else 19, overlay=False,
            name=f"High-res photo ({dated})" if dated else "High-res photo (Esri, current)",
            show=(view != VIEW_S2)).add_to(m)
        folium.TileLayer(ESRI_GRAY, attr="Esri, HERE, Garmin", max_zoom=MAX_ZOOM, max_native_zoom=16,
                         name="Plain grey map (no photo)", overlay=False, show=(view == VIEW_S2)).add_to(m)
        folium.raster_layers.ImageOverlay(
            image=rgb_rgba(res["feat"][y], res["valid"], res["stretch"]), bounds=bnds, opacity=1.0,
            pixelated=not smooth, name=f"Sentinel-2 true colour {y} (10 m)",
            show=(view == VIEW_S2)).add_to(m)
        folium.raster_layers.ImageOverlay(
            image=class_rgba(res["cls"][y]), bounds=bnds, opacity=opacity, pixelated=True,
            name=f"Land-cover classes {y}", show=(view == VIEW_CLASSES)).add_to(m)
        if y == 2025:
            folium.raster_layers.ImageOverlay(
                image=change_rgba(res["strata"]), bounds=bnds, opacity=0.9, pixelated=True,
                name="Detected change (red = both methods)", show=show_change).add_to(m)
        if labels:
            folium.TileLayer(ESRI_LABELS, attr="Esri", max_zoom=MAX_ZOOM, max_native_zoom=19,
                             name="Place names", overlay=True, show=True).add_to(m)
        folium.CircleMarker([lat, lon], radius=6, color="white", weight=2, fill=True,
                            fill_color="#00e5ff", fill_opacity=1, tooltip=label).add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
    return dual


def build_pick_map(lat, lon, radius_m, labels):
    m = folium.Map(location=[lat, lon], zoom_start=12, tiles=None, control_scale=True)
    folium.TileLayer(ESRI_SAT, attr=ESRI_ATTR, name="Satellite", max_zoom=MAX_ZOOM, max_native_zoom=19).add_to(m)
    if labels:
        folium.TileLayer(ESRI_LABELS, attr="Esri", name="Place names", overlay=True,
                         max_zoom=MAX_ZOOM, max_native_zoom=19).add_to(m)
    if BOUNDARY_PATH.exists():
        folium.GeoJson(str(BOUNDARY_PATH), name="Lahore district",
                       style_function=lambda f: {"fill": False, "color": "cyan", "weight": 2}).add_to(m)
    dlat = radius_m / 110574.0
    dlon = radius_m / (111320.0 * math.cos(math.radians(lat)))
    folium.Rectangle([[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]],
                     color="#ffeb3b", weight=2, fill=False, tooltip="Analysis window").add_to(m)
    folium.Marker([lat, lon], tooltip="Selected location").add_to(m)
    return m


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Punjab GeoGuardian", layout="wide")
    ss = st.session_state
    ss.setdefault("loc", (31.4674, 74.1928))
    ss.setdefault("label", "Hotspot H1 (default)")
    ss.setdefault("last_click", None)
    ss.setdefault("cands", [])
    ss.setdefault("hot", [])

    st.title("Punjab GeoGuardian - Lahore land-change explorer (2019 vs 2025)")
    st.caption("Learning prototype, not an official monitoring system. Results are estimated/possible "
               "change from Sentinel-2 and are not field-verified.")

    # ---------------- sidebar ----------------
    with st.sidebar:
        st.header("Find a location")
        with st.form("search_form"):
            q = st.text_input("Place name or coordinates",
                              placeholder="DHA Phase 9 Prism, Lahore   or   31.4674, 74.1928")
            go = st.form_submit_button("Search")
        if go and q.strip():
            coords = parse_coords(q)
            if coords:
                ss.loc, ss.label, ss.cands = coords, f"{coords[0]:.5f}, {coords[1]:.5f}", []
            else:
                try:
                    found = geocode(q.strip())
                except Exception as e:  # noqa: BLE001
                    found = []
                    st.error(f"Place search failed (internet needed): {e}")
                if found:
                    ss.cands = found
                    ss.cand_idx = 0
                    ss.loc, ss.label = (found[0][1], found[0][2]), found[0][0].split(",")[0]
                elif not ss.cands:
                    st.warning("No match found. Try adding ', Lahore' or paste coordinates.")

        def _use_candidate():
            name, la, lo = ss.cands[ss.cand_idx]
            ss.loc, ss.label = (la, lo), name.split(",")[0]

        if len(ss.cands) > 1:
            st.selectbox("Other matches", list(range(len(ss.cands))),
                         format_func=lambda i: ss.cands[i][0][:70], key="cand_idx", on_change=_use_candidate)

        with st.expander("Jump to a model-flagged hotspot"):
            st.caption("Places where BOTH Random Forest and Dynamic World say vegetation became built-up. "
                       "Good places to test whether the prediction is real.")
            if st.button("Find top hotspots"):
                try:
                    ss.hot = find_hotspots()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not scan the change layer: {e}")

            def _use_hot():
                la, lo, _ = ss.hot[ss.hot_idx]
                ss.loc, ss.label, ss.cands = (la, lo), f"Hotspot #{ss.hot_idx + 1}", []

            if ss.hot:
                st.selectbox("Hotspots", list(range(len(ss.hot))), key="hot_idx", on_change=_use_hot,
                             format_func=lambda i: f"#{i + 1}: {ss.hot[i][0]:.4f}, {ss.hot[i][1]:.4f} "
                                                   f"({ss.hot[i][2]:.2f} km2 changed)")

        # Esri Wayback list (dated high-resolution photos) - fetched once per session
        if "rel" not in ss:
            try:
                ss.rel, ss.wb_err = wayback_releases(), ""
            except Exception as e:  # noqa: BLE001
                ss.rel, ss.wb_err = [], str(e)
        wb_ok = bool(ss.rel)

        st.header("Settings")
        radius = st.slider("Analysis window radius (m)", 500, 6000, 1500, step=250)
        method_label = st.radio("Classification source", list(METHODS))
        method = METHODS[method_label]
        view = st.radio("What to show on both maps", [VIEW_PHOTO, VIEW_S2, VIEW_CLASSES],
                        index=0 if wb_ok else 1)
        opacity = st.slider("Class overlay opacity", 0.2, 1.0, 0.6, step=0.05)
        smooth = st.checkbox("Smooth Sentinel-2 pixels (nicer when zoomed in)", value=True)
        show_change = st.checkbox("Highlight detected change on the 2025 map", value=False)
        labels = st.checkbox("Show place names", value=True)
        map_h = st.slider("Map height (px)", 500, 1000, 720, step=20)

        photos, photo_dates = {y: ESRI_SAT for y in YEARS}, {}
        with st.expander("High-res photo dates (Esri Wayback)", expanded=not wb_ok):
            if wb_ok:
                lookup = dict(ss.rel)
                for y in YEARS:
                    opts = [d for d, _ in ss.rel if d.startswith(str(y))] or [d for d, _ in ss.rel]
                    sel = st.selectbox(f"{y} photo release", opts, index=nearest_release_index(opts, y), key=f"wb{y}")
                    photos[y], photo_dates[y] = lookup[sel], sel
                st.caption("A release date is when Esri published the mosaic, not the day the photo was taken, and "
                           "spots with no new photo look the same across releases.")
            else:
                st.warning("Could not load dated photos (internet needed), so both maps fall back to Esri's current "
                           "photo, which is identical for 2019 and 2025. Use the Sentinel-2 view instead. "
                           f"{ss.get('wb_err', '')}")
                if st.button("Retry loading dated photos"):
                    del ss["rel"]
                    st.rerun()
        st.caption("You can also click directly on the first map to choose a location.")

    # ---------------- 1. pick map ----------------
    lat, lon = ss.loc
    st.subheader("1. Choose a location")
    st.write(f"**Selected:** {ss.label}  |  {lat:.5f}, {lon:.5f}  |  yellow box = analysis window")
    pick = build_pick_map(lat, lon, radius, labels)
    try:
        out = st_folium(pick, height=400, use_container_width=True, returned_objects=["last_clicked"], key="pick_map")
    except TypeError:
        out = st_folium(pick, height=400, returned_objects=["last_clicked"], key="pick_map")
    click = out.get("last_clicked") if out else None
    if click and click != ss.last_click:
        ss.last_click = click
        ss.loc, ss.label, ss.cands = (click["lat"], click["lng"]), "Clicked point", []
        st.rerun()

    # ---------------- analysis ----------------
    try:
        res = analyze(round(lat, 6), round(lon, 6), radius, method)
        res_rf = res if method == "RF3" else analyze(round(lat, 6), round(lon, 6), radius, "RF3")
        res_dw = res if method == "DW3" else analyze(round(lat, 6), round(lon, 6), radius, "DW3")
    except FileNotFoundError as e:
        st.error(f"A raster file was not found: {e}\n\nCheck the PATHS block at the top of this file.")
        st.stop()
    except ValueError as e:
        st.warning(str(e))
        st.stop()

    # ---------------- 2. maps ----------------
    st.subheader("2. 2019 vs 2025 (maps pan and zoom together, zoom up to level 22)")
    c1, c2 = st.columns(2)
    c1.markdown("### 2019")
    c2.markdown("### 2025")
    dual = build_dual(res, view, opacity, show_change, labels, ss.label, photos, photo_dates, smooth)
    map_html = dual.get_root().render()
    if hasattr(st, "iframe"):
        st.iframe(map_html, height=map_h)
    else:
        components.html(map_html, height=map_h)
    st.markdown(
        " &nbsp; ".join(f"<span style='display:inline-block;width:14px;height:14px;background:{COLORS[k]};"
                        f"border-radius:3px;vertical-align:middle'></span> {SHORT[k]}" for k in (0, 1, 2))
        + " &nbsp; <span style='display:inline-block;width:14px;height:14px;background:#ff0000;"
          "border-radius:3px;vertical-align:middle'></span> Detected change (both methods)",
        unsafe_allow_html=True)
    st.link_button("Open this spot in Google Maps (satellite)",
                   f"https://www.google.com/maps/@?api=1&map_action=map&center={lat},{lon}&zoom=18&basemap=satellite")
    st.caption("Quality guide: the high-res photo (about 0.3-0.6 m per pixel) stays sharp to zoom 18-19 and is "
               "enlarged beyond that. Sentinel-2 is 10 m per pixel, which is the real resolution the classifier "
               "used, so it can never look sharper than a coarse mosaic. Use the photo view to judge what is on the "
               "ground and the Sentinel-2 view to see what the model saw. Use each map's layer panel to switch layers.")

    # ---------------- 3. what changed ----------------
    st.subheader("3. What changed between 2019 and 2025")
    m1, m2, m3 = st.columns(3)
    m1.metric("Window area", f"{res['window_km2']:.2f} km2")
    m2.metric("Vegetation -> Built-up", f"{res['mat'][0, 1]:.2f} km2",
              f"{100 * res['mat'][0, 1] / res['window_km2']:.1f}% of window", delta_color="inverse")
    net_b = res["mat"].sum(axis=0)[1] - res["mat"].sum(axis=1)[1]
    m3.metric("Net built-up change", f"{net_b:+.2f} km2")
    for line in summarize(res):
        st.markdown(f"- {line}")

    st.markdown("**Which categories changed into which (2019 -> 2025)**")
    trans = transition_table(res)
    st.dataframe(trans, width="stretch", hide_index=True)
    st.download_button("Download change table (CSV)", trans.to_csv(index=False).encode("utf-8"),
                       file_name="geoguardian_change_table.csv", mime="text/csv")

    # ---------------- 4. is the prediction right? ----------------
    st.subheader("4. Is the prediction right? (evidence check)")
    level, text = evidence_check(res)
    {"strong": st.success, "moderate": st.info, "weak": st.warning, "unknown": st.info}[level](text)
    st.dataframe(groups_table(res), width="stretch", hide_index=True)
    st.caption("How to read this: real conversion = NDVI falls and NDBI rises on those pixels by clearly more than on "
               "the 'stayed vegetation' control. This test uses the same indices the classifier saw, so it is a "
               "plausibility check, not ground truth. The final proof is visual: compare the two Sentinel-2 maps, "
               "then check sharper imagery (Google Earth Pro > View > Historical Imagery).")

    # ---------------- 5. final percentage table ----------------
    st.subheader("5. Summary table for the selected area: 2019 vs 2025")
    st.write(f"**Area:** {ss.label} | window about {2 * radius / 1000:.1f} km across "
             f"({res['window_km2']:.2f} km2 inside Lahore district)")
    ft = final_table(res_rf, res_dw)
    st.dataframe(ft, width="stretch", hide_index=True)
    sel = res
    a19, a25 = sel["mat"].sum(axis=1), sel["mat"].sum(axis=0)
    w = sel["window_km2"]
    st.success(f"{method_label.split(' (')[0]}: in 2019 this area was {100 * a19[1] / w:.1f}% built-up and "
               f"{100 * a19[0] / w:.1f}% vegetation/agriculture. In 2025 it is {100 * a25[1] / w:.1f}% built-up and "
               f"{100 * a25[0] / w:.1f}% vegetation/agriculture.")
    st.download_button("Download summary table (CSV)", ft.to_csv(index=False).encode("utf-8"),
                       file_name="geoguardian_summary_table.csv", mime="text/csv")
    st.caption("Random Forest is trained on Dynamic World labels, so their agreement is not independent accuracy. "
               "Percentages are of the pixels inside the Lahore district within the window.")


if __name__ == "__main__":
    main()
