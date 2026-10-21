"""
Punjab GeoGuardian - location lookup + Streamlit dashboard

Two ways to use this file
-------------------------
1) Dashboard (browser):
       pip install streamlit
       streamlit run app.py

2) Inside your notebook:
       from app import predict
       res = predict(31.4674, 74.1928, radius_m=1000)
       res["table"]

What it does
------------
You give a location (latitude, longitude) and a radius. It looks up your
ALREADY-CLASSIFIED rasters (Random Forest + Dynamic World, 2019 and 2025),
the NDVI/NDBI bands and the change layer, and returns:
  - predicted class at the point for both years
  - NDVI / NDBI at the point for both years
  - whether the point is flagged as possible Vegetation -> Built-up change
  - class areas and changed area inside the surrounding window

It does NOT forecast the future, and it only works inside the Lahore raster.
"""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

# ----------------------------------------------------------------------
# CONFIG - edit PROJECT_DIR if your folder is somewhere else
# ----------------------------------------------------------------------
PROJECT_DIR = Path(r"E:\Punjab GeoGuardian")
OUT = PROJECT_DIR / "outputs"
RAW = PROJECT_DIR / "data" / "raw"
YEARS = (2019, 2025)

PATHS = {
    "RF3": {y: OUT / "classifications" / f"RF3_{y}.tif" for y in YEARS},
    "DW3": {y: OUT / "classifications" / f"DW3_{y}.tif" for y in YEARS},
    "feat": {y: RAW / f"Sentinel2_{y}_Features.tif" for y in YEARS},
    "change": OUT / "change_detection" / "change_strata_2019_2025.tif",
}

CLASS_NAMES = {0: "Vegetation / Agriculture", 1: "Built-up / Urban", 2: "Other"}
CLASS_COLORS = ["#2e8b57", "#d73027", "#e0c068"]
STRATA_NAMES = {
    0: "No change detected",
    1: "Possible conversion (both methods agree)",
    2: "Possible conversion (Dynamic World only)",
    3: "Possible conversion (Random Forest only)",
}
NODATA = 255
# Feature band order in the Sentinel-2 export
B_RED, B_GREEN, B_BLUE, B_NDVI, B_NDBI = 3, 2, 1, 7, 8

HOTSPOTS = {  # from your Cell 5 hotspot table
    "H1": (31.4674, 74.1928),
    "H2": (31.3477, 74.3524),
    "H3": (31.4275, 74.1530),
    "H4": (31.5671, 74.5119),
    "H5": (31.6468, 74.4720),
}


def _name(cls_id):
    return CLASS_NAMES.get(int(cls_id), "Outside Lahore / no data")


def predict(lat, lon, radius_m=1000):
    """Look up predictions for a location. Returns a dict (see keys below)."""
    with rasterio.open(PATHS["RF3"][2019]) as ref:
        r, c = ref.index(lon, lat)
        H, W = ref.height, ref.width
        res_x, res_y = ref.res

    if not (0 <= r < H and 0 <= c < W):
        raise ValueError("That location is outside the exported Lahore raster.")

    # pixel size in metres (the grid is in degrees)
    ny = res_y * 110574.0
    nx = res_x * 111320.0 * np.cos(np.radians(lat))
    px_km2 = ny * nx / 1e6

    hh = max(1, int(radius_m / ny))
    hw = max(1, int(radius_m / nx))
    r0, r1 = max(0, r - hh), min(H, r + hh + 1)
    c0, c1 = max(0, c - hw), min(W, c + hw + 1)
    win = Window(c0, r0, c1 - c0, r1 - r0)
    pr, pc = r - r0, c - c0  # point position inside the window

    def read(path, bands):
        with rasterio.open(path) as s:
            return s.read(bands, window=win)

    rf = {y: read(PATHS["RF3"][y], 1) for y in YEARS}
    dw = {y: read(PATHS["DW3"][y], 1) for y in YEARS}
    feat = {y: read(PATHS["feat"][y], [B_BLUE, B_GREEN, B_RED, B_NDVI, B_NDBI]) for y in YEARS}
    strata = read(PATHS["change"], 1)

    if rf[2019][pr, pc] == NODATA:
        raise ValueError("That point is outside the Lahore district boundary.")

    # ---- point-level table ----
    rows = []
    for y in YEARS:
        rows.append({
            "Year": y,
            "Random Forest class": _name(rf[y][pr, pc]),
            "Dynamic World class": _name(dw[y][pr, pc]),
            "NDVI": round(float(feat[y][3][pr, pc]), 3),
            "NDBI": round(float(feat[y][4][pr, pc]), 3),
        })
    import pandas as pd
    table = pd.DataFrame(rows)

    s_val = int(strata[pr, pc])
    s_val = s_val if s_val in STRATA_NAMES else 0
    rf_change = bool(rf[2019][pr, pc] == 0 and rf[2025][pr, pc] == 1)
    dw_change = bool(dw[2019][pr, pc] == 0 and dw[2025][pr, pc] == 1)

    # ---- window-level area statistics ----
    valid = rf[2019] != NODATA
    area_rows = []
    for k, name in CLASS_NAMES.items():
        area_rows.append({
            "Class": name,
            "2019 (km2)": round(float(((rf[2019] == k) & valid).sum()) * px_km2, 3),
            "2025 (km2)": round(float(((rf[2025] == k) & valid).sum()) * px_km2, 3),
        })
    areas = pd.DataFrame(area_rows)
    changed_km2 = float(((strata >= 1) & (strata <= 3)).sum()) * px_km2
    window_km2 = float(valid.sum()) * px_km2

    return {
        "lat": lat, "lon": lon, "radius_m": radius_m,
        "table": table,
        "areas": areas,
        "change_stratum": STRATA_NAMES[s_val],
        "rf_change": rf_change,
        "dw_change": dw_change,
        "changed_km2": changed_km2,
        "window_km2": window_km2,
        "arrays": {"rf": rf, "dw": dw, "feat": feat, "strata": strata, "pr": pr, "pc": pc},
    }


def verdict(res):
    """One careful sentence - uses 'possible' / 'detected' wording as the task requires."""
    if res["change_stratum"].startswith("Possible"):
        return ("Detected change: this location is flagged as POSSIBLE conversion from "
                "vegetation/agriculture (2019) to built-up (2025) - " + res["change_stratum"].lower() + ".")
    return "No vegetation-to-built-up change was detected at this exact location."


def _rgb(feat_year):
    img = np.moveaxis(feat_year[:3][::-1], 0, -1)  # (B,G,R) -> (R,G,B)
    img = np.clip((img - 0.02) / (0.25 - 0.02), 0, 1)  # same fixed stretch for both years
    return np.nan_to_num(img)


def main():
    import matplotlib.pyplot as plt
    import streamlit as st
    from matplotlib.colors import ListedColormap

    st.set_page_config(page_title="Punjab GeoGuardian", layout="wide")
    st.title("Punjab GeoGuardian - Lahore land-change lookup")
    st.caption("Learning prototype, not an official monitoring system. Results are "
               "estimated/possible change from Sentinel-2 and are not field-verified.")

    with st.sidebar:
        st.header("Input")
        choice = st.selectbox("Quick pick", ["Custom"] + [f"Hotspot {k}" for k in HOTSPOTS])
        d_lat, d_lon = (31.4674, 74.1928)
        if choice != "Custom":
            d_lat, d_lon = HOTSPOTS[choice.split()[-1]]
        lat = st.number_input("Latitude", value=float(d_lat), format="%.6f")
        lon = st.number_input("Longitude", value=float(d_lon), format="%.6f")
        radius = st.slider("Window radius (m)", 250, 5000, 1000, step=250)
        st.markdown("Tip: right-click a spot in Google Maps and copy the "
                    "`lat, lon` it shows.")

    try:
        res = predict(lat, lon, radius)
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        st.stop()

    a = res["arrays"]
    st.subheader("Result at the selected point")
    st.info(verdict(res))
    st.dataframe(res["table"], use_container_width=True, hide_index=True)

    cmap_cls = ListedColormap(CLASS_COLORS)
    cmap_chg = ListedColormap(["#eeeeee", "#ff0000", "#ff8c00", "#ff8c00"])

    def show(ax, data, title, **kw):
        ax.imshow(data, **kw)
        ax.plot(a["pc"], a["pr"], marker="+", color="cyan", markersize=16, mew=2)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    show(axs[0, 0], _rgb(a["feat"][2019]), "RGB 2019")
    show(axs[0, 1], np.ma.masked_equal(a["rf"][2019], NODATA), "Classes 2019 (RF)",
         cmap=cmap_cls, vmin=0, vmax=2, interpolation="nearest")
    change_img = np.where(a["strata"] == NODATA, 0, a["strata"]).astype(np.uint8)
    show(axs[0, 2], change_img,
         "Detected change (red = both methods)", cmap=cmap_chg, vmin=0, vmax=3,
         interpolation="nearest")
    show(axs[1, 0], _rgb(a["feat"][2025]), "RGB 2025")
    show(axs[1, 1], np.ma.masked_equal(a["rf"][2025], NODATA), "Classes 2025 (RF)",
         cmap=cmap_cls, vmin=0, vmax=2, interpolation="nearest")
    dndvi = a["feat"][2025][3] - a["feat"][2019][3]
    show(axs[1, 2], dndvi, "NDVI change (2025 - 2019)", cmap="RdYlGn", vmin=-0.6, vmax=0.6)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Area statistics inside the window")
    st.write(f"Window area: {res['window_km2']:.2f} km2 | "
             f"possible conversion in window: {res['changed_km2']:.3f} km2")
    st.dataframe(res["areas"], use_container_width=True, hide_index=True)

    st.caption("Legend: green = Vegetation/Agriculture, red = Built-up, tan = Other. "
               "Cyan + marks your point. Random Forest is trained on Dynamic World labels, "
               "so its agreement with Dynamic World is not independent accuracy.")


if __name__ == "__main__":
    main()
