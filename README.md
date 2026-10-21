# Punjab GeoGuardian

**AI-based monitoring of urban expansion and farmland loss in Lahore District (2019 vs 2025) using Sentinel-2 satellite imagery.**

Built for *AIRI Team PITB - AI Internship Task 3*. This is a **learning prototype**, not an official government monitoring system. All results are *estimated* or *possible* change and have **not** been field-verified.

![Detected change](images/09_Change_detection.png)

---

## What this project does

The project compares satellite imagery of Lahore District from two years and estimates where vegetation or agricultural land may have turned into built-up land.

1. Select the area of interest (Lahore District, FAO GAUL 2025 boundary).
2. Load Sentinel-2 imagery for 2019 and 2025 (B2, B3, B4, B8, B11, B12, NDVI, NDBI).
3. Convert Google Dynamic World labels (9 classes) into 3 classes and train a Random Forest on Sentinel-2 bands.
4. Classify both years: **Vegetation / Agriculture**, **Built-up / Urban**, **Other**.
5. Detect change (Vegetation in 2019 to Built-up in 2025), confirm it with NDVI/NDBI change, and remove tiny patches.
6. Calculate the changed area in km2 using true pixel area.
7. Visualise everything in maps, an interactive map and a Streamlit web app.

## Key results (unvalidated)

Lahore District area: **1,691.1 km2**

| Class | Dynamic World 2019 | Dynamic World 2025 | Random Forest 2019 | Random Forest 2025 |
|---|---|---|---|---|
| Vegetation / Agriculture | 950.4 km2 (56.2%) | 852.6 km2 (50.4%) | 842.5 km2 (49.8%) | 789.6 km2 (46.7%) |
| Built-up / Urban | 645.5 km2 (38.2%) | 782.1 km2 (46.2%) | 631.9 km2 (37.4%) | 727.4 km2 (43.0%) |
| Other | 95.3 km2 (5.6%) | 56.5 km2 (3.3%) | 216.7 km2 (12.8%) | 174.1 km2 (10.3%) |

**Possible vegetation to built-up conversion, 2019-2025:**

| Estimate | Area | Share of Lahore |
|---|---|---|
| Conservative (both methods agree) | 28.7 km2 | 1.70% |
| Upper bound (either method flags it) | 112.4 km2 | 6.65% |

After NDVI/NDBI confirmation and small-patch removal: Dynamic World 55.2 km2 (3.26%), Random Forest 85.9 km2 (5.08%).

The notebook also lists the top 15 model-flagged change hotspots (latitude/longitude and area). These show *detected change*, not proof of any specific development.

## Accuracy and validation (please read)

| Measure | Value | Meaning |
|---|---|---|
| Random Forest vs Dynamic World, spatial hold-out | 79.2% | Agreement with Dynamic World labels on areas the model never saw |
| Random Forest vs Dynamic World, whole district | 78.8% (2019), 78.6% (2025) | Pixel agreement over about 19.9 million pixels |
| Real-world accuracy | **not measured** | No manually checked reference points have been labelled yet |

Dynamic World is itself a model-made product derived from Sentinel-2, and the Random Forest is trained to imitate it. The 79% figure is therefore **agreement with a model, not ground-truth accuracy**. Validation (50+ manually checked points) and error analysis (15+ cases) are listed under *Status and next steps*.

Known weaknesses: Built-up is the weakest class and is often confused with Other; Dynamic World shows some Built-up to Vegetation flips that should be near zero in reality (label noise); dry or fallow fields can look like built-up; and 10 m Sentinel-2 pixels cannot show single houses.

## Data

| Item | Detail |
|---|---|
| Area of interest | Lahore District, Punjab, Pakistan (FAO GAUL 2025, level 2) |
| Imagery | Sentinel-2 Surface Reflectance Harmonized, one 8-band GeoTIFF per year |
| Labels | Google Dynamic World, one GeoTIFF per year |
| Grid | EPSG:4326, 7233 x 5129 pixels, about 10 m x 8.5 m (about 85 m2 per pixel) |
| Composite details | **[FILL IN]** date range: ____ , cloud filter: ____ , composite method: ____ |

The raster files are large (about 818 MB each for Sentinel-2) and are **not stored in this repository**. Export them from Google Earth Engine and place them as shown below.

## Repository structure

```
Punjab-GeoGuardian/
├── PunjabGeo.ipynb              # full pipeline: data checks, classification, change detection, maps
├── geoguardian_app.py           # Streamlit web app (search, 2019 vs 2025 maps, change tables)
├── check_app.py                 # automatic checks that the app's numbers are correct
├── requirements.txt
├── README.md
├── images/                      # maps used in this README
├── data/
│   ├── raw/
│   │   ├── Sentinel2_2019_Features.tif
│   │   ├── Sentinel2_2025_Features.tif
│   │   ├── DynamicWorld_2019_Labels.tif
│   │   └── DynamicWorld_2025_Labels.tif
│   └── boundaries/Lahore_District.geojson
└── outputs/
    ├── classifications/         # DW3_*.tif, RF3_*.tif
    ├── change_detection/        # change_strata_2019_2025.tif, change_polygons.geojson
    ├── maps/                    # the 9 required maps (PNG)
    ├── report/                  # area statistics and other CSV tables
    ├── validation/              # validation point files and blind map
    ├── interactive/             # GeoGuardian_interactive_map.html
    └── models/                  # RF3_model.joblib
```

## Method in brief

**Indices**

- NDVI = (NIR - Red) / (NIR + Red), using B8 and B4. High values mean vegetation.
- NDBI = (SWIR - NIR) / (SWIR + NIR), using B11 and B8. High values mean built-up surfaces.

**Joining Sentinel-2 with Dynamic World labels.** Both products are exported on the same grid, so the join is pixel by pixel (same row and column). Dynamic World's 9 classes are reduced to 3:

| Dynamic World | Project class |
|---|---|
| Trees, Grass, Flooded vegetation, Crops, Shrub and scrub | 0 Vegetation / Agriculture |
| Built-up | 1 Built-up / Urban |
| Water, Bare ground, Snow and ice | 2 Other |

Only "pure" pixels (all 3 x 3 neighbours share the label) inside Lahore are sampled: 5,000 per class per year, 30,000 training rows with 8 features each. A Random Forest (120 trees) is evaluated on whole spatial blocks held out from training, then trained on all samples and applied to every pixel.

**Change detection.** A pixel is a candidate if it is Vegetation in 2019 and Built-up in 2025. It is confirmed only if NDVI drops by at least 0.05 and NDBI rises by at least 0.056, and patches under 20 pixels (about 0.17 ha) are removed. Results are stored as strata: both methods agree, Dynamic World only, Random Forest only.

**Area.** The grid is in degrees, so pixel area is computed per row (about 85 m2), not assumed to be 100 m2.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/<your-username>/Punjab-GeoGuardian.git
cd Punjab-GeoGuardian
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install streamlit streamlit-folium requests
```

The notebook lists the packages it used in `requirements.txt`. Add `streamlit`, `streamlit-folium` and `requests` if they are not in that file.

## Usage

### 1. Run the notebook

Open `PunjabGeo.ipynb`, set `PROJECT_DIR` in the configuration cell to your local folder, place the four raster files in `data/raw/`, and run the cells from the configuration cell (Cell 1) onward. The earliest cells are exploratory diagnostics; the final clean pipeline starts at the cell titled *CELL 1 - CONFIG + HELPERS*. Earth Engine is only needed to download the Lahore boundary and to export the rasters (`earthengine authenticate`).

### 2. Run the web app

```bash
streamlit run geoguardian_app.py
```

The app opens at `http://localhost:8501`. If your project is not in `E:\Punjab GeoGuardian`, set the folder first:

```bash
# Windows PowerShell
$env:GEOGUARDIAN_DIR = "C:\path\to\Punjab-GeoGuardian"
# macOS / Linux
export GEOGUARDIAN_DIR=/path/to/Punjab-GeoGuardian
```

**Input (any one):** a place name (for example *Raiwind Road, Lahore*), coordinates as `latitude, longitude`, a click on the map, or a model-flagged hotspot from the sidebar.

**Output:**

- two synchronised maps, 2019 and 2025, with zoom up to level 22 (dated high-resolution photos, Sentinel-2 true colour, or land-cover classes);
- a written summary of what changed and a table of which category became which (in km2);
- an evidence check showing whether NDVI and NDBI behave like real change on the predicted-conversion pixels;
- a table of the percentage of Vegetation, Built-up and Other in 2019 vs 2025 for the selected area.

The app does not retrain anything. It reads the classified rasters created by the notebook. It works for Lahore District only, and place search and dated photos need an internet connection.

### 3. Check the numbers

```bash
python check_app.py 31.4674 74.1928 1500   # recompute one window and compare with the app
python check_app.py --district             # compare saved rasters with the notebook totals
```

These checks prove the app adds up the data correctly. They do not prove the land-cover maps match the ground.

## Status and next steps

- [x] Sentinel-2 data for two years, boundary, masking and quality checks
- [x] NDVI and NDBI, 3-class classification, change detection, area statistics
- [x] 9 required maps, interactive map, change polygons (GeoJSON), Streamlit app
- [ ] Validation with 50+ checked points (reference points are generated; none labelled yet)
- [ ] Error analysis of 15+ weak or incorrect cases
- [ ] Record the Earth Engine export details (dates, cloud filter, composite method)

Ideas for improvement: same-season composites for both years, more and manually drawn training samples, extra classes (water, bare land), higher-resolution imagery for hotspots, Sentinel-1 radar or multi-year composites, a U-Net segmentation model, validation against official housing-scheme boundaries, and more districts.

## Responsible use

Use the words *possible conversion*, *detected change* and *estimated change*. Do not use these results to make legal claims about any specific housing society or private development. Accuracy depends on image quality, date selection, cloud cover, thresholds and validation quality.

## Tools and technologies

Python, Google Earth Engine, Sentinel-2, Google Dynamic World, rasterio, GeoPandas, scikit-learn (Random Forest), NumPy, pandas, Matplotlib, Folium, Streamlit.

## Author

**Adeel** - AIRI Team PITB AI Internship, Task 3.

[Add your LinkedIn link here]

## License

[Choose a license, for example MIT, and add a LICENSE file.]

## Acknowledgements

Contains modified Copernicus Sentinel data (Sentinel-2), Google Dynamic World, FAO GAUL administrative boundaries, Esri World Imagery, and OpenStreetMap data (place search).
