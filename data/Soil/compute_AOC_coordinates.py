import geopandas as gpd

# === Define task list ===
steps = [
    "Loading shapefile",
    "Filtering AOC wines",
    "Selecting relevant columns",
    "Dissolving polygons by AOC name",
    "Simplifying geometries",
    "Exporting to GeoJSON"
]

    # 1. Load shapefile
shapefile_path = "data/Soil/2025_06_10_soil_data.shp"
gdf = gpd.read_file(shapefile_path)

# 2. Filter AOC wines
aoc_only = gdf[gdf["signe"] == "AOC"]

# 3. Select relevant columns
aoc_clean = aoc_only[["app", "type_prod", "categorie", "geometry"]]

# 4. Dissolve polygons
aoc_dissolved = aoc_clean.dissolve(by="app", as_index=False)

# 5. Simplify geometries with inner progress
aoc_dissolved["geometry"] = aoc_dissolved["geometry"].progress_apply(
    lambda geom: geom.simplify(tolerance=0.001, preserve_topology=True)
)

# 6. Export result
output_path = "data/Soil/aoc_polygons.geojson"
aoc_dissolved.to_file(output_path, driver="GeoJSON")

print(f"Saved dissolved AOC polygons to: {output_path}")
