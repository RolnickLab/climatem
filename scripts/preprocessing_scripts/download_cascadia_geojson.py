import requests
from arcgis2geojson import arcgis2geojson

# URL of the ArcGIS FeatureServer layer
url = "https://services.arcgis.com/qboYD3ru0louQq4F/arcgis/rest/services/Cascadia_Bioregion_Boundary/FeatureServer/0/query"

# Set query parameters to get all features as GeoJSON
params = {
    "where": "1=1",          # get all features
    "outFields": "*",
    "f": "geojson",          # ask for GeoJSON format
    "outSR": "4326"          # WGS84
}

response = requests.get(url, params=params)
data = response.json()

# Save to file
with open("/home/mila/l/lastc/scratch/data/ERA5_DATA_TEST/cascadia_bioregion.geojson", "w") as f:
    import json
    json.dump(data, f)

print("Saved as cascadia_bioregion.geojson")
