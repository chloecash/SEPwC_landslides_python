"""
Landslide hazard prediction using machine learning.
Processes raster and shapefile data to train a Random Forest classifier
and generate a landslide probability map.
"""

import argparse
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Point
from proximity import proximity
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from rasterio.io import MemoryFile

def convert_to_rasterio(raster_data, template_raster):
    """Convert a NumPy array to a Rasterio in-memory raster using the profile of a template raster."""
    from rasterio.io import MemoryFile
    profile = template_raster.profile.copy()
    profile.update(dtype=rasterio.float32, count=1)
    memfile = MemoryFile()
    with memfile.open(**profile) as dataset:
        dataset.write(raster_data.astype(rasterio.float32), 1)
    return memfile.open()

def extract_values_from_raster(raster, shape_object):
    """Extract raster values at the centroids of geometries from a GeoSeries."""
    values = []
    for geom in shape_object:
        centroid = geom.centroid
        try:
            row, col = raster.index(centroid.x, centroid.y)
            if 0 <= row < raster.height and 0 <= col < raster.width:
                val = raster.read(1)[row, col]
                values.append(val)
            else:
                values.append(np.nan)
        except:
            values.append(np.nan)
    return values

def make_classifier(x, y, verbose=False):
    """Train and return a RandomForestClassifier on the input data."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(x, y)
    if verbose:
        print("Training accuracy:", accuracy_score(y, clf.predict(x)))
    return clf

def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):
    """Generate a raster of predicted landslide probabilities using a trained classifier."""
    rows, cols = topo.read(1).shape
    prob_raster = np.zeros((rows, cols), dtype=np.float32)

    elev_data = topo.read(1)
    geol_data = geo.read(1)
    lc_data = lc.read(1)
    fault_data = dist_fault.read(1)
    slope_data = slope.read(1)

    flat_data = np.column_stack([
        elev_data.ravel(),
        fault_data.ravel(),
        slope_data.ravel(),
        lc_data.ravel(),
        geol_data.ravel()
    ])

    valid = ~np.any(np.isnan(flat_data), axis=1)
    probs = np.zeros(flat_data.shape[0])
    if valid.any():
        probs[valid] = classifier.predict_proba(flat_data[valid])[:, 1]

    prob_raster = probs.reshape((rows, cols))
    return prob_raster

def create_dataframe(topo, geo, lc, dist_fault, slope, shape, landslides):
    """Create a GeoDataFrame with raster values at point geometries and landslide labels."""
    if not isinstance(shape, gpd.GeoSeries):
        shape = gpd.GeoSeries(shape)

    elev_values = extract_values_from_raster(topo, shape)
    geo_values = extract_values_from_raster(geo, shape)
    lc_values = extract_values_from_raster(lc, shape)
    dist_values = extract_values_from_raster(dist_fault, shape)
    slope_values = extract_values_from_raster(slope, shape)

    if isinstance(landslides, int):
        labels = [landslides for _ in shape]
    else:
        buffer_dist = 80
        landslide_buffers = landslides.geometry.buffer(buffer_dist)
        labels = [1 if landslide_buffers.contains(geom).any() else 0 for geom in shape]

    data = {
        'elev': elev_values,
        'fault': dist_values,
        'slope': slope_values,
        'LC': lc_values,
        'Geol': geo_values,
        'ls': labels
    }

    crs = topo.crs if hasattr(topo, 'crs') and topo.crs else "EPSG:32632"
    df = gpd.GeoDataFrame(data, geometry=shape, crs=crs)
    df.dropna(inplace=True)
    return df[['elev', 'fault', 'slope', 'LC', 'Geol', 'ls']]

def main():
    """Main function to parse arguments, process data, train model, and save output raster."""
    parser = argparse.ArgumentParser(
        prog="Landslide hazard using ML",
        description="Calculate landslide hazards using simple ML",
        epilog="Copyright 2024, Jon Hill"
    )
    parser.add_argument('--topography', required=True, help="topographic raster file")
    parser.add_argument('--geology', required=True, help="geology raster file")
    parser.add_argument('--landcover', required=True, help="landcover raster file")
    parser.add_argument('--faults', required=True, help="fault location shapefile")
    parser.add_argument("landslides", help="the landslide location shapefile")
    parser.add_argument("output", help="the output raster file")
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help="Print progress")

    args = parser.parse_args()

    topo = rasterio.open(args.topography)
    geo = rasterio.open(args.geology)
    lc = rasterio.open(args.landcover)
    faults = gpd.read_file(args.faults)
    landslides = gpd.read_file(args.landslides)

    fault_rasterized = rasterize(
        [(geom, 1) for geom in faults.geometry],
        out_shape=topo.read(1).shape,
        transform=topo.transform,
        fill=0,
        dtype='uint8'
    )
    dist_fault_array = proximity(topo, fault_rasterized, 1)
    dist_fault = convert_to_rasterio(dist_fault_array.astype('float32'), topo)

    elev_data = topo.read(1).astype("float32")
    grad_y, grad_x = np.gradient(elev_data)
    slope_array = np.sqrt(grad_x**2 + grad_y**2)
    slope = convert_to_rasterio(slope_array, topo)

    landslide_points = landslides.geometry
    df_landslides = create_dataframe(topo, geo, lc, dist_fault, slope, landslide_points, landslides)

    bounds = topo.bounds
    num_samples = len(df_landslides)
    xs = np.random.uniform(bounds.left, bounds.right, num_samples)
    ys = np.random.uniform(bounds.bottom, bounds.top, num_samples)
    random_points = gpd.GeoSeries([Point(x, y) for x, y in zip(xs, ys)], crs=topo.crs)
    df_random = create_dataframe(topo, geo, lc, dist_fault, slope, random_points, 0)

    df = pd.concat([df_landslides, df_random], ignore_index=True)
    df.dropna(inplace=True)

    x = df[['elev', 'fault', 'slope', 'LC', 'Geol']]
    y = df['ls']

    classifier = make_classifier(x, y, args.verbose)
    prob_raster = make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier)

    profile = topo.profile
    profile.update(dtype='float32', count=1)
    with rasterio.open(args.output, 'w', **profile) as dst:
        dst.write(prob_raster, 1)

    if args.verbose:
        print(f"Probability raster written to {args.output}")

if __name__ == '__main__':
    main()
