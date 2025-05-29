"""Module for terrain analysis including data processing and modeling."""
import argparse
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import geometry_mask
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from rasterio.transform import from_origin
from rasterio.io import MemoryFile

def convert_to_rasterio(raster_data, template_raster):
    """Convert a numpy array into an in-memory rasterio dataset using a template for metadata."""
    profile = template_raster.profile.copy()
    profile.update(dtype=raster_data.dtype, count=1)

    memfile = MemoryFile()
    with memfile.open(**profile) as dataset:
        dataset.write(raster_data, 1)
    return memfile.open()


def extract_values_from_raster(raster, shape_object):
    """Extract raster values at point locations."""
    values = []
    for geom in shape_object:
        coords = [(geom.x, geom.y)]
        for val in raster.sample(coords):
            values.append(val[0])
    return values


def make_classifier(x, y, verbose=False):
    """Create and train a Random Forest classifier."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(x, y)
    if verbose:
        print("Classifier trained with feature importances:")
        for feature, importance in zip(x.columns, clf.feature_importances_):
            print(f"{feature}: {importance:.4f}")
    return clf


def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):
    """Apply classifier to raster stack and produce probability raster data."""
    profile = topo.profile
    height, width = topo.read(1).shape

    # Read all rasters into arrays
    topo_data = topo.read(1)
    geo_data = geo.read(1)
    lc_data = lc.read(1)
    dist_fault_data = dist_fault.read(1)
    slope_data = slope.read(1)

    # Flatten and stack
    X = np.column_stack((
        topo_data.flatten(),
        geo_data.flatten(),
        lc_data.flatten(),
        dist_fault_data.flatten(),
        slope_data.flatten()
    ))

    # Predict probabilities
    prob = classifier.predict_proba(X)[:, 1]  # probability of class '1' (landslide)

    # Reshape back to raster shape
    prob_raster = prob.reshape(height, width).astype('float32')
    return prob_raster


def create_dataframe(elev, fault, slope, LC, Geol, shape,landslides):
    """Create a GeoDataFrame with raster values at landslide and non-landslide locations."""
    
    # Extract raster values at shape points
    elev_values = extract_values_from_raster(elev, shape)
    geo_values = extract_values_from_raster(Geol, shape)
    lc_values = extract_values_from_raster(LC, shape)
    dist_values = extract_values_from_raster(fault, shape)
    slope_values = extract_values_from_raster(slope, shape)

    labels = []
    # Label landslides (1) or not (0)
    if isinstance(landslides, int):
        # Assign all labels to landslides (0 or 1)
        labels = [landslides for _ in shape]
    else:
        # Buffer landslide points by 80 meters only if landslides is GeoDataFrame
        buffer_dist = 80  # meters
        landslide_buffers = landslides.geometry.buffer(buffer_dist)
        for geom in shape:
            labels.append(1 if landslide_buffers.contains(geom).any() else 0)

    data = {
       'elev': elev_values,
       'Geol': geo_values,
       'LC': lc_values,
       'fault': dist_values,
       'slope': slope_values,
       'ls': labels
    }

    gdf = gpd.GeoDataFrame(data, geometry=shape)
    return gdf

def main():
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

    if args.verbose:
        print("Opening raster files...")

    topo = rasterio.open(args.topography)
    geo = rasterio.open(args.geology)
    lc = rasterio.open(args.landcover)

    faults = gpd.read_file(args.faults)
    landslides = gpd.read_file(args.landslides)

    if args.verbose:
        print("Creating distance raster to faults...")

    fault_mask = geometry_mask(faults.geometry, transform=topo.transform,
                               invert=True, out_shape=topo.shape)
    dist_fault_array = np.where(fault_mask, 0, 1).astype('uint8')

    dist_fault_raster = convert_to_rasterio(dist_fault_array, topo)

    slope_array = np.gradient(topo.read(1))[0]  # Simple slope estimate using gradient
    slope_raster = convert_to_rasterio(slope_array, topo)

    if args.verbose:
        print("Extracting raster values at landslide and non-landslide locations...")

    points = landslides.geometry
    df = create_dataframe(topo, geo, lc, dist_fault_raster, slope_raster, points, landslides)

    x = df[['elev', 'Geol', 'LC', 'dist_fault', 'slope']]
    y = df['ls']

    classifier = make_classifier(x, y, args.verbose)

    if args.verbose:
        print("Creating probability raster...")

    prob_raster = make_prob_raster_data(topo, geo, lc, dist_fault_raster, slope_raster, classifier)

    profile = topo.profile
    profile.update(dtype='float32', count=1)

    with rasterio.open(args.output, 'w', **profile) as dst:
        dst.write(prob_raster, 1)

    if args.verbose:
        print(f"Probability raster written to {args.output}")


if __name__ == '__main__':
    main()