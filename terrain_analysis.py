"""Module for terrain analysis including data processing and modeling."""

# Standard library imports
import argparse

# Third-party imports
import rasterio
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from rasterio.features import rasterize

# Unused imports removed: math, os, pandas as pd

def convert_to_rasterio(raster_data, template_raster, output_file="temp_output.tif"):
    """Convert a numpy array to a rasterio dataset using metadata from a template raster."""
    out_meta = template_raster.meta.copy()
    out_meta.update({
        "count": 1,
        "dtype": raster_data.dtype,
        "height": raster_data.shape[0],
        "width": raster_data.shape[1],
        "transform": template_raster.transform
    })
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(raster_data, 1)
    return rasterio.open(output_file)

def extract_values_from_raster(raster, shape_object):
    """Extract values from a raster at given point locations."""
    values = []
    for point in shape_object:
        x_coord = point.x
        y_coord = point.y
        for val in raster.sample([(x_coord, y_coord)]):
            values.append(val[0])
    return values

def make_classifier(x_data, y_labels, verbose=False):
    """Train a random forest classifier."""
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    classifier.fit(x_data, y_labels)
    if verbose:
        print("Classifier trained successfully.")
    return classifier

def make_prob_raster_data(topo, geo, lc, dist_fault_rasterised, slope, classifier, proximity_func):
    """Create a probability raster predicting landslide hazard."""
    distance_to_fault = proximity_func(topo, dist_fault_rasterised, value=1)
    topo_arr = topo.read(1)
    geo_arr = geo.read(1)
    lc_arr = lc.read(1)

    topo_flat = topo_arr.flatten()
    geo_flat = geo_arr.flatten()
    lc_flat = lc_arr.flatten()
    dist_fault_flat = distance_to_fault.flatten()
    slope_flat = slope.flatten()

    features = np.vstack([topo_flat, geo_flat, lc_flat, dist_fault_flat, slope_flat]).T
    probs = classifier.predict_proba(features)[:, 1]
    prob_raster = probs.reshape(topo_arr.shape)
    return prob_raster

def create_dataframe(topo, geo, lc, dist_fault, slope, shape, landslides_gdf):
    """Compile raster and vector inputs into a GeoDataFrame with features and landslide labels."""
    if not isinstance(shape, gpd.GeoDataFrame):
        shape = gpd.GeoDataFrame(geometry=shape)

    if hasattr(dist_fault, 'read'):
        dist_fault = dist_fault.read(1)
    if hasattr(slope, 'read'):
        slope = slope.read(1)

    def extract_values(raster, points):
        return [val[0] for val in raster.sample([(pt.x, pt.y) for pt in points.geometry])]

    elev_vals = extract_values(topo, shape)
    geo_vals = extract_values(geo, shape)
    lc_vals = extract_values(lc, shape)
    fault_vals = extract_values(dist_fault if hasattr(dist_fault, 'sample') else topo, shape)
    slope_vals = extract_values(slope if hasattr(slope, 'sample') else topo, shape)

    labels = [0] * len(shape) if landslides_gdf == 0 else [
        1 if landslides_gdf.intersects(pt).any() else 0 for pt in shape.geometry]

    df = gpd.GeoDataFrame({
       'elev': elev_vals,
       'fault': fault_vals,
       'slope': slope_vals,
       'LC': lc_vals,
       'Geol': geo_vals,
       'ls': labels
    }, geometry=shape.geometry, crs=shape.crs)

    return df

def main():
    """Command-line interface for landslide hazard modelling."""
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
    parser.add_argument('-v', '--verbose', action='store_true', default=False,\
    help="Print progress")

    args = parser.parse_args()

    if args.verbose:
        print("Opening input datasets...")

    topo = rasterio.open(args.topography)
    geo = rasterio.open(args.geology)
    lc = rasterio.open(args.landcover)

    if args.verbose:
        print("Reading faults shapefile...")
    faults = gpd.read_file(args.faults)

    if args.verbose:
        print("Reading landslides shapefile...")
    landslides = gpd.read_file(args.landslides)

    if args.verbose:
        print("Rasterizing faults shapefile...")
    fault_shapes = [(geom, 1) for geom in faults.geometry]

    fault_raster = rasterize(
        fault_shapes,
        out_shape=topo.shape,
        transform=topo.transform,
        fill=0,
        dtype='uint8'
    )

    if args.verbose:
        print("Calculating slope raster...")
    from slope_module import calculate_slope  # assuming you move slope calc here
    slope = calculate_slope(topo.read(1), topo.transform)

    if args.verbose:
        print("Creating training dataframe...")
    training_df = create_dataframe(topo, geo, lc, fault_raster, slope, list\
    (landslides.geometry), landslides)

    if args.verbose:
        print("Training classifier...")
    classifier = make_classifier(training_df.drop('ls', axis=1), training_df\
    ['ls'], verbose=args.verbose)

    if args.verbose:
        print("Predicting landslide probabilities...")
    from proximity_module import proximity  # assuming proximity func here
    prob_raster = make_prob_raster_data(topo, geo, lc, fault_raster, slope, classifier, proximity)

    if args.verbose:
        print(f"Saving output raster to {args.output} ...")
    convert_to_rasterio(prob_raster.astype(np.float32), topo, output_file=args.output)

    if args.verbose:
        print("Done!")

if __name__ == '__main__':
    main()