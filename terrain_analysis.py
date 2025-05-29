'''
Landslide hazard prediction using machine learning.
Processes raster and shapefile data to train a Random Forest classifier
and generate a landslide probability map.
'''

import argparse
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from proximity import proximity


def convert_to_rasterio(raster_data, template_raster):
    """Convert a NumPy array to a Rasterio in-memory raster using a template's profile."""
    profile = template_raster.profile.copy()
    profile.update(dtype=rasterio.float32, count=1)
    memfile = MemoryFile()
    with memfile.open(**profile) as dataset:
        dataset.write(raster_data.astype(rasterio.float32), 1)
    return memfile.open()


def extract_values_from_raster(raster, shape_series):
    """Extract raster values at the centroids of geometries in a GeoSeries."""
    values = []
    for geom in shape_series:
        centroid = geom.centroid
        try:
            row, col = raster.index(centroid.x, centroid.y)
            if 0 <= row < raster.height and 0 <= col < raster.width:
                values.append(raster.read(1)[row, col])
            else:
                values.append(np.nan)
        except (ValueError, IndexError):
            values.append(np.nan)
    return values


def make_classifier(x, y, verbose=False):
    """Train and return a RandomForestClassifier on the input data."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(x, y)
    if verbose:
        print("Training accuracy:", accuracy_score(y, clf.predict(x)))
    return clf


def make_prob_raster_data(config):
    """Generates a probability raster from input layers using a trained classifier."""
    topo = config['topo']
    geo = config['geo']
    lc = config['lc']
    dist = config['dist_fault']
    slope = config['slope']
    clf = config['classifier']

    rows, cols = topo.read(1).shape
    flat_data, valid = build_feature_matrix(topo, geo, lc, dist, slope)
    probs = np.zeros(flat_data.shape[0])
    if valid.any():
        probs[valid] = clf.predict_proba(flat_data[valid])[:, 1]
    return probs.reshape((rows, cols))


def build_feature_matrix(topo, geo, lc, dist, slope):
    """Builds a flat feature matrix and a validity mask from raster inputs."""
    elev = topo.read(1)
    geol = geo.read(1)
    lc_data = lc.read(1)
    dist_data = dist.read(1)
    slope_data = slope.read(1)

    flat_data = np.column_stack([
        elev.ravel(), dist_data.ravel(), slope_data.ravel(),
        lc_data.ravel(), geol.ravel()
    ])
    valid = ~np.any(np.isnan(flat_data), axis=1)
    return flat_data, valid


def extract_all_raster_values(rasters, shape_series):
    """Extract values from multiple raster layers given in a dict."""
    return {
        'elev': extract_values_from_raster(rasters['topo'], shape_series),
        'fault': extract_values_from_raster(rasters['dist_fault'], shape_series),
        'slope': extract_values_from_raster(rasters['slope'], shape_series),
        'LC': extract_values_from_raster(rasters['lc'], shape_series),
        'Geol': extract_values_from_raster(rasters['geo'], shape_series)
    }


def assign_landslide_labels(points, landslides):
    """Assign binary labels to points based on proximity to landslides."""
    buffers = landslides.geometry.buffer(80)
    return [1 if buffers.contains(pt).any() else 0 for pt in points]


def _create_dataframe(rasters, points, landslides):
    """Internal: Create a GeoDataFrame with raster values and landslide labels."""
    pts_series = gpd.GeoSeries(points) if not isinstance(points, gpd.GeoSeries) else points
    values = extract_all_raster_values(rasters, pts_series)

    if isinstance(landslides, int):
        labels = [landslides] * len(pts_series)
    else:
        labels = assign_landslide_labels(pts_series, landslides)

    df = gpd.GeoDataFrame(
        values,
        geometry=pts_series,
        crs=rasters['topo'].crs or 'EPSG:32632'
    )
    df['ls'] = labels
    df.dropna(inplace=True)
    return df[['elev', 'fault', 'slope', 'LC', 'Geol', 'ls', 'geometry']]


def create_dataframe(topo, geo, lc, dist_fault, slope, points, landslides):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Create a GeoDataFrame (positional signature for backward compatibility)."""
    rasters = {'topo': topo, 'geo': geo, 'lc': lc, 'dist_fault': dist_fault, 'slope': slope}
    return _create_dataframe(rasters, points, landslides)


def generate_random_points(bounds, num_points, crs):
    """Generate random points within bounds."""
    xs = np.random.uniform(bounds.left, bounds.right, num_points)
    ys = np.random.uniform(bounds.bottom, bounds.top, num_points)
    return gpd.GeoSeries([Point(x, y) for x, y in zip(xs, ys)], crs=crs)


def prepare_raster_inputs(topo):
    """Generate slope raster from topographic data."""
    elev_data = topo.read(1).astype('float32')
    grad_y, grad_x = np.gradient(elev_data)
    slope_arr = np.sqrt(grad_x**2 + grad_y**2)
    return convert_to_rasterio(slope_arr, topo)


def prepare_fault_distance_raster(topo, faults):
    """Generate distance-to-fault raster."""
    fault_rast = rasterize(
        [(geom, 1) for geom in faults.geometry],
        out_shape=topo.read(1).shape,
        transform=topo.transform,
        fill=0,
        dtype='uint8'
    )
    dist_arr = proximity(topo, fault_rast, 1)
    return convert_to_rasterio(dist_arr.astype('float32'), topo)


def main():
    """Main entrypoint to parse args, train model, and write probability map."""
    parser = argparse.ArgumentParser(
        prog='Landslide hazard using ML',
        description='Calculate landslide hazards using simple ML'
    )
    parser.add_argument('--topography', required=True)
    parser.add_argument('--geology', required=True)
    parser.add_argument('--landcover', required=True)
    parser.add_argument('--faults', required=True)
    parser.add_argument('landslides')
    parser.add_argument('output')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    topo = rasterio.open(args.topography)
    geo = rasterio.open(args.geology)
    lc = rasterio.open(args.landcover)
    faults = gpd.read_file(args.faults)
    landslides = gpd.read_file(args.landslides)

    dist_fault = prepare_fault_distance_raster(topo, faults)
    slope = prepare_raster_inputs(topo)

    df_ls = create_dataframe(topo, geo, lc, dist_fault, slope, landslides.geometry, landslides)
    df = pd.concat([
        df_ls,
        create_dataframe(
            topo, geo, lc, dist_fault, slope,
            generate_random_points(topo.bounds, len(df_ls), topo.crs),
            0
        )
    ], ignore_index=True)

    clf = make_classifier(df[['elev', 'fault', 'slope', 'LC', 'Geol']], df['ls'], args.verbose)
    prob = make_prob_raster_data({
        'topo': topo,
        'geo': geo,
        'lc': lc,
        'dist_fault': dist_fault,
        'slope': slope,
        'classifier': clf
    })

    profile = topo.profile.copy()
    profile.update(dtype='float32', count=1)
    with rasterio.open(args.output, 'w', **profile) as dst:
        dst.write(prob, 1)
    if args.verbose:
        print(f"Probability raster written to {args.output}")


if __name__ == '__main__':
    main()
