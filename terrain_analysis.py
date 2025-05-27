# terrain_analysis.py
# Import necessary libraries
import argparse
import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
import math
from scipy import spatial
from proximity import proximity 
from sklearn.ensemble import RandomForestClassifier

# End of import section

def convert_to_rasterio(raster_data, template_raster, output_file):
    """ 
    Converts a NumPy array back into a Rasterio dataset format using a template.
    
    Parameters 
    ----------
    raster_data : numpy.ndarray
        The NumPy array containing raster data to convert.
    template_raster : rasterio.io.DatasetReader
        A Rasterio dataset to provide spatial data.
    output_file : str
        Path to save the output raster file
        
    Returns
    -------
    None
    """
    
    # Copy metadata from template
    out_meta = template_raster.meta.copy()

    # Update metadata for number of bands and data type
    out_meta.update({
        "count": 1,
        "dtype": raster_data.dtype
    })

    # Write the new raster
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(raster_data, 1)


def extract_values_from_raster(raster, shape_object):
    """
    Extracts values from a raster at given point locations.

    Parameters
    ----------
    raster : rasterio.io.DatasetReader
        The raster that we are sampling from.
    shape_object : list
        A list of point objects with x and y coordinates.

    Returns
    -------
    list
        A list of raster values at the point locations.

    """
    
    values = []
    
    for point in shape_object:
        x = point.x
        y = point.y
        
        for val in raster.sample([(x,y)]):
            values.append(val[0])
   
    return values

def make_classifier(x, y, verbose=False):
    """
    Trains a machine learning classifier based on training data.

    Parameters
    ----------
    x : pandas.DataFrame or numpy.ndarray
        The input feature data for training.
    y : pandas.Series or numpy.ndarray
        The target labels.
    verbose : bool, optional
        If True, prints progress and performance details.

    Returns
    -------
    object
        A trained classifier object.
    """

    # Create the classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)

    # Fit the classifier to the training data
    classifier.fit(x, y)

    # Optionally print progress details
    if verbose:
        print("Classifier trained successfully.")

    return classifier

def make_prob_raster_data(topo, geo, lc, dist_fault_rasterised, slope, classifier):
    """
    Creates a probability raster predicting landslide hazard using input rasters and a classifier.

    Parameters
    ----------
    topo : rasterio.io.DatasetReader
        Topography raster dataset (used as spatial template).
    geo : rasterio.io.DatasetReader
        Geology raster dataset.
    lc : rasterio.io.DatasetReader
        Landcover raster dataset.
    dist_fault_rasterised : numpy.ndarray
        Rasterised faults (e.g., binary array: 1 where fault exists, else 0).
    slope : numpy.ndarray
        Array containing slope values.
    classifier : object
        Trained classifier used for prediction.

    Returns
    -------
    numpy.ndarray
        Array containing predicted landslide probabilities.
    """

    # Calculate distance raster to fault pixels (where value == 1)
    distance_to_fault = proximity(topo, dist_fault_rasterised, value=1)

    # Read the other rasters as numpy arrays
    topo_arr = topo.read(1)
    geo_arr = geo.read(1)
    lc_arr = lc.read(1)

    # Flatten all arrays for classifier input (stack features column-wise)
    # Make sure all arrays have the same shape!
    height, width = topo_arr.shape
    n_pixels = height * width

    # Flatten arrays
    topo_flat = topo_arr.flatten()
    geo_flat = geo_arr.flatten()
    lc_flat = lc_arr.flatten()
    dist_fault_flat = distance_to_fault.flatten()
    slope_flat = slope.flatten()

    # Stack features into 2D array (n_samples x n_features)
    X = np.vstack([topo_flat, geo_flat, lc_flat, dist_fault_flat, slope_flat]).T

    # Predict probabilities of landslide hazard
    # Assuming classifier.predict_proba returns probabilities with shape (n_samples, n_classes)
    # and that landslide class is at index 1
    probs = classifier.predict_proba(X)[:, 1]

    # Reshape probabilities back to raster shape
    prob_raster = probs.reshape((height, width))

    return prob_raster

def create_dataframe(topo, geo, lc, dist_fault, slope, shape, landslides):
    """
    Compiles input raster and vector data into a pandas DataFrame for model training.

    Parameters
    ----------
    topo : rasterio.io.DatasetReader
        Topography raster dataset.
    geo : rasterio.io.DatasetReader
        Geology raster dataset.
    lc : rasterio.io.DatasetReader
        Landcover raster dataset.
    dist_fault : numpy.ndarray
        Array containing distance-to-fault values.
    slope : numpy.ndarray
        Array containing slope values.
    shape : geopandas.GeoDataFrame
        GeoDataFrame containing point locations for sampling.
    landslides : geopandas.GeoDataFrame
        GeoDataFrame containing known landslide locations.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing input variables and corresponding landslide labels.
    """
    return


def main():
    """
    Command-line interface for landslide hazard modelling.

    Parses user-specified input arguments for raster and vector files,
    runs the landslide hazard model pipeline, and writes output raster file.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    parser = argparse.ArgumentParser(
                     prog="Landslide hazard using ML",
                     description="Calculate landslide hazards using simple ML",
                     epilog="Copyright 2024, Jon Hill"
                     )
    parser.add_argument('--topography',
                    required=True,
                    help="topographic raster file")
    parser.add_argument('--geology',
                    required=True,
                    help="geology raster file")
    parser.add_argument('--landcover',
                    required=True,
                    help="landcover raster file")
    parser.add_argument('--faults',
                    required=True,
                    help="fault location shapefile")
    parser.add_argument("landslides",
                    help="the landslide location shapefile")
    parser.add_argument("output",
                    help="the output raster file")
    parser.add_argument('-v', '--verbose',
                    action='store_true',
                    default=False,
                    help="Print progress")

    args = parser.parse_args()


if __name__ == '__main__':
    main()
