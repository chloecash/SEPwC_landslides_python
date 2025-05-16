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

# End of import section

def convert_to_rasterio(raster_data, template_raster):
    """ 
    Converts a NumPy array back into a Rasterio dataset format using a template.
    
    Parameters 
    ----------
    raster_data : numpy.ndarray
        The NumPy array containing raster data to convert.
    template_raster : rasterio.io.DatasetReader
        A Rasterio dataset to provide spatial data.
        
    Returns
    -------
    rasterio.io.DatasetWriter
        A Rasterio raster dataset containing the converted data.
    """
    return


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
    return

def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):
    """
    Creates a probability raster predicting landslide hazard using input rasters and a classifier.

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
    classifier : object
        Trained classifier used for prediction.

    Returns
    -------
    numpy.ndarray
        Array containing predicted landslide probabilities.
    """
    return

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
