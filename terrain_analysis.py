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

    return

def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):

    return

def create_dataframe(topo, geo, lc, dist_fault, slope, shape, landslides):

    return


def main():


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
