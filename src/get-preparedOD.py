import os
import sys

import pandas as pd
import geopandas as gpd

from datetime import datetime

##########################################################################################
#GOAL: Prepare the OD matrices of each FUA for the distance computations

fua_code = sys.argv[1]
start = datetime.now()

##########################################################################################

def trim_centroids(od_matrix, buffered_boundary, bdry_as_gdf=True):
    
    if bdry_as_gdf:
        buffered_boundary = buffered_boundary.geometry[0]
    
    centroids_pt = gpd.points_from_xy(x= od_matrix.intptlon, y=od_matrix.intptlat, crs='EPSG:4326')
    rows_to_keep = centroids_pt.within(buffered_boundary)
    trimmed_od_matrix = od_matrix[rows_to_keep].reset_index(drop=True)

    return trimmed_od_matrix

##########################################################################################

print('FUA: ', fua_code)

fua_buffered_boundaries = gpd.read_file('../data/d03_intermediate/FUA-buffered-shapefile/FUA-buffered.shp').set_index('fuacode')

for fua_code, od_matrix in tqdm(od_matrix_dict.items(), total=len(od_matrix_dict)):
    buffered_boundary = fua_buffered_boundaries.loc[[fua_code]]
    trimmed_od = trim_centroids(od_matrix, buffered_boundary)
    trimmed_od.to_csv('/scratch/g.spessatoagostini/d02_processed-safegraph/trimmed-OD-per-FUA/' + fua_code+'_trimmed-ODmatrix.csv')

print('RUNTIME:', datetime.now()-start)