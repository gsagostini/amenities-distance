import os
import sys
from multiprocessing import Pool

import pandas as pd
import geopandas as gpd
import osmnx as ox

import warnings
warnings.filterwarnings("ignore")

##########################################################################################
#GOAL: Prepare the OD matrices of each FUA for the distance computations

fua_codes = list(set(pd.read_csv('../data/d02_processed-safegraph/weeks_od_us_fua.csv',
                                 usecols=['fuacode']).fuacode.dropna().values))
def num(code):
    return int(code[3:])
fua_codes.sort(key=num)

def get_expanded_OD(fua_code, threshold=2000):
    print('FUA: ', fua_code)
    if fua_code not in ['USA02', 'USA56']:
        #Get the files:
        od_matrix = pd.read_csv('../data/d02_processed-safegraph/OD-per-FUA/'+fua_code+'_ODmatrix.csv').drop(['Unnamed: 0'], axis=1)
        walk_graph = ox.project_graph(ox.load_graphml('../data/d03_intermediate/FUA-networks/walk/'+fua_code+'.graphml'), to_crs='EPSG:5070')
        drive_graph = ox.project_graph(ox.load_graphml('../data/d03_intermediate/FUA-networks/drive/'+fua_code+'.graphml'), to_crs='EPSG:5070')
        print('  got all files')
        
        #Get the geometries of origin and destinations:
        places_pt = gpd.points_from_xy(x= od_matrix.longitude, y=od_matrix.latitude, crs='EPSG:4326').to_crs('EPSG:5070')
        centroids_pt = gpd.points_from_xy(x= od_matrix.intptlon, y=od_matrix.intptlat, crs='EPSG:4326').to_crs('EPSG:5070')

        od_matrix['origin_x'], od_matrix['origin_y'] = centroids_pt.x, centroids_pt.y
        od_matrix['dest_x'], od_matrix['dest_y'] = places_pt.x, places_pt.y

        #Get the Boolean value of whether we walk or drive:
        od_matrix['walk'] = places_pt.distance(centroids_pt) <= threshold
        print('  got preferred mode')

        #Now we split the dataframe into two (one for walking and one for driving):
        od_matrix_dict = {walk: df for walk, df in od_matrix.groupby('walk')}
        G = {False: drive_graph, True: walk_graph}

        #For each of those dataframes, we do nearest nodes from OSMnx on the appropriate graph:
        full_dfs = []
        for walk, df in od_matrix_dict.items():
            df['origin_node'] = ox.nearest_nodes(G[walk], df['origin_x'], df['origin_y'])
            df['destination_node'] = ox.nearest_nodes(G[walk], df['dest_x'], df['dest_y'])
            full_dfs.append(df)

        merged_df = pd.concat(full_dfs, ignore_index=True)
        expanded_OD = merged_df.drop(['origin_x', 'origin_y', 'dest_x', 'dest_y'], axis=1)
        print('  got expanded matrix')
        
        expanded_OD.to_csv('../data/d04_final-OD-matrices/OD-per-FUA_nodistance/' + fua_code+'_expanded_nodistance-ODmatrix.csv')
        print('  saved')
    
    else:
        expanded_OD = None
        
    return expanded_OD


# number of cores you have allocated for your slurm task:
number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])

# multiprocssing pool to distribute tasks to:
with Pool(number_of_cores) as pool:
    # distribute computations and collect results:
    results = pool.starmap(get_expanded_OD, fua_codes)