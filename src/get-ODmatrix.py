import sys
import os
from multiprocessing import Pool, cpu_count

import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox

from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

##########################################################################################
#GOAL: Get the network distance corresponding to each commute logged by SafeGraph

script_args = sys.argv

print('ARGS:', script_args, '\n')

fua_code = sys.argv[1]
threshold = 2000
number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])

#Maybe we passed the additional argument to select lower or upper rows:
if len(script_args) > 2:
    split = int(sys.argv[2])
else:
    split = False

start = datetime.now()

##########################################################################################

#Defining the output file:
outdir = '/scratch/g.spessatoagostini/OD-per-FUA/'

if split:
    outpath = outdir + fua_code + '_ODmatrix_' + str(split) + '.csv'
else:
    outpath = outdir + fua_code + '_ODmatrix.csv'

##########################################################################################

#Defining input directories:

fua_buffered_shapefile_dir = '../data/d03_intermediate/FUA-buffered-shapefile/'
full_od_matrix_dir = '../data/d02_processed-safegraph/'
networks_dir = '/scratch/g.spessatoagostini/expanded_networks/'

##########################################################################################

#Functions adapted from OSMnx to compute distance:

def _single_shortest_path_distance(G, orig, dest, weight):
    try:
        return nx.shortest_path_length(G, orig, dest, weight=weight) #change function here from G. Boeing's OSMNx repo
    except nx.exception.NetworkXNoPath:
        return None
    
def shortest_path_distance(G, orig, dest, weight="length", cpus=1):
    if not (hasattr(orig, "__iter__") or hasattr(dest, "__iter__")):
        # if neither orig nor dest is iterable, just return the shortest path
        return _single_shortest_path_distance(G, orig, dest, weight)

    elif hasattr(orig, "__iter__") and hasattr(dest, "__iter__"):
        # if both orig and dest are iterables ensure they have same lengths
        if len(orig) != len(dest):  # pragma: no cover
            raise ValueError("orig and dest must contain same number of elements")

        if cpus is None:
            cpus = cpu_count()
        cpus = min(cpus, cpu_count())

        if cpus == 1:
            # if single-threading, calculate each shortest path one at a time
            paths = [_single_shortest_path_distance(G, o, d, weight) for o, d in zip(orig, dest)]
        else:
            # if multi-threading, calculate shortest paths in parallel
            args = ((G, o, d, weight) for o, d in zip(orig, dest))
            pool = Pool(cpus)
            sma = pool.starmap_async(_single_shortest_path_distance, args)
            paths = sma.get()
            pool.close()
            pool.join()
        return paths

    else:
        # if only one of orig or dest is iterable and the other is not
        raise ValueError("orig and dest must either both be iterable or neither must be iterable")

##########################################################################################

def get_boundary(fua_code):
    return gpd.read_file(fua_buffered_shapefile_dir + 'FUA-buffered.shp').set_index('fuacode').loc[[fua_code]]

def get_fua_ODmatrix(fua_code, split_in_half=split):
    full_od_matrix = pd.read_csv(full_od_matrix_dir + 'weeks_od_us_fua.csv')
    fua_raw_od_matrix = full_od_matrix[full_od_matrix.fuacode==fua_code].reset_index(drop=True)
    fua_raw_od_matrix['fuacode'] = fua_code
    
    if split_in_half == 1:
        return fua_raw_od_matrix[:len(fua_raw_od_matrix)//2].reset_index(drop=True)
    elif split_in_half == 2:
        return fua_raw_od_matrix[len(fua_raw_od_matrix)//2:].reset_index(drop=True)
    else:
        return fua_raw_od_matrix

def load_graphs(fua_code, proj_crs='EPSG:5070'):
    walk_graph = ox.project_graph(ox.load_graphml(networks_dir + 'walk/'+fua_code+'.graphml'), to_crs=proj_crs)
    drive_graph = ox.project_graph(ox.load_graphml(networks_dir + 'drive/'+fua_code+'.graphml'), to_crs=proj_crs)
    
    return walk_graph, drive_graph

def trim_centroids(od_matrix, buffered_boundary, bdry_as_gdf=True):
    
    if bdry_as_gdf:
        buffered_boundary = buffered_boundary.geometry[0]
    
    centroids_pt = gpd.points_from_xy(x= od_matrix.intptlon, y=od_matrix.intptlat, crs='EPSG:4326')
    rows_to_keep = centroids_pt.within(buffered_boundary)
    trimmed_od_matrix = od_matrix[rows_to_keep].reset_index(drop=True)
    
    print('.   total of', len(trimmed_od_matrix), ' rows')

    return trimmed_od_matrix

def add_od_geometries(od_matrix, proj_crs='EPSG:5070'):
    centroids_pt = gpd.points_from_xy(x=od_matrix.intptlon, y=od_matrix.intptlat, crs='EPSG:4326').to_crs(proj_crs)
    od_matrix['origin_x'], od_matrix['origin_y'] = centroids_pt.x, centroids_pt.y

    places_pt = gpd.points_from_xy(x= od_matrix.longitude, y=od_matrix.latitude, crs='EPSG:4326').to_crs(proj_crs)
    od_matrix['dest_x'], od_matrix['dest_y'] = places_pt.x, places_pt.y
    
    return od_matrix

def add_preferred_mode(od_matrix, max_walk_dist=2000, proj_crs='EPSG:5070'):
    
    places_pt = gpd.points_from_xy(x= od_matrix.longitude, y=od_matrix.latitude, crs='EPSG:4326').to_crs(proj_crs)
    centroids_pt = gpd.points_from_xy(x=od_matrix.intptlon, y=od_matrix.intptlat, crs='EPSG:4326').to_crs(proj_crs)
    
    od_matrix['mode'] = places_pt.distance(centroids_pt) <= max_walk_dist
    od_matrix['mode'] = od_matrix['mode'].map({True: 'walk', False:'drive'})
    
    return od_matrix
    
def add_distances(od_matrix, walk_graph, drive_graph, cpus=1):
    
    #Split the dataframe in two according to commute mode:
    od_matrix_dict = {mode: df for mode, df in od_matrix.groupby('mode')}
    G = {'drive': drive_graph, 'walk': walk_graph}
    
    #For each of the commute modes, do the distance computation:
    full_dfs = []
    for mode, df in od_matrix_dict.items():
        df['origin_node'], df['origin_node_dist'] = ox.nearest_nodes(G[mode], df['origin_x'], df['origin_y'], return_dist=True)
        df['destination_node'], df['destination_node_dist'] = ox.nearest_nodes(G[mode], df['dest_x'], df['dest_y'], return_dist=True)
        df['distance'] = shortest_path_distance(G[mode],
                                                df['origin_node'].values, df['destination_node'].values,
                                                cpus)
        full_dfs.append(df)    
    
    #Merge the two dataframes:
    od_matrix_naivedistance = pd.concat(full_dfs, ignore_index=True)
    
    return od_matrix_naivedistance

def refine_distances(od_matrix_with_distances, drive_graph, max_walk_dist=2000, cpus=1):
    
    rows_to_repeat = (od_matrix_with_distances['mode']=='walk') & (od_matrix_with_distances['distance'] > max_walk_dist)
    df_to_repeat = od_matrix_with_distances[rows_to_repeat]
    
    print('.   repeat for', len(df_to_repeat), ' rows')
    
    if len(df_to_repeat) > 0:
        
        od_matrix_with_distances.loc[rows_to_repeat, 'mode'] = 'drive'
        
        df_to_repeat['origin_node'], df_to_repeat['origin_node_dist'] = ox.nearest_nodes(drive_graph,
                                                                                         df_to_repeat['origin_x'], df_to_repeat['origin_y'],
                                                                                         return_dist=True)
        od_matrix_with_distance.loc[rows_to_repeat, 'origin_node_dist'] = df_to_repeat['origin_node_dist'].values
        od_matrix_with_distance.loc[rows_to_repeat, 'origin_node'] = df_to_repeat['origin_node'].values
        
        df_to_repeat['destination_node'], df_to_repeat['destination_node_dist'] = ox.nearest_nodes(drive_graph,
                                                                                                   df_to_repeat['dest_x'], df_to_repeat['dest_y'],
                                                                                                   return_dist=True)
        od_matrix_with_distance.loc[rows_to_repeat, 'destination_node_dist'] = df_to_repeat['destination_node_dist'].values
        od_matrix_with_distance.loc[rows_to_repeat, 'destination_node'] = df_to_repeat['destination_node'].values

        od_matrix_with_distances.loc[rows_to_repeat, 'distance'] = shortest_path_distance(drive_graph,
                                                                                          df_to_repeat['origin_node'].values, df_to_repeat['destination_node'].values,
                                                                                          cpus)
    
    return od_matrix_with_distances

def drop_cols(od_matrix, cols_to_drop=['origin_x', 'origin_y', 'origin_node', 'destination_node', 'dest_x', 'dest_y']):
    
    for col in od_matrix_naivedistance.columns:
        if 'Unnamed' in col:
            cols_to_drop.append(col)
            
    final_matrix = od_matrix.drop(cols_to_drop, axis=1).reset_index(drop=True)
    
    return final_matrix

##########################################################################################

print('FUA: ', fua_code, '\n')

try:

    start=datetime.now()

    #1. LOAD ALL THE FILES:
    fua_buffered_boundary = get_boundary(fua_code) #get the FUA boundary
    print('   bdry loaded')
    fua_raw_od_matrix = get_fua_ODmatrix(fua_code) #get the commutes within that FUA
    print('   od matrix loaded')
    walk_graph, drive_graph = load_graphs(fua_code) #get the graphs
    loading_complete=datetime.now()
    print(' Loaded all files in:', loading_complete-start)

    #2. PREPROCESS THE MATRIX:
    fua_od_matrix = trim_centroids(fua_raw_od_matrix, fua_buffered_boundary)
    georeferenced_fua_od_matrix = add_od_geometries(fua_od_matrix)
    georeferenced_fua_od_matrix_with_mode = add_preferred_mode(georeferenced_fua_od_matrix, threshold)

    processing_complete=datetime.now()
    print(' Prepared matrix in:', processing_complete-loading_complete)

    #3. OBTAIN THE DISTANCES:
    matrix_naive_distances = add_distances(georeferenced_fua_od_matrix_with_mode, walk_graph, drive_graph, cpus=number_of_cores)

    distances_complete=datetime.now()
    print(' Obtained distances in:', distances_complete-processing_complete)

    #4. REPEAT FOR EDGE CASES:
    matrix_final_distances = refine_distances(matrix_naive_distances, drive_graph, threshold, cpus=number_of_cores)

    all_complete=datetime.now()
    print(' Refined distances in:', all_complete-distances_complete)

    #5. WRAP-UP MATRIX AND SAVE IT:
    final_matrix = drop_cols(matrix_final_distances)
    final_matrix.to_csv(outpath)

except:
    print('\nERROR')
