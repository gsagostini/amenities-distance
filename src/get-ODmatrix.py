import os
import sys
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

fua_code = sys.argv[1]
threshold = 2000
number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
outpath = '/scratch/g.spessatoagostini/OD-per-FUA/' + fua_code + '_ODmatrix.csv'

start = datetime.now()

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

#Function to prepare the matrix by clipping points outside the FUA region:

def trim_centroids(od_matrix, buffered_boundary, bdry_as_gdf=True):
    
    if bdry_as_gdf:
        buffered_boundary = buffered_boundary.geometry[0]
    
    centroids_pt = gpd.points_from_xy(x= od_matrix.intptlon, y=od_matrix.intptlat, crs='EPSG:4326')
    rows_to_keep = centroids_pt.within(buffered_boundary)
    trimmed_od_matrix = od_matrix[rows_to_keep].reset_index(drop=True)

    return trimmed_od_matrix

##########################################################################################

print('FUA: ', fua_code)

try:
    
    #Get the FUA boundary:
    fua_buffered_boundary = gpd.read_file('../data/d03_intermediate/FUA-buffered-shapefile/FUA-buffered.shp').set_index('fuacode').loc[[fua_code]]
    
    print('  got the boundary')
    
    #Get the commutes within that FUA:
    full_od_matrix = pd.read_csv('../data/d02_processed-safegraph/weeks_od_us_fua.csv')
    fua_raw_od_matrix = full_od_matrix[full_od_matrix.fuacode==fua_code].reset_index(drop=True)
    fua_raw_od_matrix['fuacode'] = fua_code
    print('  got the SafeGraph od matrix')
    
    #Trim rows for which centroids lie outside the FUA:
    od_matrix = trim_centroids(fua_raw_od_matrix, fua_buffered_boundary)
    print('  trimmed the od matrix')
    
    #Get the graphs:
    walk_graph = ox.project_graph(ox.load_graphml('../data/d03_intermediate/FUA-networks/walk/'+fua_code+'.graphml'), to_crs='EPSG:5070')
    drive_graph = ox.project_graph(ox.load_graphml('../data/d03_intermediate/FUA-networks/drive/'+fua_code+'.graphml'), to_crs='EPSG:5070')
    print('  got the street networks')

    #Get the geometries of origin and destinations:
    centroids_pt = gpd.points_from_xy(x= od_matrix.intptlon, y=od_matrix.intptlat, crs='EPSG:4326').to_crs('EPSG:5070')
    od_matrix['origin_x'], od_matrix['origin_y'] = centroids_pt.x, centroids_pt.y
    
    places_pt = gpd.points_from_xy(x= od_matrix.longitude, y=od_matrix.latitude, crs='EPSG:4326').to_crs('EPSG:5070')
    od_matrix['dest_x'], od_matrix['dest_y'] = places_pt.x, places_pt.y
    
    print('  georeferenced origin and destination')
    
    #Get the preferred commute mode:
    od_matrix['mode'] = places_pt.distance(centroids_pt) <= threshold
    od_matrix['mode'] = od_matrix['mode'].map({True: 'walk', False:'drive'})
    print('  got preferred mode of commute')

    #Now we split the dataframe into two (one for walking and one for driving):
    od_matrix_dict = {mode: df for mode, df in od_matrix.groupby('mode')}
    G = {'drive': drive_graph, 'walk': walk_graph}

    #For each of those dataframes, we do nearest nodes from OSMnx on the appropriate graph and the distance:
    full_dfs = []
    for mode, df in od_matrix_dict.items():
        df['origin_node'], df['origin_node_dist'] = ox.nearest_nodes(G[mode], df['origin_x'], df['origin_y'], return_dist=True)
        df['destination_node'], df['destination_node_dist'] = ox.nearest_nodes(G[mode], df['dest_x'], df['dest_y'], return_dist=True)
        df['distance'] = shortest_path_distance(G[mode],
                                                df['origin_node'].values, df['destination_node'].values,
                                                cpus=number_of_cores)
        full_dfs.append(df)    
    
    #Merge these dataframes to obtain the OD matrix with naive network distance:
    od_matrix_naivedistance = pd.concat(full_dfs, ignore_index=True)
    print('  got naive network distance')
    
    #Get the rows that need reworking:
    bad_rows = (od_matrix_naivedistance['mode']=='walk') & (od_matrix_naivedistance['distance'] > threshold)
    print('  got bad rows')
    
    #Set the Boolean value of whether we walk or drive to False in the bad rows:
    od_matrix_naivedistance.loc[bad_rows, 'mode'] = 'drive'

    #We do nearest nodes from OSMnx on the driving graph and the distance for those rows:
    od_matrix_naivedistance.loc[bad_rows, 'origin_node'], od_matrix_naivedistance.loc[bad_rows, 'origin_node_dist'] = ox.nearest_nodes(drive_graph,
                                                                                                                                       od_matrix_naivedistance.loc[bad_rows, 'origin_x'],
                                                                                                                                       od_matrix_naivedistance.loc[bad_rows, 'origin_y'], 
                                                                                                                                       return_dist=True)
    od_matrix_naivedistance.loc[bad_rows, 'destination_node'], od_matrix_naivedistance.loc[bad_rows, 'destination_node_dist'] = ox.nearest_nodes(drive_graph,
                                                                                                                                                 od_matrix_naivedistance.loc[bad_rows, 'dest_x'],
                                                                                                                                                 od_matrix_naivedistance.loc[bad_rows, 'dest_y'], 
                                                                                                                                                 return_dist=True)
    od_matrix_naivedistance.loc[bad_rows, 'distance'] = shortest_path_distance(drive_graph,
                                                                               od_matrix_naivedistance.loc[bad_rows, 'origin_node'].values,
                                                                               od_matrix_naivedistance.loc[bad_rows, 'destination_node'].values,
                                                                               cpus=number_of_cores)
    print('  got final network distance')
    
    #We need to drop some columns (and potentially a few more created by merges and droping indices):
    cols_to_drop = ['origin_x', 'origin_y', 'dest_x', 'dest_y']
    for col in od_matrix_naivedistance.columns:
        if 'Unnamed' in col:
            cols_to_drop.append(col)
    od_matrix_finaldistance = od_matrix_naivedistance.drop(cols_to_drop, axis=1)
    od_matrix_finaldistance.to_csv(outpath)
    print('  saved')

except:
    od_matrix_finaldistance = None
    print('ERROR')

print('RUNTIME:', datetime.now() - start)
