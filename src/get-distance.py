import os
import sys
from multiprocessing import Pool, cpu_count

import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox

import warnings
warnings.filterwarnings("ignore")

##########################################################################################
#GOAL: Prepare the OD matrices of each FUA for the distance computations

fua_code = sys.argv[1]
threshold = 2000
number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])

##########################################################################################
#Functions adapted from OSMnx:

def _single_shortest_path_distance(G, orig, dest, weight):
    """
    Get shortest path distance from an origin node to a destination node.
    This function is a convenience wrapper around networkx.shortest_path, with
    exception handling for unsolvable paths.
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    orig : int
        origin node ID
    dest : int
        destination node ID
    weight : string
        edge attribute to minimize when solving shortest path
    Returns
    -------
    dist : float
        shortest (weighted) distance between origin and destination nodes
    """
    try:
        return nx.shortest_path_length(G, orig, dest, weight=weight) #change function here from G. Boeing's repo
    except nx.exception.NetworkXNoPath:
        return None

def shortest_path_distance(G, orig, dest, weight="length", cpus=1):
    """
    Get shortest path distance from origin node(s) to destination node(s).
    If `orig` and `dest` are single node IDs, this will return a list of the
    nodes constituting the shortest path between them.  If `orig` and `dest`
    are lists of node IDs, this will return a list of lists of the nodes
    constituting the shortest path between each origin-destination pair. If a
    path cannot be solved, this will return None for that path. You can
    parallelize solving multiple paths with the `cpus` parameter, but be
    careful to not exceed your available RAM.
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    orig : int or list
        origin node ID, or a list of origin node IDs
    dest : int or list
        destination node ID, or a list of destination node IDs
    weight : string
        edge attribute to minimize when solving shortest path
    cpus : int
        how many CPU cores to use; if None, use all available
    Returns
    -------
    path : list
        list of node IDs constituting the shortest path, or, if orig and dest
        are lists, then a list of path lists
    """
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

print('FUA: ', fua_code)
if fua_code not in ['USA01', 'USA02', 'USA56']:
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

    #For each of those dataframes, we do nearest nodes from OSMnx on the appropriate graph and the distance:
    full_dfs = []
    for walk, df in od_matrix_dict.items():
        df['origin_node'] = ox.nearest_nodes(G[walk], df['origin_x'], df['origin_y'])
        df['destination_node'] = ox.nearest_nodes(G[walk], df['dest_x'], df['dest_y'])
        df['distance'] = shortest_path_distance(G[walk],
                                                df['origin_node'].values, df['destination_node'].values,
                                                cpus=number_of_cores)
        full_dfs.append(df)

    merged_df = pd.concat(full_dfs, ignore_index=True)
    expanded_OD = merged_df.drop(['origin_x', 'origin_y', 'dest_x', 'dest_y'], axis=1)
    print('  got expanded matrix')

    expanded_OD.to_csv('/scratch/g.spessatoagostini/OD-per-FUA/' + fua_code+'_full-ODmatrix.csv')
    print('  saved')

else:
    expanded_OD = None
