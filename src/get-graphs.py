import geopandas as gpd
import osmnx as ox

from multiprocessing import Pool
import pickle as pkl

##########################################################################################
#GOAL: Collect FUA street networks and save them to disk

buffered_shp_filepath = '../data/d03_intermediate/FUA-buffered-shapefile/FUA-buffered.shp'

walkgraph_dict_filepath = '../data/d03_intermediate/FUA-networks-dictionary-pkl/walk-graphs.pkl'
drivegraph_dict_filepath = '../data/d03_intermediate/FUA-networks-dictionary-pkl/drive-graphs.pkl'

##########################################################################################

def get_graph(boundary, graph_type):
    return ox.graph_from_polygon(boundary, network_type=graph_type)

#Read the Geodataframe of Buffered FUAs:
gdf = gpd.read_file(buffered_shp_filepath)[:20].set_index('fuacode')
fua_codes = gdf.index.values
fua_buffered_boundaries = gdf.geometry.values

#Do it for walk and then for drive:
for net_type, out_path in zip(['walk', 'drive'], [walkgraph_dict_filepath, drivegraph_dict_filepath]):
    #Parallelize over some CPUs:
    with Pool(20) as p:
        #Get the argument list:
        args_generator = ((bdry, net_type) for bdry in fua_buffered_boundaries)
        #Build the dictionary:
        dictionary = dict(zip(fua_codes, p.starmap(get_graph, args_generator)))
        #Save:
        with open(out_path, 'wb') as file:
            pkl.dump(dictionary, file)