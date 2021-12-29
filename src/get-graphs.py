import geopandas as gpd
import osmnx as ox

import pickle as pkl
from tqdm import tqdm

##########################################################################################
#GOAL: Collect FUA street networks and save them to disk

buffered_shp_filepath = '../data/d03_intermediate/FUA-buffered-shapefile/FUA-buffered.shp'

walkgraph_dict_filepath = '../data/d03_intermediate/FUA-networks-dictionary-pkl/walk-graphs.pkl'
drivegraph_dict_filepath = '../data/d03_intermediate/FUA-networks-dictionary-pkl/drive-graphs.pkl'

##########################################################################################

#Do it for walk and then for drive:
def get_graph(boundary, graph_type):
    return ox.graph_from_polygon(boundary, network_type=graph_type)

def save(file, filepath):
    with open(filepath, 'wb') as f:
        pkl.dump(file, f)
    return None

def create_dictionary(graph_type, fua_codes, fua_boundaries, directory='../data/d03_intermediate/FUA-networks-dictionary-pkl/', filename=None):
    if filename is None:
        filename = graph_type + '-graphs.pkl'
    filepath = directory+filename
    
    #If the file already exists we can continue from where we stopped:
    try:
        with open(filepath, 'rb') as f:
            dictionary = pkl.load(f)
            keys = list(dictionary.keys())
    except:
        dictionary = {}
        keys = []
    
    remaining_keys = list(set(fua_codes)-set(keys))
    
    #Iterate over the remaining keys:
    for fua_code, fua_boundary in tqdm(zip(fua_codes, fua_boundaries), total=len(fua_codes)):
        #Only do something if we need to:
        if fua_code in remaining_keys:
            graph = get_graph(fua_boundary, graph_type)
            dictionary[fua_code] = graph
            #Save dictionary:
            save(dictionary, filepath)
            
    return dictionary



##########################################################################################

#Read the Geodataframe of Buffered FUAs:
gdf = gpd.read_file(buffered_shp_filepath)[:5].set_index('fuacode')
fua_codes = gdf.index.values
fua_buffered_boundaries = gdf.geometry.values

d = create_dictionary('walk', fua_codes, fua_buffered_boundaries)