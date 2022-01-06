import sys

import geopandas as gpd
import osmnx as ox

from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

##########################################################################################
#GOAL: Collect FUA street networks and save them to disk

fua_code = sys.argv[1]
start = datetime.now()

##########################################################################################

def save_graph(graph_name, boundary, graph_type):
    #Get the graph
    graph = ox.graph_from_polygon(boundary, network_type=graph_type)
    #Get the filename
    filepath = '/scratch/g.spessatoagostini/FUA-networks/' + graph_type + '/' + graph_name + '.graphml'
    #Save the graph
    ox.save_graphml(graph, filepath=filepath)
    return graph

##########################################################################################

print('FUA: ', fua_code)
fua_buffered_boundary = gpd.read_file(buffered_shp_filepath).set_index('fuacode').loc[[fua_code]]

drive_graph = save_graph(fua_code, fua_buffered_boundary, 'drive')
print('  done for drive')

walk_graph = save_graph(fua_code, fua_buffered_boundary, 'walk')
print('  done for walk')

print('RUNTIME:', datetime.now()-start)