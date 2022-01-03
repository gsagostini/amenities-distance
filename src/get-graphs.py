import os
import sys
from multiprocessing import Pool

from geopandas import read_file
from osmnx import graph_from_polygon, save_graphml

import warnings
warnings.filterwarnings("ignore")

##########################################################################################
#GOAL: Collect FUA street networks and save them to disk

buffered_shp_filepath = '../data/d03_intermediate/FUA-buffered-shapefile/FUA-buffered.shp'

# function you want to run in parallel:
def save_graph(graph_name, boundary, graph_type=sys.argv[1]):
    #Get the graph
    graph = graph_from_polygon(boundary, network_type=graph_type)
    #Get the filename
    filepath = 'graphs/' + graph_type + '/' + graph_name + '.graphml'
    #Save the graph
    save_graphml(graph, filepath=filepath)
    #Document
    print(graph_name) 
    return graph

# list of tuples to serve as arguments to function:
gdf = read_file(buffered_shp_filepath).set_index('fuacode')
fua_codes = gdf.index.values
fua_buffered_boundaries = gdf.geometry.values
args = list(zip(fua_codes, fua_buffered_boundaries))

# number of cores you have allocated for your slurm task:
number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])

# multiprocssing pool to distribute tasks to:
with Pool(number_of_cores) as pool:
    # distribute computations and collect results:
    results = pool.starmap(save_graph, args)
