import sys
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import momepy
import pandana

import rtree
import itertools

from shapely.geometry import MultiPoint, LineString
from shapely.ops import snap, split

pd.options.mode.chained_assignment = None

import warnings
warnings.filterwarnings("ignore")

##########################################################################################
#GOAL: Expand the street networks by adding the POIs and centroids of interest

fua_code = sys.argv[1]
start = datetime.now()

d_thresh = 2000
n_thresh = 20

max_distance = 50000
max_pois = 500

##########################################################################################

#Defining input directories:
talia_scratch_dir = '/scratch/kaufmann.t/Processed_data/'
gabriel_scratch_dir = '/scratch/g.spessatoagostini/'
fua_buffered_shapefile_dir = '../data/d03_intermediate/FUA-buffered-shapefile/'
full_od_matrix_dir = '../data/d02_processed-safegraph/'
networks_dir = '../data/d03_intermediate/FUA-networks/'

##########################################################################################

# Source code at https://github.com/ywnch/toolbox/blob/master/toolbox.py (modified)
# Yuwen Chang
# 2020-08-16

def connect_poi(pois, nodes, edges, key_col=None, path=False, threshold=200, knn=5, meter_epsg=5070):
    """Connect and integrate a set of POIs into an existing road network.
    Given a road network in the form of two GeoDataFrames: nodes and edges,
    link each POI to the nearest edge (road segment) based on its projection
    point (PP) and generate a new integrated road network including the POIs,
    the projected points, and the connection edge.
    Args:
        pois (GeoDataFrame): a gdf of POI (geom: Point)
        nodes (GeoDataFrame): a gdf of road network nodes (geom: Point)
        edges (GeoDataFrame): a gdf of road network edges (geom: LineString)
        key_col (str): a unique key column of pois should be provided,
                       e.g., 'index', 'osmid', 'poi_number', etc.
                       Currently, this will be renamed into 'osmid' in the output.
                       [NOTE] For use in pandana, you may want to ensure this
                              column is numeric-only to avoid processing errors.
                              Preferably use unique integers (int or str) only,
                              and be aware not to intersect with the node key,
                              'osmid' if you use OSM data, in the nodes gdf.
        path (str): directory path to use for saving files (nodes and edges).
                      Outputs will NOT be saved if this arg is not specified.
        threshold (int): the max length of a POI connection edge, POIs with
                         connection edge beyond this length will be removed.
                         The unit is in meters as crs epsg is set to 3857 by
                         default during processing.
        knn (int): k nearest neighbors to query for the nearest edge.
                   Consider increasing this number up to 10 if the connection
                   output is slightly unreasonable. But higher knn number will
                   slow down the process.
        meter_epsg (int): preferred EPSG in meter units. Suggested 3857 or 3395.
    Returns:
        nodes (GeoDataFrame): the original gdf with POIs and PPs appended
        edges (GeoDataFrame): the original gdf with connection edges appended
                              and existing edges updated (if PPs are present)
    Note:
        1. Make sure all three input GeoDataFrames have defined crs attribute.
           Try something like `gdf.crs` or `gdf.crs = 'epsg:4326'`.
           They will then be converted into epsg:3857 or specified meter_epsg for processing.
    """

    ## STAGE 0: initialization
    # 0-1: helper functions
    def find_kne(point, lines):
        dists = np.array(list(map(lambda l: l.distance(point), lines)))
        kne_pos = dists.argsort()[0]
        kne = lines.iloc[[kne_pos]]
        kne_idx = kne.index[0]
        return kne_idx, kne.values[0]

    def get_pp(point, line):
        """Get the projected point (pp) of 'point' on 'line'."""
        # project new Point to be interpolated
        pp = line.interpolate(line.project(point))  # PP as a Point
        return pp

    def split_line(line, pps):
        """Split 'line' by all intersecting 'pps' (as multipoint).
        Returns:
            new_lines (list): a list of all line segments after the split
        """
        # IMPORTANT FIX for ensuring intersection between splitters and the line
        # but no need for updating edges_meter manually because the old lines will be
        # replaced anyway
        line = snap(line, pps, 1e-8)  # slow?

        try:
            new_lines = list(split(line, pps))  # split into segments
            return new_lines
        except TypeError as e:
            print('Error when splitting line: {}\n{}\n{}\n'.format(e, line, pps))
            return []

    def update_nodes(nodes, new_points, ptype, meter_epsg=5070, _pois_gdf=None):
        """Update nodes with a list (pp) or a GeoDataFrame (poi) of new_points.
        
        Args:
            ptype: type of Point list to append, 'pp' or 'poi'
            pois_gdf: GeoDataFrame of POIS, if passed we will mark the assigned node id
        """
        # create gdf of new nodes (projected PAPs)
        if ptype == 'pp':
            new_nodes = gpd.GeoDataFrame(new_points, columns=['geometry'], crs=f'epsg:{meter_epsg}')
            n = len(new_nodes)
            new_nodes['highway'] = node_highway_pp
            new_nodes['osmid'] = [int(osmid_prefix + i) for i in range(n)]

        # create gdf of new nodes (original POIs)
        elif ptype == 'poi':
            new_nodes = new_points[['geometry']]
            new_nodes['highway'] = node_highway_poi
            new_nodes['osmid'] = [int(osmid_prefix + i) for i in range(n)]

        else:
            print("Unknown ptype when updating nodes.")
            
        if _pois_gdf is not None:
            _pois_gdf['access_node'] = new_nodes['osmid'].values

        # merge new nodes (it is safe to ignore the index for nodes)
        gdfs = [nodes, new_nodes]
        nodes = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True, sort=False),
                                 crs=gdfs[0].crs)
        if _pois_gdf is not None:
            return nodes, new_nodes, _pois_gdf  # all nodes, newly added nodes only, POI-osmid mapping
        
        else:
            return nodes, new_nodes  # all nodes, newly added nodes only

    def update_edges(edges, new_lines, replace=True):
        """
        Update edge info by adding new_lines; or,
        replace existing ones with new_lines (n-split segments).
        Args:
            replace: treat new_lines (flat list) as newly added edges if False,
                     else replace existing edges with new_lines (often a nested list)
        
        Note:
            kne_idx refers to 'fid in Rtree'/'label'/'loc', not positional iloc
        """
        # for interpolation (split by pp): replicate old line
        if replace:
            # create a flattened gdf with all line segs and corresponding kne_idx
            kne_idxs = list(line_pps_dict.keys())
            lens = [len(item) for item in new_lines]
            new_lines_gdf = gpd.GeoDataFrame({'kne_idx': np.repeat(kne_idxs, lens), 'geometry': list(itertools.chain.from_iterable(new_lines))})
            # merge to inherit the data of the replaced line
            cols = list(edges.columns)
            cols.remove('geometry')  # don't include the old geometry
            new_edges = new_lines_gdf.merge(edges[cols], how='left', left_on='kne_idx', right_index=True)
            new_edges.drop('kne_idx', axis=1, inplace=True)
            new_lines = new_edges['geometry'].values  # now a flatten list
        # for connection (to external poi): append new lines
        else:
            new_edges = gpd.GeoDataFrame(pois.index.values, geometry=new_lines)
            new_edges['oneway'] = False
            new_edges['highway'] = edge_highway

        # update features (a bit slow)
        new_edges['length'] = [l.length for l in new_lines]
        new_edges['from'] = new_edges['geometry'].map(lambda x: nodes_id_dict.get(list(x.coords)[0], None))
        new_edges['to'] = new_edges['geometry'].map(lambda x: nodes_id_dict.get(list(x.coords)[-1], None))
        new_edges['osmid'] = ['_'.join(list(map(str, s))) for s in zip(new_edges['from'], new_edges['to'])]

        # remember to reindex to prevent duplication when concat
        start = edges.index[-1] + 1
        stop = start + len(new_edges)
        new_edges.index = range(start, stop)

        # for interpolation: remove existing edges
        if replace:
            edges = edges.drop(kne_idxs, axis=0)
        # for connection: filter invalid links
        else:
            valid_pos = np.where(new_edges['length'] <= threshold)[0]
            n = len(new_edges)
            n_fault = n - len(valid_pos)
            f_pct = n_fault / n * 100
            print("Remove faulty projections: {}/{} ({:.2f}%)".format(n_fault, n, f_pct))
            new_edges = new_edges.iloc[valid_pos]  # use 'iloc' here

        # merge new edges
        dfs = [edges, new_edges]
        edges = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=False, sort=False), crs=dfs[0].crs)

        # all edges, newly added edges only
        return edges, new_edges

    # 0-2: configurations
    # set poi arguments
    node_highway_pp = 'projected_pap'  # POI Access Point
    node_highway_poi = 'poi'
    edge_highway = 'projected_footway'
    osmid_prefix = int(nodes.osmid.max()) + 1

    # convert CRS
    pois_meter = pois.to_crs(epsg=meter_epsg)
    nodes_meter = nodes.to_crs(epsg=meter_epsg)
    edges_meter = edges.to_crs(epsg=meter_epsg)

    # build rtree
    print("Building rtree...")
    Rtree = rtree.index.Index()
    [Rtree.insert(fid, geom.bounds) for fid, geom in edges_meter['geometry'].iteritems()]

    ## STAGE 1: interpolation
    # 1-1: assign external node IDs (pois)
    #print("Assigning external nodes...")
    #nodes_meter, _new_nodes, poi_keys = update_nodes(nodes_meter, pois_meter, ptype='poi', meter_epsg=meter_epsg, key_col='safegraph_place_id')
    
    # 1-2: update internal nodes (interpolated pps)
    # locate nearest edge (kne) and projected point (pp)
    print("Projecting POIs to the network...")
    pois_meter['near_idx'] = [list(Rtree.nearest(point.bounds, knn))
                              for point in pois_meter['geometry']]  # slow
    
    pois_meter['near_lines'] = [edges_meter['geometry'][near_idx]
                                for near_idx in pois_meter['near_idx']]  # very slow
    pois_meter['kne_idx'], knes = zip(*[find_kne(point, near_lines) for point, near_lines in zip(pois_meter['geometry'], pois_meter['near_lines'])])  # slow
    pois_meter['pp'] = [get_pp(point, kne) for point, kne in zip(pois_meter['geometry'], knes)]

    # update nodes
    print("Assigning nodes to POIs and updating...")
    nodes_meter, new_nodes_meter, pois_meter = update_nodes(nodes_meter, list(pois_meter['pp']), ptype='pp', meter_epsg=meter_epsg, _pois_gdf=pois_meter)
    nodes_coord = nodes_meter['geometry'].map(lambda x: x.coords[0])
    nodes_id_dict = dict(zip(nodes_coord, nodes_meter['osmid'].astype('Int64')))

    # 1-3: update internal edges (split line segments)
    print("Updating internal edges...")
    # split
    line_pps_dict = {k: MultiPoint(list(v)) for k, v in pois_meter.groupby(['kne_idx'])['pp']}
    new_lines = [split_line(edges_meter['geometry'][idx], pps) for idx, pps in line_pps_dict.items()]  # bit slow
    edges_meter, _ = update_edges(edges_meter, new_lines, replace=True)

    ## STAGE 2: connection
    # 2-1: add external distances (projected footways connected to pois) to the POIs
    print("Adding external distance...")
    pps_gdf = nodes_meter[nodes_meter['highway'] == node_highway_pp]
    new_lines = [LineString([p1, p2]) for p1, p2 in zip(pois_meter['geometry'], pps_gdf['geometry'])]
    pois_meter['access_distance'] = [l.length for l in new_lines]
    #edges_meter, _ = update_edges(edges_meter, new_lines, replace=False)

    ## STAGE 3: output
    # convert CRS
    nodes = nodes_meter.to_crs(epsg=4326)
    edges = edges_meter.to_crs(epsg=4326)

    # preprocess for pandana
    nodes.index = nodes['osmid']  # IMPORTANT
    nodes['x'] = [p.x for p in nodes['geometry']]
    nodes['y'] = [p.y for p in nodes['geometry']]

    # edges.reset_index(drop=True, inplace=True)
    edges['length'] = edges['length'].astype(float)

    # report issues
    # - examine key duplication
    if len(nodes_meter) != len(nodes_id_dict):
        print("NOTE: duplication in node coordinates keys")
        print("Nodes count:", len(nodes_meter))
        print("Node coordinates key count:", len(nodes_id_dict))
    # - examine missing nodes
    print("Missing 'from' nodes:", len(edges[edges['from'] == None]))
    print("Missing 'to' nodes:", len(edges[edges['to'] == None]))

    # save and return
    if path:
        nodes.to_file(path+'_nodes.shp')
        edges.to_file(path+'_edges.shp')

    return nodes, edges, pois_meter

##########################################################################################

def get_boundary(fua_code):
    return gpd.read_file(fua_buffered_shapefile_dir + 'FUA-buffered.shp').set_index('fuacode').loc[[fua_code]]

def get_fua_ODmatrix(fua_code):
    full_od_matrix = pd.read_csv(full_od_matrix_dir + 'weeks_od_us_fua.csv')
    fua_raw_od_matrix = full_od_matrix[full_od_matrix.fuacode==fua_code].reset_index(drop=True)
    fua_raw_od_matrix['fuacode'] = fua_code
    return fua_raw_od_matrix

def load_graphs(fua_code, proj_crs='EPSG:5070'):
    walk_graph = ox.project_graph(ox.load_graphml(networks_dir + 'walk/'+fua_code+'.graphml'), to_crs=proj_crs)
    return walk_graph

def trim_centroids(od_matrix, buffered_boundary_gdf):
    buffered_boundary = buffered_boundary_gdf.geometry[0]
    centroids_pt = gpd.points_from_xy(x= od_matrix.intptlon, y=od_matrix.intptlat, crs='EPSG:4326')
    rows_to_keep = centroids_pt.within(buffered_boundary)
    trimmed_od_matrix = od_matrix[rows_to_keep].reset_index(drop=True)
    return trimmed_od_matrix

def get_pois(fua_code):
    pois = pd.read_csv(full_od_matrix_dir+'POI_fua_sub.csv')
    fua_pois = pois[pois.fuacode==fua_code].reset_index(drop=True)
    return fua_pois

def get_files(fua_code):
    fua_buffered_boundary = get_boundary(fua_code) #get the FUA boundary
    fua_raw_od_matrix = get_fua_ODmatrix(fua_code) #get the commutes within that FUA
    fua_od_matrix = trim_centroids(fua_raw_od_matrix, fua_buffered_boundary) #exclude far away centroids
    
    all_cbgs = fua_od_matrix[['census_block_group', 'intptlon', 'intptlat']]
    cbgs = all_cbgs.drop_duplicates(subset='census_block_group', ignore_index=True)
    
    pois = get_pois(fua_code)
    
    graph = load_graphs(fua_code) #get the graph
    
    return cbgs, pois, graph

def add_median_columns(all_cbgs_df, fua_cbgs_gdf):
    relevant_cols = all_cbgs_df[['census_block_group', 'top_category', 'median_dist']].set_index('census_block_group')
    relevant_dict = dict(tuple(relevant_cols.groupby('top_category')))
    for cat, df_with_cat in relevant_dict.items():
        col_name = 'md_' + cat.replace(' ','_')
        df = df_with_cat[['median_dist']].rename({'median_dist':col_name}, axis=1)
        fua_cbgs_gdf = fua_cbgs_gdf.merge(df, how='left', left_index=True, right_index=True)
    return fua_cbgs_gdf

##########################################################################################

def trim_by_number(isochron_df, number):
    max_number = int(list(isochron_df.columns)[-1][3:])
    number = min(number, max_number)
    
    distances_df = isochron_df.iloc[:, 2:max_number+2]
    ids_df = isochron_df.iloc[:, max_number+2:]
    trimmed_df = isochron_df[[]]
    
    if number < max_number:
        dists_to_drop = [k for k in range(number+1, max_number)]
        ids_to_drop = ['poi{}'.format(k) for k in range(number+1, max_number)]
        
        trimmed_dists = distances_df.drop(labels=dists_to_drop, axis=1)
        trimmed_df['nearest_pois_distances'] = trimmed_dists.values.tolist()
        
        trimmed_ids = ids_df.drop(labels=ids_to_drop, axis=1)
        trimmed_df['nearest_pois_ids'] = trimmed_ids.values.tolist()
        
    else:
        trimmed_df['nearest_pois_distances'] = distances_df.values.tolist()
        trimmed_df['nearest_pois_ids'] = ids_df.values.tolist()
    
    return trimmed_df

def trim_by_dist(isochron_df, dist, max_dist=100000, max_pois=max_pois):
    max_number = int(list(isochron_df.columns)[-1][3:])
    dist = min(dist, max_dist)
    
    distances_lists_df = trim_by_number(isochron_df, max_pois)
    trimmed_df = isochron_df[[]]
    
    from bisect import bisect
    def get_trim_number(x, val=dist):
        return bisect(x, val)
    distances_lists_df['number'] = distances_lists_df['nearest_pois_distances'].apply(get_trim_number)
    
    def trim(l, end):
        return l[:end]
    
    trimmed_df['nearest_pois_distances'] = distances_lists_df.apply(lambda x: trim(x.nearest_pois_distances, x.number), axis=1)
    trimmed_df['nearest_pois_ids'] = distances_lists_df.apply(lambda x: trim(x.nearest_pois_ids, x.number), axis=1)
    
    return trimmed_df

def trim_by_median(isochron_df, max_median=100000, max_pois=max_pois):
    max_number = int(list(isochron_df.columns)[-1][3:])
    
    distances_lists_df = trim_by_number(isochron_df, max_pois)
    distances_lists_df['md'] = isochron_df.iloc[:,2]
    trimmed_df = isochron_df[[]]
    trimmed_df['cbg_mobility_median'] = isochron_df.iloc[:,2]
    
    from bisect import bisect
    def get_trim_number(x, val):
        return bisect(x, val)
    distances_lists_df['number'] = distances_lists_df.apply(lambda x: get_trim_number(x.nearest_pois_distances,
                                                                                      x.md), axis=1)
    
    def trim(l, end):
        return l[:end]
    
    trimmed_df['nearest_pois_distances'] = distances_lists_df.apply(lambda x: trim(x.nearest_pois_distances, x.number), axis=1)
    trimmed_df['nearest_pois_ids'] = distances_lists_df.apply(lambda x: trim(x.nearest_pois_ids, x.number), axis=1)
    
    return trimmed_df

def summarize_isochron(isochron_df, trim_method, trim_val=None, cat=None, fua_code='USA11'):
    """
    Summarizes the isochron dataframe according to our thresholding method
    
    :param isochron_df: dataframe with columns [origin_node, md_cat, 1, ... 100, poi1, ... poi100]
    :param trim_method: string, one of 'fixed_dist', 'fixed_number', 'median'
    """
    if cat is None:
        cat = [s for s in list(isochron_df.columns) if 'md' in str(s)][0][3:]
    
    if trim_method == 'fixed_dist':
        trimmed_df = trim_by_dist(isochron_df, trim_val)
    elif trim_method == 'fixed_number':
        trimmed_df = trim_by_number(isochron_df, trim_val)
    elif trim_method == 'median':
        trimmed_df = trim_by_median(isochron_df)
    else:
        print('NOT ABLE TO TRIM!')
        trimmed_df = isochron_df
    
    #Compute summary statistics:
    def median_nan(arr):
        try:
            return np.median(arr)
        except:
            return np.nan
    def min_nan(arr):
        try:
            return np.min(arr)
        except:
            return np.nan
    def max_nan(arr):
        try:
            return np.max(arr)
        except:
            return np.nan
    
    trimmed_df['median_dist'] = trimmed_df['nearest_pois_distances'].apply(median_nan)
    trimmed_df['min_dist'] = trimmed_df['nearest_pois_distances'].apply(min_nan)
    trimmed_df['max_dist'] = trimmed_df['nearest_pois_distances'].apply(max_nan)
    trimmed_df['count_dist'] = trimmed_df['nearest_pois_distances'].apply(len)
    
    #Save:
    filename = fua_code + '_' + cat + '_isochron_' + trim_method + '.csv'
    trimmed_df.to_csv('/work/accessibility/Isochrons/'+trim_method+'/'+filename)
    
    return trimmed_df

##########################################################################################

print('FUA: ', fua_code, '\n')

try:
    
    #1. LOAD ALL THE FILES:
    start=datetime.now()
    
    cbgs, pois, graph = get_files(fua_code)
    full_cbg_stats = pd.read_csv(talia_scratch_dir+'cbg_links_stats.csv')

    loading_complete=datetime.now()
    print(' Loaded all files in:', loading_complete-start)
    
    #2. GET THE GDFS FROM THE GRAPH:
    nodes_ind, edges_multi = ox.utils_graph.graph_to_gdfs(graph)
    edges_raw = edges_multi.reset_index()

    nodes_clean, edges_clean = momepy.nx_to_gdf(momepy.gdf_to_nx(edges_raw.explode(index_parts=True)))
    edges = edges_clean[['node_start', 'node_end', 'mm_len', 'geometry']].rename({'node_start':'from', 'node_end':'to',
                                                                                  'mm_len':'length'}, axis=1)
    nodes = nodes_clean.rename({'nodeID':'osmid'}, axis=1)
    
    gdfs_complete=datetime.now()
    print(' Converted graph to gdf in:', gdfs_complete-loading_complete)
    
    #3. GEOREFERENCE THE POINTS OF INTEREST:
    pois_pt = gpd.points_from_xy(x=pois.longitude, y=pois.latitude, crs='EPSG:4326')
    pois_gdf = gpd.GeoDataFrame(pois[['safegraph_place_id', 'top_category']], geometry=pois_pt)
    pois_gdf.drop_duplicates(subset='safegraph_place_id', keep='first', inplace=True)
    pois_gdf = pois_gdf.set_index('safegraph_place_id')

    cbgs_pt = gpd.points_from_xy(x=cbgs.intptlon, y=cbgs.intptlat, crs='EPSG:4326')
    cbgs_gdf = gpd.GeoDataFrame(cbgs[['census_block_group']], geometry=cbgs_pt)
    cbgs_gdf.drop_duplicates(subset='census_block_group', keep='first', inplace=True)
    cbgs_gdf = cbgs_gdf.set_index('census_block_group')
    cbgs_withmedians_gdf = add_median_columns(full_cbg_stats, cbgs_gdf)
    cbgs_withmedians_gdf = cbgs_withmedians_gdf.fillna(value=2000)

    georeferencing_complete=datetime.now()
    print(' Georeferenced POIs in:', georeferencing_complete-gdfs_complete)
    
    #3. EXPAND THE GRAPH:
    exp_nodes, exp_edges, pois_with_access = connect_poi(pois_gdf, nodes, edges)

    graph_complete=datetime.now()
    print(' Expanded graph in:', graph_complete-georeferencing_complete)
    
    #4. BUILD THE NETWORK IN PANDANA:
    expanded_graph = pandana.Network(exp_nodes.geometry.x, exp_nodes.geometry.y,
                                     exp_edges['from'], exp_edges['to'],
                                     edge_weights=exp_edges[['length']])
    
    pandana_complete=datetime.now()
    print(' Built pandana network in:', pandana_complete-graph_complete)
    
    #5. BUILD POI DICTIONARY:
    pois_gdf['x'] = pois_gdf.geometry.x
    pois_gdf['y'] = pois_gdf.geometry.y
    pois_with_access_gdf = pois_gdf.merge(pois_with_access[['access_node', 'access_distance']], left_index=True, right_index=True, how='left')
    pois_with_access_dict = dict(tuple(pois_with_access_gdf.groupby('top_category')))

    poi_dictionary_complete=datetime.now()
    print(' Found POI dictionary in:', poi_dictionary_complete-pandana_complete)
    
    #7. SUMMARIZE OVER CATEGORIES:
    for category in pois_with_access_dict.keys():
        print('-- Category: ', category)

        median_col = 'md_'+category.replace(' ', '_')
        if median_col in list(cbgs_withmedians_gdf.columns):
        
            expanded_graph.set_pois(category,
                                    maxdist=max_distance,
                                    maxitems=max_pois,
                                    x_col=pois_with_access_dict[category]['x'],
                                    y_col=pois_with_access_dict[category]['y'])

            cbgs_withmedians_gdf['origin_node'] = expanded_graph.get_node_ids(x_col=cbgs_withmedians_gdf['geometry'].x,
                                                                              y_col=cbgs_withmedians_gdf['geometry'].y)

            d = expanded_graph.nearest_pois(max_distance, category,
                                            num_pois=max_pois,
                                            imp_name='length',
                                            include_poi_ids=True)

            isochron_df = cbgs_withmedians_gdf[['origin_node', median_col]].merge(d, left_on='origin_node', right_index=True, how='left')
        
            isochron_complete=datetime.now()
            print(' Found the isochrons in:', isochron_complete-poi_dictionary_complete)
        
            summarized_dist = summarize_isochron(isochron_df, 'fixed_dist', trim_val=d_thresh)
            summarized_numb = summarize_isochron(isochron_df, 'fixed_number', trim_val=n_thresh)
            summarized_median = summarize_isochron(isochron_df, 'median')

            summary_complete = datetime.now()
            print(' Summarized the isochrons in:', summary_complete-isochron_complete)
            
    else:
        print('  category median not given')
        
    print('TOTAL TIME: ', datetime.now()-start)
        
except Exception as e:
    print(e)