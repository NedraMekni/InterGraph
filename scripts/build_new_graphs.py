import torch
import sys
import os
from dotenv import load_dotenv
from InterGraph.pdb_to_multigraph import (
    get_cache,
    biograph_from_file,
    valid_pdb,
    preprocess_csv,
    data_activities_from_file,
    filter_with_y,
    build_graph_dict,
    save_graphs,
    write_cache,
)

load_dotenv()
ROOT_PATH = os.getenv("ROOT_PATH")

if __name__ == "__main__":
    # valid_pdb(ROOT_PATH+"/PDB/data/")
    get_cache(ROOT_PATH + "/.cachePDB", ROOT_PATH + "/cached_graph")
    graph_dict, cache = biograph_from_file(ROOT_PATH + "/external_PDB/data/")
    write_cache(ROOT_PATH + "/.cachePDB", cache)

    print("After biograph_from_file graph dict len = {}".format(len(graph_dict.keys())))
    if(len(graph_dict.keys())==0):
        print('NO NEW GRAPHS IN THIS ROUND')
        exit(1)
    #preprocess_csv(ROOT_PATH + "/csv/data.csv", ROOT_PATH + "/csv/y_preprocessed.csv")
    y_dict = data_activities_from_file(
        ROOT_PATH + "/external_csv", ROOT_PATH + "/external_csv/y_preprocessed_ki_w.csv", False
    )
    graph_dict = filter_with_y(graph_dict, y_dict, "")
    print("After filter_with_y graph dict len = {}".format(len(graph_dict.keys())))
    if(len(graph_dict.keys())==0):
        print('NO NEW GRAPHS IN THIS ROUND')
    graph_dict = {k: graph_dict[k] for k in list(graph_dict.keys())[:]}
    print("After comprehension graph dict len = {}".format(len(graph_dict.keys())))
    
    global_G = build_graph_dict(graph_dict, y_dict, ROOT_PATH, pre = "")
    print("After build_graph_dict global_G len = {}".format(len(global_G.keys())))
    if(len(global_G.keys())==0):
        print('NO NEW GRAPHS IN THIS ROUND')
    save_graphs(global_G, ROOT_PATH + "/cached_graph")
    print("BUILT NEW GRAPHS: ", len(global_G.keys()))
    exit(0)  # mda exception caused by forcing fclose
