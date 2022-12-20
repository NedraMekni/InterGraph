import math
import os
import itertools
import pandas as pd
import numpy as np
import torch
import csv
from Bio.PDB import *
import MDAnalysis as mda

from torch_geometric.data import Data
from functools import reduce

parser = PDBParser()
device = "cpu"  # torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

CACHE_PDB = None


def get_cache(cache_fname: str, cached_graph: str) -> None:

    """
    Get the cached PDB files from a cache file or create a new cache.

    Parameters:
    - cache_fname (str): A string specifying the cache file name.
    - cached_graph (str): A string specifying the directory for the cached PDB files.

    Returns:
    None
    """
    global CACHE_PDB
    if os.path.isfile(cache_fname):
        if not os.path.exists(cached_graph):
            os.mkdir(cached_graph)
        with open(cache_fname, "r") as f:
            lines = f.readlines()
            CACHE_PDB = [x.strip() for x in lines]

    else:
        if os.path.exists(cached_graph):
            os.system("rm -r " + cached_graph)
        os.mkdir(cached_graph)
        CACHE_PDB = []

    print("CACHE PDB = {}".format(CACHE_PDB))


# max_t and min_t are nano
def preprocess_csv(fname: str, outfname: str, max_t=100000, min_t=10) -> None:
    """
    Preprocess a CSV file by filtering rows based on activity values and saving the resulting dataframe to a new file.

    Parameters:
    - fname (str): A string specifying the input CSV file name.
    - outfname (str): A string specifying the output CSV file name.
    - max_t (int, optional): An integer specifying the maximum activity value to include in the output dataframe. Default is 100000nM.
    - min_t (int, optional): An integer specifying the minimum activity value to include in the output dataframe. Default is 10nM.

    Returns:
    None
    """
    df = pd.read_csv(fname, usecols=[" pdb_code", " activity"])
    df = df.mask(df.eq(" None")).dropna()
    df = df.astype({" activity": "float64"})
    df = df[df[" activity"].between(min_t, max_t)]
    assert max(df[" activity"]) <= max_t and min(df[" activity"]) >= min_t
    new_df_cap = df.copy()
    new_df_cap.to_csv(outfname, index=False)


def valid_pdb(dirname: str) -> None:
    my_files = list(os.walk(dirname))[0]

    for z in my_files[2]:
        if "_H.pdb" not in z:
            continue
        with open(dirname + "/" + z, "r") as f:
            lines = f.readlines()
            if (
                not any(
                    [row.split()[0] == "ATOM" for row in lines if len(row.split()) > 0]
                )
            ) or (
                not any(
                    [
                        row.split()[0] == "HETATM"
                        for row in lines
                        if len(row.split()) > 0
                    ]
                )
            ):
                print("removing file {}".format(z))
                os.remove(dirname + "/" + z)


def biograph_from_file(dirname: str) -> dict:
    """
    Generate graph dict from PDB files
    graph dict:
    -key: PDB file name -> str
    -value: graph -> tuple of dicts
    """
    global CACHE_PDB
    graph_dict = {}
    my_files = list(os.walk(dirname))[0]
    print("my files before", my_files[2])
    my_files = [x for x in my_files[2] if x not in CACHE_PDB]
    print("my files ", my_files)
    if len(my_files) == 0:
        print("NO NEW PDB FOUND")
        exit(0)
    cntPdb = 0
    for z in my_files:
        nodes_p_entry = {}
        nodes_p = {}
        nodes_i = {}
        elements_p = {}
        if "_H.pdb" not in z:
            continue
        structure = parser.get_structure(z, dirname + "/" + z)
        print("building ", z)
        i = 0
        for atom in structure.get_atoms():
            coord = atom.coord
            nodes_p[atom.serial_number] = atom
            nodes_i[atom.serial_number] = i
            elements_p[atom.serial_number] = atom.element
            # check if atom is hetatm -> https://stackoverflow.com/questions/25718201/remove-heteroatoms-from-pdb
            tags = atom.get_full_id()
            nodes_p_entry[atom.serial_number] = (
                "HETATM" if tags[3][0] != " " else "ATOM"
            )
            i += 1

        try:  # for hydro
            u = mda.Universe(dirname + "/" + z)
            if not hasattr(u, "atoms") or not all(
                [hasattr(atom, "bonds") for atom in u.atoms]
            ):
                # clean_fd_mda(dirname + "/" + z)
                continue

        except:
            continue

        graph_dict[z] = (nodes_p, nodes_p_entry, u.atoms, elements_p, nodes_i)
        cntPdb += 1

    print("Tot pdb = {}".format(cntPdb))
    return graph_dict, my_files


def data_activities_from_file(dirname, fname: str) -> dict:
    """
    Extract data and activities from a file.

    Parameters:
    - dirname (str): A string specifying the directory name.
    - fname (str): A string specifying the file name.

    Returns:
    - dict: A dictionary containing data and activities extracted from the file.
    """

    label = pd.read_csv(fname, usecols=[" pdb_code", " activity"])
    label.to_csv(dirname + "/test_y.csv", index=False)
    all_rows = []
    y_dict = {}
    with open(dirname + "/test_y.csv", "r") as f:
        lines = f.readlines()[1:]

        for l in lines:
            elements = l.split(",")
            elements[0] = elements[0].strip()
            elements[1] = elements[1].strip()

            for k in elements[0].split(" "):

                if k not in y_dict.keys():
                    y_dict[k] = []

                try:
                    v = float(elements[1])
                    v = v * (10**-9)
                    v = -math.log10(v)
                    y_dict[k].append(v)

                except:
                    # print("### cast error, skip activity reading")
                    continue

    return y_dict


def filter_with_y(graph_dict: dict, y_dict: dict) -> dict:
    return {
        k: graph_dict[k]
        for k in graph_dict.keys()
        if k.split("_")[0] + "_1" in y_dict.keys()
    }


def get_ohe_order(fname: str) -> list:
    """
    Get the order of the one-hot encoded features from a file.

    Parameters:
    - fname (str): A string specifying the file name.

    Returns:
    - list: A list of strings containing the one-hot encoded feature names in the order they appear in the file.
    """
    with open(fname, "r") as f:
        return [x.strip() for x in f.readlines()]


def update_old_dict(
    diff_ohe_atom_len: int, diff_ohe_element_len: int, graph_dir: str
) -> None:
    """
    Update the attributes of one-hot encoded atoms and elements in a dictionary of graphs stored in a directory.

    Parameters:
    - diff_ohe_atom_len (int): An integer specifying the number of additional one-hot encoded atom attributes to add.
    - diff_ohe_element_len (int): An integer specifying the number of additional one-hot encoded element attributes to add.
    - graph_dir (str): A string specifying the directory containing the stored graphs.

    Returns:
    None
    """
    files = os.listdir(graph_dir + "/cached_graph")
    files = [
        graph_dir + "/cached_graph/" + x for x in files if x[-3:] == ".pt"
    ]  # select all .pt files

    for f in files:
        print("opening {}".format(f))
        global_g = torch.load(f)
        print("after load {}".format(f))

        for k in global_g.keys():
            local_g = global_g[k]
            for el in local_g[1].keys():
                local_g[1][el]["attributes"][1] += [0] * diff_ohe_atom_len
                local_g[1][el]["attributes"][2] += [0] * diff_ohe_element_len
        print("removing {}".format(f))

        os.remove(f)  # maybe it does not overwrite on save
        print("saving {}".format(f))

        torch.save(global_g, f)


def store_element_cache(fname: str, cache_list: list) -> None:
    with open(fname, "w") as f:  # no append
        for el in cache_list:
            f.write(el + "\n")


def build_graph_dict(graph_dict, y_dict: dict, ohe_path: str) -> dict:
    """
    Build a graph dictionary from a y dictionary and a path to an OHE file.

    Parameters:
    - graph_dict (dict): A dictionary containing graph information.
    - y_dict (dict): A dictionary containing y values.
    - ohe_path (str): A string specifying the path to the OHE file.

    Returns:
    - dict: A dictionary containing updated graph information.
    """
    global_G = {}
    graph_x = {}
    print("IN BUILD GRAPH")

    # helper structures for one hot encoding labels and for cast dictionary graph into list
    atom_type_list = [
        [graph_dict[k][0][k1].name for k1 in graph_dict[k][0].keys()]
        for k in graph_dict.keys()
    ]

    atom_type_list = list(set(reduce(lambda x, y: x + y, atom_type_list)))

    element_list = [
        [graph_dict[k][3][k1] for k1 in graph_dict[k][3].keys()]
        for k in graph_dict.keys()
    ]
    element_list = list(set(reduce(lambda x, y: x + y, element_list)))

    # retrieve the order
    # get the new order
    # merge
    # appending a series of 0s to the end of the one-hot encoding
    # set the new order to Element_list and atom_type_list

    print("len cache pdb ", len(CACHE_PDB))

    if len(CACHE_PDB) > 0:
        print("before read get_ohe_order")
        print("reading ", ohe_path + "/.ohe_atom_order")
        old_atom_order = get_ohe_order(ohe_path + "/.ohe_atom_order")
        old_element_order = get_ohe_order(ohe_path + "/.ohe_element_order")
        print("after read get_ohe_order")
        diff_ohe_atom = [x for x in atom_type_list if x not in old_atom_order]
        diff_ohe_element = [x for x in element_list if x not in old_element_order]
        if len(diff_ohe_atom) == 0:
            atom_type_list = old_atom_order.copy()
            element_list = old_element_order.copy()
        else:
            # update_old_dict
            update_old_dict(len(diff_ohe_atom), len(diff_ohe_element), ohe_path)
            atom_type_list = old_atom_order.copy()
            atom_type_list += diff_ohe_atom

            element_list = old_element_order.copy()
            element_list += diff_ohe_element

    print("after ohe read")
    global_node_list_order = {
        k: list(graph_dict[k][0].keys()) for k in graph_dict.keys()
    }

    print("building adjacence list for each graph")
    graph_dict = {
        k: graph_dict[k]
        for k in graph_dict.keys()
        if k.split("_")[0] + "_1" in y_dict.keys()
    }
    for fname in graph_dict.keys():
        print("building adjacence list of ", fname)
        my_edges = []
        # node_p and node_p_etry are two dictionaries storing nodes

        nodes_p = graph_dict[fname][0]
        nodes_p_entry = graph_dict[fname][1]
        atom_p = graph_dict[fname][2]
        elements_p = graph_dict[fname][3]
        nodes_i = graph_dict[fname][4]

        ns = NeighborSearch(list(nodes_p.values()))
        node_list_order = global_node_list_order[fname]  # to build adj list

        # bulding H label for each node
        # for each atom in atom struct extract the n of H
        hydrogen_label = {}
        for atom in atom_p:
            neighbour_atoms = list(atom.bonds)
            n_hydro = len(
                [bond[1].element for bond in neighbour_atoms if bond[1].element == "H"]
            )
            n_hydro_hot = [0] * 5
            assert n_hydro < 5
            n_hydro_hot[n_hydro] = 1
            hydrogen_label[atom.id] = n_hydro_hot

        # Multigraph generation
        # Find adjacencies for each node in nodes_p and add them to my_edges

        for k in nodes_p.keys():
            node = nodes_p[k]
            adjs = [h for h in ns.search(node.get_coord(), 3, "A")]
            adjs += [h for h in ns.search(node.get_coord(), 6, "A")]
            adjs += [h for h in ns.search(node.get_coord(), 9, "A")]
            my_edges += [
                (nodes_i[node.serial_number], nodes_i[adj.serial_number])
                for adj in adjs
            ]

        print("end treshold loop ", fname)
        # build pytorch data graph
        nodes_p_final = list(nodes_p.keys())
        nodes_p_tensor = [[x] for x in list(nodes_p.keys())]
        graph_data_x = torch.tensor(nodes_p_tensor, dtype=torch.float, device=device)
        graph_edge = torch.tensor(my_edges, dtype=torch.long, device=device)
        y_tensor = torch.tensor(
            [[y_dict[fname.split("_")[0] + "_1"][0]]], dtype=torch.float, device=device
        )

        G = Data(
            x=graph_data_x,
            edge_index=graph_edge.t().contiguous(),
            y=y_tensor,
            dtype=torch.float,
        )
        graph_x[fname] = nodes_p_final
        labels = {}

        # build hot-encoding for each node in G.

        for node in nodes_p_final:
            label_node = [0] if nodes_p_entry[node] == "ATOM" else [1]
            atom_type_node = [0] * len(atom_type_list)  # inizialise vector all 0
            atom_type_node[
                atom_type_list.index(nodes_p[node].name)
            ] = 1  # build when atom type== 1
            element_node = [0] * len(element_list)
            element_node[element_list.index(elements_p[node])] = 1
            label = {
                "attributes": [
                    label_node,
                    atom_type_node,
                    element_node,
                    hydrogen_label[node],
                ]
            }

            labels[node] = label

        global_G[fname] = (G, labels)
        print("{} ohe completed".format(fname))

    store_element_cache(ohe_path + "/.ohe_atom_order", atom_type_list)
    store_element_cache(ohe_path + "/.ohe_element_order", element_list)

    return global_G


def euclidean_distance(p:float, q: float) -> float:
    """
    Compute euclidian distance between pair of coordinates
    """
    return math.sqrt(((p[0] - q[0]) ** 2) + ((p[1] - q[1]) ** 2) + ((p[2] - q[2]) ** 2))


def save_graphs(global_G: dict, dirname: str) -> None:
    my_files = list(os.walk(dirname))[0][2]
    torch.save(global_G, dirname + "/tensorG_{}.pt".format(len(my_files)))


def write_cache(cache_fname: str, cache_list: list) -> None:
    with open(cache_fname, "a") as f:
        for k in cache_list:
            f.writelines(k + "\n")
