import math
import os
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import csv
import torch.nn as nn
import MDAnalysis as mda
import matplotlib.pyplot as olt

from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from functools import reduce

"""
GCN extends torch.nn.module adding some properties and defying forward method
"""


class GCN(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        torch.manual_seed(1234)

        self.conv1 = GCNConv(num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, 1)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        out = self.classifier(h)

        return out, h


def train(data_loader):
    calculate_mse = torch.nn.MSELoss()
    loss = 0.0
    for data in data_loader:
        optimizer.zero_grad()  # Clear gradients.
        ref = data.y
        out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
        loss_for_batch = calculate_mse(out.flatten(), ref)  # Compute the loss.
        loss_for_batch.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        loss += loss_for_batch.detach().item()

    return loss, h


"""
Generate graph dict from PDB files
graph dict:
	-key: PDB file name -> str 
	-value: graph -> tuple of dicts
"""


def graph_from_file(dirname: str) -> dict:
    graph_dict = {}
    my_files = list(os.walk(dirname))[0]

    for z in my_files[2]:
        nodes_p_entry = {}
        nodes_p = {}
        if "_H.pdb" not in z:
            continue

        with open(dirname + "/" + z, "r") as f:

            for l in f:
                l = l.split()

                if l[0] == "ATOM" or l[0] == "HETATM":
                    nodes_p[int(l[1])] = (l[2], float(l[6]), float(l[7]), float(l[8]))
                    nodes_p_entry[int(l[1])] = l[0]

        u = mda.Universe(dirname + "/" + z)
        graph_dict[z] = (nodes_p, nodes_p_entry, u.atoms)

    return graph_dict


"""
Get activity dictionary from csv file
y_dict:
	-key:PDBid -> str
	-value: activity -> float
"""


def data_activities_from_file(fname: str) -> dict:
    label = pd.read_csv(fname, usecols=["pdb_code", "activity"])
    label.to_csv(r"test_y.csv", index=False)
    all_rows = []
    y_dict = {}
    with open("test_y.csv", "r") as f:
        lines = f.readlines()[1:]

        for l in lines:
            elements = l.split(",")
            elements[0] = elements[0].strip()
            elements[1] = elements[1].strip()

            for k in elements[0].split(" "):
                if k not in y_dict.keys():
                    y_dict[k] = []

                elements[1] = float(elements[1])
                y_dict[k].append(elements[1])

    return y_dict


"""
Build multigraph edges from graph_dict

global_g:
	-key:PDB file name -> str
	-value: graph object, node attribute -> tuple
"""

def build_graph_dict(graph_dict, y_dict):
    global_G = {}
    graph_x = {}
    # helper structures for one hot encoding labels and for cast dictionary graph into list
    atom_type_list = [
        [graph_dict[k][0][k1][0] for k1 in graph_dict[k][0].keys()]
        for k in graph_dict.keys()
    ]

    atom_type_list = list(set(reduce(lambda x, y: x + y, atom_type_list)))
    element_list = list(set([el[0] for el in atom_type_list]))
    global_node_list_order = {
        k: list(graph_dict[k][0].keys()) for k in graph_dict.keys()
    }

    print("building adjacence list for each graph")
    for fname in graph_dict.keys():
        my_edges = []
        # node_p and node_p_etry are two dictionaries storing nodes

        nodes_p = graph_dict[fname][0]
        nodes_p_entry = graph_dict[fname][1]
        atom_p = graph_dict[fname][2]
        node_list_order = global_node_list_order[fname]  # to build adj list
        # bulding H label for each node
        # for each atom in atom struct extract the n of H
        hydrogen_label = []
        for atom in atom_p:
            neighbour_atoms = list(atom.bonds)
            n_hydro = len(
                [bond[1].element for bond in neighbour_atoms if bond[1].element == "H"]
            )
            n_hydro_hot = [0] * 5
            assert n_hydro < 5
            n_hydro_hot[n_hydro] = 1
            hydrogen_label.append(n_hydro_hot)
            # print('element {}\n bond list {}\n n hydro {}'.format(atom.element,[bond[1].element for bond in neighbour_atoms if bond[1].element =='H'], n_hydro) )

        # Multigraph generation
        # add edges in a edge list based on distance threshold. We add edge if the distance between two nodes, is < threshold

        for treshold in [3.0, 6.0, 9.0]:

            for kv_1, kv_2 in itertools.combinations(
                nodes_p.items(), 2
            ):  # we consider all possible nodes pair (kv_1,kv_2)
                node_k1, node_v1 = kv_1[0], kv_1[1]
                node_k2, node_v2 = kv_2[0], kv_2[1]
                if euclidean_distance(node_v1[1:], node_v2[1:]) < treshold:
                    # since I'm building a multigraph, I hope there is a way to put a property on same edges
                    my_edges.append(
                        (node_list_order.index(node_k1), node_list_order.index(node_k2))
                    )  # ,{'treshold':treshold}))
                    my_edges.append(
                        (node_list_order.index(node_k2), node_list_order.index(node_k1))
                    )  # ,{'treshold':treshold}))

        # build pytorch data graph
        nodes_p_final = list(nodes_p.keys())
        nodes_p_tensor = [[x] for x in list(nodes_p.keys())]
        graph_data_x = torch.tensor(nodes_p_tensor, dtype=torch.float)
        graph_edge = torch.tensor(my_edges, dtype=torch.long)
        G = Data(
            x=graph_data_x,
            edge_index=graph_edge.t().contiguous(),
            y=torch.tensor([y_dict[fname.split("_")[0] + "_1"][0]]),
            dtype=torch.float,
        )
        graph_x[fname] = nodes_p_final
        labels = {}

        # build hot-encoding for each node in G. Not included yet in the graph
        for i, node in enumerate(nodes_p_final):
            label_node = [0] if nodes_p[node] == "ATOM" else [1]
            atom_type_node = [0] * len(atom_type_list)  # inizialise vector all 0
            atom_type_node[
                atom_type_list.index(nodes_p[node][0])
            ] = 1  # build when atom type== 1
            element_node = [0] * len(element_list)
            element_node[element_list.index(nodes_p[node][0][0])] = 1
            label = {
                "attributes": label_node
                + atom_type_node
                + element_node
                + hydrogen_label[i]
            }

            labels[node] = label

        global_G[fname] = (G, labels)
        print("{} ohe completed".format(fname))

    return global_G


"""
Compute euclidian distance between pair of coordinates
"""


def euclidean_distance(p, q: float, float) -> float:
    return math.sqrt(((p[0] - q[0]) ** 2) + ((p[1] - q[1]) ** 2) + ((p[2] - q[2]) ** 2))


if __name__ == "__main__":

    graph_dict = graph_from_file(
        "/home/nmekni/Documents/NLRP3/InterGraph/data/PDB/data"
    )

    csv = "/home/nmekni/Documents/NLRP3/InterGraph/data/csv/data.csv"

    y_dict = data_activities_from_file(csv)

    global_G = build_graph_dict(graph_dict, y_dict)

    pytg_graph_dict = {k: global_G[k][0] for k in global_G.keys()}

    graph_list = []
    graph_y = []

    for k in pytg_graph_dict.keys():
        pytg_graph_dict[k].label = 0
        pytg_graph_dict[k]["label"] = global_G[k][1]
        graph_list.append(pytg_graph_dict[k])
        graph_y.append(pytg_graph_dict[k].y)

    loader = DataLoader(graph_list, batch_size=1)

    model = GCN(1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.

    loss_values = []

    for epoch in range(401):
        loss, h = train(loader)
        loss_values.append(loss)
        if epoch % 100 == 0:
            print(f"Epoch: {epoch:03d}, loss: {loss:.4f}")

    print(loss_values)
    plt.plot(loss_values, label="loss value")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
