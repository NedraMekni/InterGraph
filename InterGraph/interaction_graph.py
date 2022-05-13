import os.path
import pandas as pd
import os
import pickle
import urllib
import pandas as pd
import numpy as np
import networkx as nx
import openbabel

from tqdm import tqdm
from rdkit import Chem
from scipy.spatial.distance import cdist
from itertools import product
from openbabel import openbabel

pdb_raw_d = "../data/" + "pdb_raw"


def file_path():
    global pdb_raw_d
    if not os.path.exists("../data/" + "data"):
        os.makedirs("../data/" + "data")
    if not os.path.exists("../data/" + "pdb_raw"):
        os.makedirs("../data/" + "pdb_raw")


def get_pdb_components(pdb_id):
    global pdb_raw_d

    pdb_r = urllib.request.urlretrieve(
        "https://files.rcsb.org/download/{}.pdb".format(pdb_id),
        "{}/{}.pdb".format(pdb_raw_d, pdb_id),
    )


def parsePDB(lig, filename):
    prot = []
    ligs = []

    with open(filename, "r") as f:
        for l in f:
            if l.split()[0] == "ATOM":
                prot += [l]
            if len(l.split()) > 3 and l.split()[3] == lig:
                ligs += [l]

    return prot, ligs


def write_pdb(prot, lig, filename):
    with open(filename, "w") as f:
        for l in prot:
            f.write(l)
            f.write("TER\n")
        for l in lig:
            f.write(l)


def write_ligand_pdb(lig, filename):

    with open(filename, "w") as f:
        for l in lig:
            l.split(",")
            f.write(l)
            return f


def load_structures():
    raw_data = "data"
    list_of_pdbcodes = [i for i in os.listdir(raw_data)]
    Atom_Keys = pd.read_csv("../data/csv/PDB_Atom_Keys.csv", sep=",")

    for pdb in tqdm(list_of_pdbcodes):
        lig_path = os.path.join(raw_data, pdb, f"{pdb[-3:]}.mol2")
        protein_path = os.path.join(raw_data, pdb, f"{pdb}.pdb")

        # load the ligand and handle invalid mol2 file
        try:
            c_mol = Chem.AddHs(
                Chem.MolFromMol2File(
                    lig_path, sanitize=False, removeHs=True, cleanupSubstructures=True
                ),
                addCoords=True,
            )
            print("Mol from Mol2 conversion succeded  ", lig_path)
        except:
            print("Mol from Mol2 conversion failed ", lig_path)
            continue

        # 6 Angstrom
        mol_graphs_crystal_6A = {}
        contacts_6A = GetAtomContacts(
            protein_path, c_mol, Atom_Keys, distance_cutoff=6.0
        )
        graph_c_6 = mol_to_graph(c_mol, contacts_6A, Atom_Keys)
        mol_graphs_crystal_6A[pdb] = graph_c_6
        return mol_graphs_crystal_6A


def GetAtomType(atom):

    AtomType = [
        atom.GetSymbol(),
        str(atom.GetExplicitValence()),
        str(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() != "H"])),
        str(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() == "H"])),
        str(int(atom.GetIsAromatic())),
        str(int(atom.IsInRing())),
    ]

    return ";".join(AtomType)


def LoadSDFasDF(mol):

    m = mol

    atoms = []

    for atom in m.GetAtoms():
        if atom.GetSymbol() != "H":  # Include only non-hydrogen atoms
            entry = [int(atom.GetIdx())]
            entry.append(str(atom.GetSymbol()))
            pos = m.GetConformer().GetAtomPosition(atom.GetIdx())
            entry.append(float("{0:.4f}".format(pos.x)))
            entry.append(float("{0:.4f}".format(pos.y)))
            entry.append(float("{0:.4f}".format(pos.z)))
            atoms.append(entry)

    df = pd.DataFrame(atoms)
    df.columns = ["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"]

    return df


def LoadPDBasDF(PDB, Atom_Keys):
    # This function converts a protein PDB file into a pandas DataFrame with the protein atom position in 3D (X,Y,Z)

    prot_atoms = []

    f = open(PDB)
    for i in f:
        if i[:4] == "ATOM":
            # Include only non-hydrogen atoms
            if (
                len(i[12:16].replace(" ", "")) < 4
                and i[12:16].replace(" ", "")[0] != "H"
            ) or (
                len(i[12:16].replace(" ", "")) == 4
                and i[12:16].replace(" ", "")[1] != "H"
                and i[12:16].replace(" ", "")[0] != "H"
            ):
                prot_atoms.append(
                    [
                        int(i[6:11]),
                        i[17:20] + "-" + i[12:16].replace(" ", ""),
                        float(i[30:38]),
                        float(i[38:46]),
                        float(i[46:54]),
                    ]
                )

    f.close()

    df = pd.DataFrame(prot_atoms, columns=["ATOM_INDEX", "PDB_ATOM", "X", "Y", "Z"])
    df = (
        df.merge(Atom_Keys, left_on="PDB_ATOM", right_on="PDB_ATOM")[
            ["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"]
        ]
        .sort_values(by="ATOM_INDEX")
        .reset_index(drop=True)
    )
    if list(df["ATOM_TYPE"].isna()).count(True) > 0:
        print(
            "WARNING: Protein contains unsupported atom types. Only supported atom-type pairs are counted."
        )
    return df


# This function returns the list of protein atom types the ligand interacts with for a given distance cutoff
# cutoff = 6 Angstrom is standard
def GetAtomContacts(PDB_protein, mol, Atom_Keys, distance_cutoff=6.0):

    Target = LoadPDBasDF(PDB_protein, Atom_Keys)
    Ligand = LoadSDFasDF(mol)

    # A cubic box around the ligand is created using the proximity threshold specified (here distance_cutoff = 6 Angstrom by default).
    for i in ["X", "Y", "Z"]:
        Target = Target[Target[i] < float(Ligand[i].max()) + distance_cutoff]
        Target = Target[Target[i] > float(Ligand[i].min()) - distance_cutoff]

    # Calculate the possible pairs
    Pairs = list(product(Target["ATOM_TYPE"], Ligand["ATOM_INDEX"]))
    Pairs = [str(x[0]) + "-" + str(x[1]) for x in Pairs]
    Pairs = pd.DataFrame(Pairs, columns=["ATOM_PAIR"])

    Distances = cdist(
        Target[["X", "Y", "Z"]], Ligand[["X", "Y", "Z"]], metric="euclidean"
    )
    Distances = Distances.reshape(Distances.shape[0] * Distances.shape[1], 1)
    Distances = pd.DataFrame(Distances, columns=["DISTANCE"])

    # Select pairs with distance lower than the cutoff
    Pairs = pd.concat([Pairs, Distances], axis=1)
    Pairs = Pairs[Pairs["DISTANCE"] <= distance_cutoff].reset_index(drop=True)

    contact_pair_list = [i.split("-")[0] for i in Pairs["ATOM_PAIR"]]
    Pairs["PROT_ATOM"] = contact_pair_list
    Pairs["LIG_ATOM"] = [int(i.split("-")[1]) for i in Pairs["ATOM_PAIR"]]

    return Pairs


def atom_features(
    atom,
    features=[
        "num_heavy_atoms",
        "total_num_Hs",
        "explicit_valence",
        "is_aromatic",
        "is_in_ring",
    ],
):

    feature_list = []
    if "num_heavy_atoms" in features:
        feature_list.append(
            len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() != "H"])
        )
    if "total_num_Hs" in features:
        feature_list.append(
            len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() == "H"])
        )
    if "explicit_valence" in features:
        feature_list.append(atom.GetExplicitValence())
    if "is_aromatic" in features:

        if atom.GetIsAromatic():
            feature_list.append(1)
        else:
            feature_list.append(0)
    if "is_in_ring" in features:
        if atom.IsInRing():
            feature_list.append(1)
        else:
            feature_list.append(0)
    return np.array(feature_list)


# Generates the protein-ligand interaction features for the PLIG creation
def atom_features_PLIG(atom_idx, atom, contact_df, extra_features, Atom_Keys):

    possible_contacts = list(dict.fromkeys(Atom_Keys["ATOM_TYPE"]))
    feature_list = np.zeros(len(possible_contacts), dtype=int)
    contact_df_slice = contact_df[contact_df["LIG_ATOM"] == atom_idx]

    # count the number of contacts between ligand and protein atoms
    for i, contact in enumerate(possible_contacts):
        for k in contact_df_slice["PROT_ATOM"]:
            if k == contact:
                feature_list[i] += 1

    extra_feature_array = atom_features(atom, extra_features)
    output = np.append(extra_feature_array, feature_list)

    return output


def mol_to_graph(
    mol,
    contact_df,
    Atom_Keys,
    extra_features=[
        "num_heavy_atoms",
        "total_num_Hs",
        "explicit_valence",
        "is_aromatic",
        "is_in_ring",
    ],
):

    c_size = len([x.GetSymbol() for x in mol.GetAtoms() if x.GetSymbol() != "H"])
    features = []
    heavy_atom_index = []
    idx_to_idx = {}
    counter = 0

    # Generate nodes
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H":  # Include only non-hydrogen atoms
            idx_to_idx[atom.GetIdx()] = counter
            counter += 1
            heavy_atom_index.append(atom.GetIdx())
            feature = atom_features_PLIG(
                atom.GetIdx(), atom, contact_df, extra_features, Atom_Keys
            )
            features.append(feature)

    # Generate edges
    edges = []
    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        if idx1 in heavy_atom_index and idx2 in heavy_atom_index:
            edges.append(
                [idx_to_idx[bond.GetBeginAtomIdx()], idx_to_idx[bond.GetEndAtomIdx()]]
            )
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    # return molecular graph with its node features and edge indices
    return c_size, features, edge_index


# Save structure in PDB and mol2 format
def save_structure(fname):
    global pdb_raw_d
    pdb_prot_downloaded = []
    with open(fname, "r") as f:
        lines = f.readlines()[1:]
        for l in lines:
            # print(l)
            pdb_prot_list, pdb_lig = (
                l.strip().split(",")[2].strip(),
                l.strip().split(",")[3].strip(),
            )
            pdb_prot_list = [x[:4] for x in pdb_prot_list.split()]

            # print(pdb_lig_list)
            # print(pdb_prot_list)

            for pdb_prot in pdb_prot_list:
                if pdb_prot not in pdb_prot_downloaded:
                    x = get_pdb_components(pdb_prot)
                    pdb_prot_downloaded += [pdb_prot]

                prot, lig = parsePDB(pdb_lig, pdb_raw_d + "/" + pdb_prot + ".pdb")

                if not os.path.exists("../data/PDB/data/" + pdb_prot + "_" + pdb_lig):
                    os.makedirs("../data/PDB/data/" + pdb_prot + "_" + pdb_lig)
                    write_pdb(
                        prot,
                        lig,
                        "../data/PDB/data/"
                        + pdb_prot
                        + "_"
                        + pdb_lig
                        + "/"
                        + pdb_prot
                        + "_"
                        + pdb_lig
                        + ".pdb",
                    )
                    if not os.path.exists(
                        "../data/PDB/data/"
                        + pdb_prot
                        + "_"
                        + pdb_lig
                        + "/"
                        + pdb_lig
                        + ".pdb"
                    ):
                        write_ligand_pdb(
                            lig,
                            "../data/PDB/data/"
                            + pdb_prot
                            + "_"
                            + pdb_lig
                            + "/"
                            + pdb_lig
                            + ".pdb",
                        )

                        for file in os.listdir(
                            "../data/PDB/data/" + pdb_prot + "_" + pdb_lig + "/"
                        ):
                            if file.endswith(f"{pdb_lig}.pdb"):
                                file_path = (
                                    "../data/PDB/data/"
                                    + pdb_prot
                                    + "_"
                                    + pdb_lig
                                    + "/"
                                    + pdb_lig
                                    + ".pdb"
                                )
                                obConversion = openbabel.OBConversion()
                                obConversion.SetInAndOutFormats("pdb", "mol2")
                                mol = openbabel.OBMol()
                                obConversion.ReadFile(mol, file_path)
                                mol.AddHydrogens()
                                obConversion.WriteFile(
                                    mol,
                                    "../data/PDB/data/"
                                    + pdb_prot
                                    + "_"
                                    + pdb_lig
                                    + "/"
                                    + f"{pdb_lig}.mol2",
                                )  # else:
                    # print("The write is skipped")
