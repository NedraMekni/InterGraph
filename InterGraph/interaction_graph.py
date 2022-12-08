import pandas as pd
import os
import pickle
import urllib.request
import pandas as pd
import numpy as np
import networkx as nx
import numpy.typing as npt
import tqdm

from openbabel import openbabel
from rdkit import Chem
from scipy.spatial.distance import cdist
from itertools import product
from typing import Tuple, Any, Dict


pdb_raw_d = "/data/shared/projects/NLRP3/data/"


def file_path(data_raw_path, pdb_raw_path, csv_path, IFG_path):
    """
    This function creates data and pdb_raw subdirectories
    pdb_raw: unsplitted pdb
    """

    pdb_raw_d = pdb_raw_path

    if not os.path.exists(data_raw_path):
        os.makedirs(data_raw_path)
    if not os.path.exists(pdb_raw_path):
        os.makedirs(pdb_raw_path)
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    if not os.path.exists(IFG_path):
        os.makedirs(IFG_path)


def get_pdb_components(pdb_id: str):

    """This function performs an HTTPS request to get PDBs from rcsb.org"""
    try:
        pdb_r = urllib.request.urlretrieve(
            "https://files.rcsb.org/download/{}.pdb".format(pdb_id),
            "{}/{}.pdb".format(pdb_raw_d, pdb_id),
        )
    except:
        pass
    
    return get_pdb_components


def parse_pdb(lig: str, filename: str) -> Tuple[list, list]:
    """This function splits proten-ligand file into ligand and protein for each PDB in /data/pdb_raw"""

    prot = []
    ligs = []
    try:
        with open(filename, "r") as f:
            for l in f:
                if l.split()[0] == "ATOM":
                    prot += [l]
                if l.split()[0] == "HETATM" and len(l.split()) > 3 and l.split()[3] == lig:
                    ligs += [l]
    except:
        pass                    

    return prot, ligs


def write_pdb(prot: str, lig: str, filename: str) -> None:
    """
    This function writes PDB file containing the protein complexed with the ligand of interest.
    Water molecules, salts and metal ions are excluded
    """

    with open(filename, "w") as f:
        for l in prot:
            f.write(l)
        f.write("TER\n")
        for l in lig:
            f.write(l)


def write_ligand_pdb(lig: str, filename: str):
    """This function saves each ligand as a PDB file"""
    with open(filename, "w") as f:
        for l in lig:
            l.split(",")
            f.write(l)


def load_structures() -> dict:
    """
    This function returns protein-ligand interaction graph.
    The protein-ligand contact proximinity threashold it is set to 6Å
    PDB_Atom_Keys.csv stores all possible protein atom types
    reference: 10.1093/bioinformatics/btaa982
    """
    raw_data = "data"
    list_of_pdbcodes = [i for i in os.listdir(raw_data)]

    atom_keys = pd.read_csv("../data/csv/PDB_Atom_Keys.csv", sep=",")

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

        mol_graphs_crystal_6A = {}
        contacts_6A = get_atom_contacts(
            protein_path, c_mol, atom_keys, distance_cutoff=6.0
        )
        graph_c_6 = mol_to_graph(c_mol, contacts_6A, atom_keys)
        mol_graphs_crystal_6A[pdb] = graph_c_6
    print(type(mol_graphs_crystal_6A))
    return mol_graphs_crystal_6A


def get_atom_type(atom: str) -> str:
    """This function identifies the unique protein atom types using rdkit modules"""
    AtomType = [
        atom.GetSymbol(),
        str(atom.GetExplicitValence()),
        str(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() != "H"])),
        str(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() == "H"])),
        str(int(atom.GetIsAromatic())),
        str(int(atom.IsInRing())),
    ]

    return ";".join(AtomType)


def load_sdf_as_df(mol):
    """This function converts ligand mol2 file into pandas DataFrame"""
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


def load_pdb_as_df(pdb: str, atom_keys: str) -> pd.DataFrame:
    """This function converts protein PDB file into a pandas DataFrame with the protein atom position in 3D (X,Y,Z)"""

    prot_atoms = []

    f = open(pdb)
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
        df.merge(atom_keys, left_on="PDB_ATOM", right_on="PDB_ATOM")[
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


def get_atom_contacts(pdb_protein, mol, atom_keys, distance_cutoff=6.0):
    """
    This function returns the list of protein atom types the ligand interacts with for a given distance cutoff
    cutoff = 6 Angstrom
    """
    target = load_pdb_as_df(pdb_protein, atom_keys)
    ligand = load_sdf_as_df(mol)

    # A cubic box around the ligand is created using the proximity threshold specified (here distance_cutoff = 6 Angstrom by default).
    for i in ["X", "Y", "Z"]:
        target = target[target[i] < float(ligand[i].max()) + distance_cutoff]
        target = target[target[i] > float(ligand[i].min()) - distance_cutoff]

    # Calculate the possible pairs
    pairs = list(product(target["ATOM_TYPE"], ligand["ATOM_INDEX"]))
    pairs = [str(x[0]) + "-" + str(x[1]) for x in pairs]
    pairs = pd.DataFrame(pairs, columns=["ATOM_PAIR"])

    distances = cdist(
        target[["X", "Y", "Z"]], ligand[["X", "Y", "Z"]], metric="euclidean"
    )
    distances = distances.reshape(distances.shape[0] * distances.shape[1], 1)
    distances = pd.DataFrame(distances, columns=["DISTANCE"])

    # Select pairs with distance lower than the cutoff
    pairs = pd.concat([pairs, distances], axis=1)
    pairs = pairs[pairs["DISTANCE"] <= distance_cutoff].reset_index(drop=True)

    contact_pair_list = [i.split("-")[0] for i in pairs["ATOM_PAIR"]]
    pairs["PROT_ATOM"] = contact_pair_list
    pairs["LIG_ATOM"] = [int(i.split("-")[1]) for i in pairs["ATOM_PAIR"]]

    return pairs


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
    """
    This function computes ligand atom features (number of heavy atoms, nomber of hydrogen atom neighbors,
    explicit valence of the atom, aromticity, atom in a ring) to generate node graph
    """

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


def atom_features_plig(
    atom_idx: str,
    atom: str,
    contact_df: pd.DataFrame,
    extra_features: str,
    atom_keys: Dict[Any, Any],
) -> npt.NDArray[Any]:

    """This function generates the protein-ligand interaction features"""
    possible_contacts = list(dict.fromkeys(atom_keys["ATOM_TYPE"]))
    feature_list = np.zeros(len(possible_contacts), dtype=int)
    contact_df_slice = contact_df[contact_df["LIG_ATOM"] == atom_idx]

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
    atom_keys,
    extra_features=[
        "num_heavy_atoms",
        "total_num_Hs",
        "explicit_valence",
        "is_aromatic",
        "is_in_ring",
    ],
):
    """
    This function returns molecular graph with its node features and edge indices
    """
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
            feature = atom_features_plig(
                atom.GetIdx(), atom, contact_df, extra_features, atom_keys
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

    return c_size, features, edge_index


def save_structure(fname: str, datadir: str):
    """This function generated multiple subdirestories for each protein-ligand system .

    Every subdirectory is named after its protein-ligand complex and it contains the input files required to generated the molecular graph:
    1) Protein_ligand complex in pdb format
    2) Ligand in pdb format
    3) Ligand pdb structures are converted and saved as mol2 format."""

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

                prot, lig = parse_pdb(pdb_lig, pdb_raw_d + "/" + pdb_prot + ".pdb")

                if not os.path.exists(
                    datadir+"/PDB/data/"
                    + pdb_prot
                    + "_"
                    + pdb_lig
                ):
                    os.makedirs(
                        datadir+"/PDB/data/"
                        + pdb_prot
                        + "_"
                        + pdb_lig
                    )
                    write_pdb(
                        prot,
                        lig,
                        datadir+"/PDB/data/"
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
                        datadir+"/PDB/data/"
                        + pdb_prot
                        + "_"
                        + pdb_lig
                        + "/"
                        + pdb_lig
                        + ".pdb"
                    ):
                        write_ligand_pdb(
                            lig,
                            datadir+"/PDB/data/"
                            + pdb_prot
                            + "_"
                            + pdb_lig
                            + "/"
                            + pdb_lig
                            + ".pdb",
                        )

                        for file in os.listdir(
                            datadir+"/PDB/data/"
                            + pdb_prot
                            + "_"
                            + pdb_lig
                            + "/"
                        ):
                            if file.endswith(f"{pdb_lig}.pdb"):
                                file_path = (
                                    datadir+"/PDB/data/"
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
                                    datadir+"/PDB/data/"
                                    + pdb_prot
                                    + "_"
                                    + pdb_lig
                                    + "/"
                                    + f"{pdb_lig}.mol2",
                                )
                                obConversion.CloseOutFile()

def merge_window(window_csv, final_csv, has_checkpoint):
    if(not has_checkpoint):
        os.system('cp '+window_csv+' '+final_csv)
    else:
        with open(window_csv,'r') as w:
            with open(final_csv,'a') as f:
                for l in w.readlines()[1:]:
                    f.write(l)
