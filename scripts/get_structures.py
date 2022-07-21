import sys
import os
import pickle
from tqdm import tqdm
from openbabel import openbabel

from InterGraph.target_req import (
    retrieve_chembl_data,
    chembl_comp_to_pdb_new,
    retrieve_pdb_data,
)
from InterGraph.source_data import write_clean_raw_data, clear_raw_data
from chembl_webresource_client.new_client import new_client
from InterGraph.interaction_graph import (
    write_ligand_pdb,
    get_pdb_components,
    parse_pdb,
    write_pdb,
    save_structure,
    mol_to_graph,
    file_path,
)


if __name__ == "__main__":

    # create output directories
    file_path("../data", "../data/pdb_raw", "../data/csv", "../data/IFG_output")

    # get access to bioactive molecules data in ChEBML database and get the PDB structures
    # results are written in a csv file
    compounds = retrieve_chembl_data("../data/csv/data_from_chembl.csv")
    retrieve_pdb_data(compounds, "../data/csv/raw_data.csv")

    # save target, activity, assay and PDBid data
    clear_raw_data("../data/csv/raw_data.csv", "../data/csv/cleaned_raw_data.csv")
    write_clean_raw_data("../data/csv/cleaned_raw_data.csv", "../data/csv/data.csv")

    # save protein-ligand and ligand structures in pdb and mol2 files respectively
    save_structure("../data/csv/data.csv")

    # generate protein-ligand interaction graph
    mol_graphs_crystal_6A = {}
    # save protein-ligand interaction graph as pickle file
    output_file_graphs = "../data/IFG_output/PLIG_test_run_6A_std.pickle"

    exit()
    with open(output_file_graphs, "wb") as handle:
        pickle.dump(mol_graphs_crystal_6A, handle, protocol=pickle.HIGHEST_PROTOCOL)
