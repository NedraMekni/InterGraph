import sys
import os
import openbabel
import pickle
from openbabel import openbabel
from InterGraph.target_req import retrieve_chembl_data
from InterGraph.source_data import write_clean_raw_data
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

    # get access to bioactive molecules data in ChEBML database and get the PDB structures
    retrieve_chembl_data()

    # save target, activity, assay and PDBid data
    write_clean_raw_data("data.csv")
    file_path()

    # save protein-ligand and ligand structures in pdb and mol2 files respectively
    save_structure("data.csv")
    
    # generate protein-ligand interaction graph
    mol_graphs_crystal_6A = {}
    output_file_graphs = os.path.join(
        "../data/IFG_output/", "PLIG_test_run_6A_std.pickle"
    )
    # save protein-ligand interaction graph as pickle file
    with open(output_file_graphs, "wb") as handle:
        pickle.dump(mol_graphs_crystal_6A, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    