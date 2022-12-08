import sys
import os
import pickle
from tqdm import tqdm
from openbabel import openbabel
from dotenv import load_dotenv

from InterGraph.target_req import (
    set_ntarget,
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
    merge_window,
    mol_to_graph,
    file_path,
)

load_dotenv()
ROOT_PATH = os.getenv('ROOT_PATH')

EXIST_CHECKPOINT = os.path.isfile(ROOT_PATH+'/.checkpoint')
print(EXIST_CHECKPOINT)

if __name__ == "__main__":

    # create output directories
    file_path(ROOT_PATH, ROOT_PATH+"/pdb_raw", ROOT_PATH+"/csv", ROOT_PATH+"/IFG_output")

    set_ntarget(8)
    # get access to bioactive molecules data in ChEBML database and get the PDB structures
    # results are written in a csv file
    compounds = retrieve_chembl_data(ROOT_PATH+"/csv/data_from_chembl.csv",ROOT_PATH+"/.checkpoint")
    retrieve_pdb_data(compounds, ROOT_PATH+"/csv/raw_data.csv")

    # save target, activity, assay and PDBid data
    clear_raw_data(ROOT_PATH+"/csv/raw_data.csv", ROOT_PATH+"/csv/cleaned_raw_data.csv")
    write_clean_raw_data(ROOT_PATH+"/csv/cleaned_raw_data.csv", ROOT_PATH+"/csv/data_window.csv")

    # save protein-ligand and ligand structures in pdb and mol2 files respectively
    save_structure(ROOT_PATH+"/csv/data_window.csv",ROOT_PATH)
    merge_window(ROOT_PATH+"/csv/data_window.csv",ROOT_PATH+"/csv/data.csv",EXIST_CHECKPOINT)
    