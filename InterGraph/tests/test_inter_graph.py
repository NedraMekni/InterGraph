from dis import dis
from fileinput import filename
from itertools import product

import pandas as pd
from rdkit import Chem
from scipy.spatial.distance import cdist
from InterGraph.interaction_graph import get_atom_contacts,load_pdb_as_df, load_sdf_as_df
import os
import pytest


@pytest.fixture
def mol():
    m = Chem.MolFromMol2File('data/PDB/data/1RKP_IBM/IBM.mol2')
    return m
def test_load_sdf_as_df(mol):
    """ This function converts ligand mol2 file into pandas DataFrame """
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
    assert isinstance(df, pd.DataFrame)
    assert df.empty == False
    
    
# Test pdb file is converted into pandas dataframe.
# Check second column contains protein atom symbol dataframe has

def test_load_pdb_as_df():
    prot_atoms = []
    pdb= 'data/PDB/data/1RKP_IBM/1RKP_IBM.pdb'
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
    
    assert len(prot_atoms) > 0
    assert len(prot_atoms[0]) == 5
    
    df_check = df.iloc[0]['PDB_ATOM']
    print(df_check)
    assert prot_atoms[0][1] == df_check
    print(df)
    return df


    

def test_get_atom_contacts() :
    
    pdb_protein =  '../../data/PDB/data/1RKP_IBM/1RKP_IBM.pdb'
    mol = Chem.MolFromMol2File('data/PDB/data/1RKP_IBM/IBM.mol2')
    distance_cutoff = 6.0
   
    atom_keys = pd.read_csv("data/csv/PDB_Atom_Keys.csv", sep=",") 

    target = load_pdb_as_df(pdb_protein, atom_keys)
    ligand = load_sdf_as_df(mol)

    if distance_cutoff  in range(4,7):
        return True
    else:
        print("Distance cut off must be between 4 and 6 Å")

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
   
    number_atom_in_ligand = pairs.index.tolist()
    
    assert len(number_atom_in_ligand) == (len(contact_pair_list))
   

    

    
   
if __name__=='__main__':
     test_load_pdb_as_df()

    
