
import sys
from prody import *
from rdkit import Chem
from rdkit.Chem import AllChem
from io import StringIO
import pypdb
import os
from rdkit import Chem

import string
from Bio.PDB import *
from pathlib import Path
from Bio import PDB
from Bio.PDB import PDBParser, PDBIO, Select
from pdbfixer import PDBFixer
from openmm.app import PDBFile, Topology

class ChainNoWaterSelect(Select):
    def accept_chain(self, chain):
        print(chain.get_id())
        return True
        '''
        if chain.get_id() == "A":
            return True
        else:
            return False
        '''
    def accept_residue(self, residue):
        if residue.get_resname().strip() in [
            "HOH",
            "WAT",
            "TIP",
            "DOD",
            "YB",
            "ACT",
            "K",
            "NA",
            "CL",
            "CA",
            "MG",
            "ZN",
            "CU",
            "FE",
            "CO",
            "NI",
            "MN",
            "SO4",
            "PO4",
            "ACY",
            "FORM",
            "TAR",
            "MES",
            "TRIS",
            "HEPES",
            "BES",
            "CHES",
            "ACET",
            "CIT",
            "EDTA",
            "ETH",
            "NAG",
            "EG",
            "PG",
            "POH",
            "THF",
            "MPD",
            "GOL",
        ]:
            return False
        else:
            return True


class ChainSelect(Select):
    def __init__(self, chain):
        self.chain = chain

    def accept_chain(self, chain):
        if chain.get_id() == self.chain:
            return 1
        else:          
            return 0

def clean_pdb(input_file, output_file):
    # Parse the input PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", input_file)

    # Select only chain A, water, salts, and metals
    chain_select = ChainNoWaterSelect()
    # Write the clean PDB file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file, chain_select)

def get_pdb_components(pdb_id):
    """
    Split a protein-ligand pdb into protein and ligand components
    :param pdb_id:
    :return:
    """
    pdb = parsePDB(pdb_id)
    protein = pdb.select('protein')
    ligand = pdb.select('not protein and not water')
    return ligand



def write_conect_records(mol, filename):
    with open(filename, 'w') as f:
        for atom in mol.GetAtoms():
            serial = atom.GetIdx() + 1
            bonded_atoms = atom.GetNeighbors()
            conect_records = ['CONECT', str(serial)]
            for bonded_atom in bonded_atoms:
                conect_records.append(str(bonded_atom.GetIdx() + 1))
            f.write(' '.join(conect_records) + '\n')

#chain_select = ChainNoWaterSelect()


in_dir = './out_uncompress/'
out_dir = './out_with_lig_pipe1_prova/' # PATH_DIRECTORY
cache_dir = './cache_lig_def_pipe1_prova/'


blacklist = []
for fname in os.listdir(in_dir):
    print(fname)
    input_pdb=in_dir+fname
    clean_pdb(input_pdb,cache_dir+fname)
    #ligand=get_pdb_components(str(ligand))
    ligand=get_pdb_components(cache_dir+fname)
    #res_name_list = list(set(ligand.getResnames()))
    #try:
    if 1==1:
        #x=ligand.getResnames()
        #print(set(x),type(x))
        #res_name_list=list(set(ligand.getResnames()))
        #print(res_name_list)
        #res_name=res_name_list[0]
        chains = [x for x in string.ascii_uppercase]

        io = PDBIO()
        parser = PDBParser()
        structure = parser.get_structure("structure", cache_dir+fname)
        #writer.save(PATH_DIRECTORY+"4L2L_"+res_name+"_H.pdb")#, select=ChainNoWaterSelect())
        for chain in chains:
            try:
                io.set_structure(structure)
                print(out_dir+fname.split('.')[0]+'_'+str(chain)+".pdb")
                io.save(out_dir+fname.split('.')[0]+'_'+str(chain)+".pdb",ChainSelect(chain))
                ligand=get_pdb_components(out_dir+fname.split('.')[0]+'_'+str(chain)+".pdb")
                ligand = ligand.getResnames()
                if(len(set(ligand))!=1):
                    blacklist+=[fname]
                        
                #ligand = list(set(ligand))[0]
                #os.system('mv '+out_dir+fname.split('.')[0]+'_'+str(chain)+".pdb"+' '+out_dir+fname.split('.')[0]+'_'+ligand+".pdb")
            except:
                if os.path.isfile(out_dir+fname.split('.')[0]+'_'+str(chain)+".pdb"):
                    os.system('rm '+out_dir+fname.split('.')[0]+'_'+str(chain)+".pdb")

                continue
    
with open('blacklist.txt','w') as f:
    f.write(str(blacklist))
    '''
    except:
        print('Error in '+fname)
        if os.path.isfile(cache_dir+fname):
            os.system('rm '+cache_dir+fname)
        if os.path.isfile(out_dir+fname.split('_')[0]+'_'+res_name+"_H.pdb"):
            os.system('rm '+out_dir+fname.split('_')[0]+'_'+res_name+"_H.pdb")
        continue
    '''

