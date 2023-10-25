
import os
import MDAnalysis as mda
from Bio import PDB
from Bio.PDB import PDBParser, PDBIO, Select
from pdbfixer import PDBFixer
from openmm.app import PDBFile, Topology
from MDAnalysis.coordinates import PDB
from rdkit import Chem
from rdkit.Chem import AllChem


def write_conect(input_filename, output_filename):
    u = mda.Universe(input_filename)
    try:
        u.atoms.guess_bonds()
    except:
        return 1
    with PDB.PDBWriter(output_filename, bonds = "all") as writer:
        writer.write(u)


def write_conect_call():

    in_dir = "./out_with_lig_def"
    out_dir = "./out_conect_def"

    PDB_FILES = os.listdir(in_dir)
    parser = PDBParser()

    io = PDBIO()

    count = 0
    for filename in PDB_FILES:
        input_filename = in_dir + "/" + filename
        output_filename = out_dir + "/" + filename
        print(input_filename, "=> ", output_filename )
        write_conect(input_filename,output_filename)
        count +=1
        print(count)

        
        
if __name__ == '__main__':
    write_conect_call()