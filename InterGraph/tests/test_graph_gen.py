
import tempfile
import os

from InterGraph.pdb_to_multigraph import biograph_from_file
from Bio.PDB import *

parser = PDBParser()

def test_biograph_from_file_inner_loop():
    test_dir = "."
    test_file = os.path.join(test_dir, "test_file_H.pdb")
    with open(test_file, 'w') as f:
        f.write("""
    ATOM      1  N   GLU A   1      14.706   2.157   6.857  1.00  0.00           N  
    HETATM    2  C   GLU A   1      14.706   2.157   6.857  1.00  0.00           C  
        """)

    my_files = [test_file]
    cntPdb = 0
    nodes_p_entry = {}
    nodes_p = {}
    nodes_i = {}
    elements_p = {}

    structure = parser.get_structure(test_file,"test_file_H.pdb")
    
    i = 0
    for atom in structure.get_atoms():
        coord = atom.coord
        nodes_p[atom.serial_number] = atom
        nodes_i[atom.serial_number] = i
        elements_p[atom.serial_number] = atom.element
        # check if atom is hetatm
        tags = atom.get_full_id()
        nodes_p_entry[atom.serial_number] = (
        "HETATM" if tags[3][0] != " " else "ATOM"
        )
        i += 1

        assert nodes_p_entry[1] == "ATOM"
        assert nodes_p_entry[2] == "HETATM"
        assert nodes_i[1] == 0
        assert nodes_i[2] == 1
        assert elements_p[1].name == "N"
        assert elements_p[2].name == "C"

