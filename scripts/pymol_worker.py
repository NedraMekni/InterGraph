import pymol
import sys


def pymol_istance(path_dir,complex_input):
    complex_w = (path_dir+complex_input.split("/")[-1].split(".")[0] +'_H'+'.pdb')	

    pymol.cmd.load(complex_input,'myprotein')	
    pymol.cmd.select('chain A')
    pymol.cmd.get_chains('chain A')
    pymol.cmd.h_add()	

    pymol.cmd.save(filename = complex_w, selection = 'chain A')
    pymol.cmd.delete(complex_w)

    

if __name__ == "__main__":
    path_dir = sys.argv[1]
    complex_input = sys.argv[2]
    pymol_istance(path_dir,complex_input)