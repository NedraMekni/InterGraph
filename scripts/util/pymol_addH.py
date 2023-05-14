import os
import pymol 


def add_hydrogens_atoms(path_dir):
	list_pdb = []
	complex_pdb=[]
	for path,subdirs,files in os.walk(path_dir):
		#print(files)
		
		for name in files:
			#print(name)
			#print(len(name))
			if len(name) == 8 and name.endswith('.pdb'):
				complex_pdb.append(name)

				x=(os.path.join(path,name))
				print("THIS IS X={}".format(x))
				list_pdb.append(x)
	
			

	for el in list_pdb:
		
		os.system("python3 {} {} {}".format("pymol_worker.py",'/data/shared/projects/NLRP3/GNN_IG/complete_DATASET/out_all_chain_with_h/',el))				
	
		
	
if __name__=='__main__':
	
	direct = "/data/shared/projects/NLRP3/GNN_IG/complete_DATASET/out_uncompress"
	add_hydrogens_atoms(direct)
