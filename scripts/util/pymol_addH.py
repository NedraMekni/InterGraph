import os
from select import select
import pymol 
from dotenv import load_dotenv

load_dotenv()
#ROOT_PATH = os.getenv('ROOT_PATH')
#ROOT_PATH = "/data/shared/projects/NLRP3/GNN_IG/PDB_cleaned/"
ROOT_PATH = os.getenv('ROOT_PATH')

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
				print("THIS IS X{}".format(x))
				list_pdb.append(x)
	
			

	for el in list_pdb:
		
		complex_input = (os.path.join(el[:-4]+'/'+el[:-4]  +'.pdb'))
		#complex_w = os.path.join('/data/shared/projects/NLRP3/graphs/'+el[:-4]  +'_H'+'.pdb')
		print(complex_input)
		
		complex_w = (path_dir+el.split("/")[-1].split(".")[0] +'_H'+'.pdb')
		
		os.system("python3 {} {} {}".format("pymol_worker.py",path_dir,el))				
	
		
	
if __name__=='__main__':
	
	#direct = ROOT_PATH+'/PDB/data/'
	direct = "/data/shared/projects/NLRP3/GNN_IG/PDB_cleaned/"
	add_hydrogens_atoms(direct)
