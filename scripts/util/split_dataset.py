import os
import math
import shutil

y_filename = "./y_preprocessed_ki_filtered.csv"
input_dataset = "./out_conect_mono_def"
output_dataset = "./out_conect_mono_def_splitted"
y_out_splitted = "./Y_splitted"

lines = []
with open(y_filename, "r") as f:
    lines = f.readlines()
lines = [line.strip() for line in lines[1:]]
title = " pdb_code, activity\n"
token_size = 200
y_tokens = [lines[i*token_size:(i+1) * token_size] for i in range(math.ceil(len(lines)/token_size))]

pdb_files = os.listdir(input_dataset)

for i,y_token in enumerate(y_tokens):
    token_datase_dir = "{}/{}".format(output_dataset,f"{i}")
    if not os.path.exists(token_datase_dir):
        #print("creating folder","{}/{}".format(output_dataset,f"{i}"))
        os.makedirs(token_datase_dir)
    
    for mol in y_token:
        mol = mol.split(",")[0]
        pdb_file = [fname for fname in pdb_files if fname.split("_")[0]==mol]
        
        if (len(pdb_file)>0):
            pdb_file = pdb_file[0]
        else:
            continue
        input_pdb_file =input_dataset + "/" + pdb_file
        print("{}/{}".format(token_datase_dir,pdb_file))
        shutil.copy(input_pdb_file,token_datase_dir)
    y_filename_token = y_out_splitted + "/" + str(i) + ".csv"
    with open (y_filename_token, "w") as f:
        f.write(title)
        for mol in y_token:
            f.write(" {}\n".format(mol))
    
