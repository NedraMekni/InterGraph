
import os

in_dir = "./out_conect_mono_def"
in_y_filename = "./y_preprocessed_ki.csv"
out_y_filename = "./y_preprocessed_ki_filtered.csv"

PDB_files = os.listdir(in_dir)
PDB_files = [name.split("_")[0] for name in PDB_files]

lines = []
with open(in_y_filename, "r") as f:
    lines = f.readlines()

lines = lines[1:]
print(len(lines))
lines = [line.strip() for line in lines if len(line.split(",")) == 3]
lines = [line for line in lines if line.split(",")[0] in PDB_files]

y_dict = {line.split(",")[0]: float(line.split(",")[1]) for line in lines}

title = " pdb_code, activity\n"

with open(out_y_filename, "w") as f:
    f.write(title)
    for k in y_dict.keys():
        f.write(" {},{}\n".format(k,y_dict[k]))

print(PDB_files)
print([x for x in PDB_files if x not in y_dict.keys()])
