import csv

pdb_id = []
header = ["pdb_id","k_value"]
get_pdb = []
with open("PDBbind_refined_set_all.csv","r") as f:
    for lines in f:
        l =f.readline()
        print(l)
        split = l.strip().split(",")
        print(split)
        
        for el in split:
            print(el)
            if el != '':
                print('ok')
           
            
                pdb_id.append([split[1]+","+ split[3]])
                get_pdb.append(split[1])
get_pdb = list(set(get_pdb))

with open("load_pdbid.csv","w") as l:
    writer = csv.writer(l)
    writer.writerow(header)
    writer.writerows(pdb_id)
with open("get_pdb.csv", "w") as k:
    writer = csv.writer(k)
    writer.writerow(get_pdb)

