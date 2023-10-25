import requests
import json
import string
import os

from tqdm import tqdm
from chembl_webresource_client.new_client import new_client

target = new_client.target
activity = new_client.activity
assay = new_client.assay


# Limit target variable specifies the number of targets pulled back for each url call
NEW_TARGET = None

def set_ntarget(new_t:int):
    global NEW_TARGET
    NEW_TARGET = new_t


def get_checkpoint(fcheckpoint):
    if(os.path.isfile(fcheckpoint)):
        with open(fcheckpoint,'r') as f:
            l = f.readline()
            try:
                return int(l)
            except:
                return 0
    return 0

def update_checkpoint(fcheckpoint,new_checkpoint):
    with open(fcheckpoint,'w+') as f:
        f.write(str(new_checkpoint))
            


def chembl_comp_to_pdb_new(chembl_compound_id: str):
    """This function performs UniChem API query using the ligand chemblID to get PDB ligand ID."""

    try:
        url = "https://www.ebi.ac.uk/unichem/api/v1/compounds"
        payload = {"type": "sourceID", "compound": chembl_compound_id}
        headers = {"Content-Type": "application/json"}
        response = requests.request("POST", url, json=payload, headers=headers)
        resp_json = response.json()
        # pdb_l_id = resp_json['compounds'][0]['sources'][2]['compoundId']
        # if len(pdb_l_id) == 3:
        for d in resp_json["compounds"][0]["sources"]:
            pdb_l_id = d["compoundId"]
            if len(pdb_l_id) == 3 and type(pdb_l_id) == str:
                return pdb_l_id
        return None
    except:
        return None


def check_uniprot(uniprot_id: str):
    """This function checks if it's a valid uniprot id"""
    return (
        len(uniprot_id) == 6
        and uniprot_id[0] in string.ascii_uppercase
        and len(
            set(uniprot_id).intersection(
                set(string.ascii_lowercase + string.punctuation)
            )
        )
        == 0
    )


"""
Create csv data_from_chembl file
"""


def retrieve_chembl_data(fname,checkpoint):

    """This function allows to access ChEMBL data through ChEMBL websource client.
    target chebmlID, uniprotID, protein name,ligand smiles, activity,assay type and molecule chembl id are stored in a csv file"""
    global target, activity, NEW_TARGET

    # Get checkpoint 
    target_start = get_checkpoint(checkpoint)

    r, results = [], []
    f = open(fname, "w")

    f.write(
        "target_chembl_id, target_uniprot_id, pref_name, canonical_smiles, IC50 uM, assay_chembl_id, molecule_chembl_id, IC50_units\n"
    )

    count_target,global_counter = 1,0

    # t_chembl = target_api[0]['target_chembl_id']
    for prot in target.all():
        if global_counter<target_start:
            global_counter+=1
            continue
        global_counter+=1
        prot_chembl_id = prot["target_chembl_id"]
        if len(prot["cross_references"]) == 0:
            print('Missing cross-reference target window index {}'.format(global_counter))
            continue
        prot_uniprot_id = prot["cross_references"][0]["xref_id"]
        pref_name = prot["pref_name"]
        # check if prot uniprot id is valid

        if not check_uniprot(prot_uniprot_id):
            prot_uniprot_id = None

        print(prot_chembl_id, prot_uniprot_id, pref_name)
        prot_activities = activity.filter(target_chembl_id=prot_chembl_id).filter(
            standard_type="Ki", 
            standard_units="nM",
            assay_type  = 'B'
        )

        #assay_filter = assay.filter(target_chembl_id=prot_chembl_id).filter(confidence_score=9)
        #print(len(assay_filter))
        try:
            get_target_activity = []
            for prot_activity in prot_activities:
                if "assay_chembl_id" in prot_activity.keys():
                    assay_filter = assay.filter(assay_chembl_id=prot_activity["assay_chembl_id"])
                    if len(assay_filter)==1 and "confidence_score" in assay_filter[0].keys():
                        assay_filter = assay_filter[0]
    
                        try:
                            confidence_score = float(assay_filter["confidence_score"])
                            if confidence_score >= 4.0:
                                print(prot_chembl_id,confidence_score)
                                get_target_activity.append(prot_activity)
                        except ValueError:
                            print("activity error cast ")
                            continue
                    else:
                        print("len(assay_filter) = {} or Confidence score not found".format(len(assay_filter)))
                else:
                    print("key error")
                    print(prot_activity.keys())

        except:
            print('target skipped, activity not found')
            continue
        """
		prot_activities is a list 
		"""
        has_t = False 
        for t in get_target_activity:
            has_t = True
            print('\t {} {} {}'.format(t['molecule_chembl_id'],t['canonical_smiles'],t['standard_value']))
            results += [
                [
                    prot_chembl_id,
                    prot_uniprot_id,
                    pref_name,
                    t["canonical_smiles"],
                    t["standard_value"],
                    t["assay_chembl_id"],
                    t["molecule_chembl_id"],
                ]
            ]
            f.write(
                "{}, {}, {}, {}, {}, {}, {}\n".format(
                    prot_chembl_id,
                    prot_uniprot_id,
                    pref_name,
                    t["canonical_smiles"],
                    t["standard_value"],
                    t["assay_chembl_id"],
                    t["molecule_chembl_id"]
                    
                )
            )
        
        if has_t: count_target += 1
        if count_target == NEW_TARGET+1:
            break


        
    
    new_checkpoint = global_counter+target_start
    update_checkpoint(checkpoint,new_checkpoint)

    print("FILE CREATED")
    f.close()
    return results


def retrieve_pdb_data(compounds: list, filename):
    """Given the ligand ChEBML id, this function writes a csv file adding the ligand PDBid to the information stored in data_from_chembl.csv"""
    f = open(filename, "w")
    f.write(
        "target_chembl_id, target_uniprot_id, pref_name, canonical_smiles, IC50 uM, molecule_chembl_id, assay_chembl_id, pdb_comp_id\n"
    )
    print("retrieving ", len(compounds))

    ligand_dict = {}

    for c in tqdm(compounds):
        if c[-1] not in ligand_dict.keys():
            ligand_dict[c[-1]] = chembl_comp_to_pdb_new(c[-1])
        # print(c)
        f.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(*c, ligand_dict[c[-1]]))
    f.close()
