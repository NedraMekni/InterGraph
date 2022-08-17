import requests
import json
import string

from tqdm import tqdm
from chembl_webresource_client.new_client import new_client

target = new_client.target
activity = new_client.activity

# Limit target variable specifies the number of targets pulled back for each url call
LIMIT_TARGET = 20


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


def retrieve_chembl_data(fname):
    """This function allows to access ChEMBL data through ChEMBL websource client.
    target chebmlID, uniprotID, protein name,ligand smiles, activity,assay type and molecule chembl id are stored in a csv file"""
    global target, activity, LIMIT_TARGET

    r, results = [], []
    f = open(fname, "w")
    f.write(
        "target_chembl_id, target_uniprot_id, pref_name, canonical_smiles, IC50 uM, assay_chembl_id, molecule_chembl_id, IC50_units\n"
    )

    count_target = 1

    # t_chembl = target_api[0]['target_chembl_id']
    for prot in target.all():
        prot_chembl_id = prot["target_chembl_id"]
        if len(prot["cross_references"]) == 0:
            continue
        prot_uniprot_id = prot["cross_references"][0]["xref_id"]
        pref_name = prot["pref_name"]
        # check if prot uniprot id is valid

        if not check_uniprot(prot_uniprot_id):
            prot_uniprot_id = None

        print(prot_chembl_id, prot_uniprot_id, pref_name)
        prot_activities = activity.filter(target_chembl_id=prot_chembl_id).filter(
            standard_type="IC50", 
            standard_units="nM"
        )

        try:
            get_target_activity = []
            for prot_activity in prot_activities:
                get_target_activity.append(prot_activity)
        except:
            print('target skipped, activity not found')
        """
		prot_activities is a list 
		"""
        for t in get_target_activity:

            # print('\t {} {} {}'.format(t['molecule_chembl_id'],t['canonical_smiles'],t['standard_value']))
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
        count_target += 1
        if count_target == LIMIT_TARGET:

            break

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
