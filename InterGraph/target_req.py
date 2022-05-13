import requests
import json
import string

from chembl_webresource_client.new_client import new_client

target = new_client.target
activity = new_client.activity

LIMIT_TARGET = 1


def chembl_comp_to_pdb(chembl_compound_id: str) -> str:

    chembl_q = (
        f"https://www.ebi.ac.uk/unichem/rest/src_compound_id/{chembl_compound_id}/1/3"
    )
    try:
        chembl_res = requests.get(chembl_q)
        n = json.loads(chembl_res.text)

    except:
        return None
    # print(n)
    if (type(n) is list) and (len(n) == 0):
        return None
    elif type(n) is dict and "error" in n.keys():
        return None
    else:
        return n[0]["src_compound_id"]
    return n_target


def check_uniprot(uniprot_id):
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


def retrieve_chembl_data():
    global target, activity, LIMIT_TARGET
    
    r, results = [], []
    f = open("data_from_chembl.csv", "w")
    f.write(
        "target_chembl_id, target_uniprot_id, pref_name, canonical_smiles, IC50 uM, assay_chembl_id, molecule_chembl_id\n"
    )

    count_target = 0

    for prot in target.all():
        prot_chembl_id = prot["target_chembl_id"]
        prot_uniprot_id = prot["cross_references"][0]["xref_id"]
        pref_name = prot["pref_name"]
        # check if prot uniprot id is valid

        if not check_uniprot(prot_uniprot_id):
            prot_uniprot_id = None

        print(prot_chembl_id, prot_uniprot_id, pref_name)
        prot_activities = activity.filter(target_chembl_id=prot_chembl_id).filter(
            standard_type="IC50"
        )

        """
		prot_activities is a list 
		"""

        for t in prot_activities:
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
                    t["molecule_chembl_id"],
                )
            )

        if count_target == LIMIT_TARGET:

            break
        count_target += 1

    print("FILE CREATED")
    f.close()
    return results
