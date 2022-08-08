from urllib import response
import requests
import json
import csv
import os
import pytest

from chembl_webresource_client.new_client import new_client
from InterGraph.source_data import retrieve_pdb_data, write_clean_raw_data
from InterGraph.target_req import chembl_comp_to_pdb_new, retrieve_chembl_data

target = new_client.target
activity = new_client.activity


# Here we test that the https request succeded by checking the responde status code
# Function argument is compound chembl ID
@pytest.fixture
def chembl_compound_id():
    return chembl_compound_id == "CHEMBL2114210"


def test_chembl_comp_to_pdb_new(chembl_compound_id: str):
    from InterGraph.target_req import chembl_comp_to_pdb_new

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
    assert response.status_code == 200


# test ChEMBL websource client handled the interaction with the HTTPS protocol and only the required values are cached
def test_retrieve_chembl_data():

    results = []

    get = target.get("CHEMBL202")
    prot_chembl_id = get["target_chembl_id"]
    prot_uniprot_id = get["cross_references"][0]["xref_id"]
    pref_name = get["pref_name"]
    prot_activities = activity.filter(target_chembl_id=prot_chembl_id).filter(
        standard_type="IC50"
    )

    #t = prot_activities
    try:
        get_target_activity = []
        for prot_activity in prot_activities:
            get_target_activity.append(prot_activity)
    except:
        print('target skipped, activity not found')
    for t in get_target_activity:
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

    counter = len(results[0])
    assert counter == 7


# Test that cleaned_raw_data.csv exists, it is not empty and has the expected number of header columns
def test_write_clean_raw_data():

    with open("../../InterGraph/data/csv/cleaned_raw_data.csv") as csv_file:
        reader = csv.reader(csv_file)
        csv_to_list = list(reader)
    header = len(csv_to_list[0])
    assert header == 8

    filesize = os.path.getsize("../../InterGraph/data/csv/cleaned_raw_data.csv")
    assert filesize != 0
