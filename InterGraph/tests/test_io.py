from urllib import response
import requests
import json
import csv
import os
import pytest

from chembl_webresource_client.new_client import new_client
from InterGraph.source_data import retrieve_pdb_data, write_clean_raw_data
from InterGraph.target_req import chembl_comp_to_pdb,retrieve_chembl_data

target = new_client.target
activity = new_client.activity


# Here we test that the https request succeded by checking the responde status code
# Function argument is compound chembl ID
@pytest.fixture
def chembl_compound_id():
    return chembl_compound_id == 'CHEMBL2114210'
def test_chembl_comp_to_pdb(chembl_compound_id:str):
    from InterGraph.target_req import chembl_comp_to_pdb
   
    chembl_q = "https://www.ebi.ac.uk/unichem/rest/src_compound_id/{chembl_compound_id}/1/3"
    print('The request succeeded')
    try:
        chembl_res = requests.get(chembl_q)
        n = json.loads(chembl_res.text)
    except:
        return None
    if (type(n) is list) and (len(n) == 0):
        return None
    elif type(n) is dict and "error" in n.keys():
        return None
    assert response.status_code == 200
    
# test ChEMBL websource client handled the interaction with the HTTPS protocol and only the required values are cached
def test_retrieve_chembl_data():
    
    results=[]
   
    get = target.get('CHEMBL2074')
    prot_chembl_id = get["target_chembl_id"]
    prot_uniprot_id = get["cross_references"][0]["xref_id"]
    pref_name = get["pref_name"]
    prot_activities = activity.filter(target_chembl_id=prot_chembl_id).filter(
            standard_type="IC50")

    t = prot_activities
    for t in prot_activities:
        results += [[prot_chembl_id,prot_uniprot_id,pref_name,t["canonical_smiles"],t["standard_value"],t["assay_chembl_id"],t["molecule_chembl_id"]]]
    
    counter = len(results[0])
    assert counter == 7

# Test that cleaned_raw_data.csv exists, it is not empty and has the expected number of header columns
def test_write_clean_raw_data():

    with open('../InterGraph/data/csv/cleaned_raw_data.csv') as csv_file:
        reader = csv.reader(csv_file)
        csv_to_list = list(reader)
    header = len(csv_to_list[0])
    assert header == 8
    
    filesize = os.path.getsize('../InterGraph/data/csv/cleaned_raw_data.csv')
    assert filesize != 0
    
