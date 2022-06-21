import requests
import json
import string
import urllib
import tqdm

from html.parser import HTMLParser
from requests.utils import requote_uri
from chembl_webresource_client.new_client import new_client
from InterGraph.target_req import chembl_comp_to_pdb


target = new_client.target
activity = new_client.activity
chembl_pdb_target_dict = {}
chembl_pdb_lig_dict = {}
pdb_target_lig_dict = {}


def retrieve_pdb_data(compounds: str):
    """Given the ligand ChEBML id, this function writes a csv file adding the ligand PDBid to the information stored in data_from_chembl.csv"""
    f = open("raw_data.csv", "w")
    f.write(
        "target_chembl_id, target_uniprot_id, pref_name, canonical_smiles, IC50 uM, molecule_chembl_id, assay_chembl_id, pdb_comp_id\n"
    )

    print("retrieving ", len(compounds))

    ligand_dict = {}

    for c in tqdm(compounds):
        if c[-1] not in ligand_dict.keys():
            ligand_dict[c[-1]] = chembl_comp_to_pdb(c[-1])
        # print(c)
        f.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(*c, ligand_dict[c[-1]]))

    f.close()


def clear_raw_data(filename):
    """This function cleans the raw_data.csv file from compounds whose activity values are not published"""
    with open("cleaned_raw_data.csv", "w") as f:
        with open("raw_data.csv", "r") as g:
            c = 0
            c1 = 0
            for l in g:
                # print("reading {}".format(l))

                if c == 0:
                    f.write(l)
                    # print(l)
                else:
                    l_list = l.split(",")
                    l_last = l_list[-1]
                    if l_last != " None\n":
                        s = ",".join(l_list)
                        assert s == l

                        # print(s)
                        # print("Writing {}".format(l))
                        if len(l_list) == 8:
                            c1 += 1
                            f.write(s)
                c += 1

    # print("lenght file : ", c1)


def retrieve_pdb_from_target(chemblID: str) -> str:
    """Performs an HTTPS request to get target uniprot identifier using the protein ChEMBL"""
    url = "https://www.uniprot.org/uploadlists/"

    params = {
        "from": "CHEMBL_ID",
        "to": "ACC",
        "format": "tab",
        "query": chemblID,
    }
    print("call uniprot db for => {}".format(chemblID))
    data = urllib.parse.urlencode(params)
    data = data.encode("utf-8")
    try:
        req = urllib.request.Request(url, data)
    except:
        return None
    with urllib.request.urlopen(req) as f:
        response = f.read()
    result = response.decode("utf-8")
    # print(result)
    result = result.split("\n")[1].split("\t")[1]

    return result.strip()


def get_pdb_entries_with_pdb_comp(pdb_comp_id: str):
    """This function uses PDB's web service to retrieve crystal structures of the ligand of interest bound to a protein"""

    pdb_q = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_nonpolymer_instance_feature_summary.comp_id",
                "operator": "exact_match",
                "value": "",
            },
        },
        "request_options": {"return_all_hits": True},
        "return_type": "entry",
    }
    pdb_q["query"]["parameters"]["value"] = pdb_comp_id
    pdb_res = requests.get(
        "https://search.rcsb.org/rcsbsearch/v1/query?json=" + json.dumps(pdb_q)
    )
    # print(requote_uri("https://search.rcsb.org/rcsbsearch/v1/query?json=" + json.dumps(pdb_q)))
    if len(pdb_res.text) > 0:
        resp = json.loads(pdb_res.text)
        return [
            resp["result_set"][i]["identifier"]
            for i in range(len(resp["result_set"]))
            if resp["result_set"][i]["score"] == 1
        ]


def match_uniprot_from_pdbids(pdb_ids, uniprot_id) -> list:
    """Query the PDB database on Uniprot identifier"""
    pdb_str = "["
    for pdbid in pdb_ids[:-1]:
        pdb_str += f""" "{pdbid}","""
    pdb_str += f""" "{pdb_ids[-1]}"]"""

    pdb_q1 = (
        """{
  entries(entry_ids:"""
        + pdb_str
        + """) {
    polymer_entities {
      rcsb_id
      rcsb_polymer_entity_container_identifiers {
        reference_sequence_identifiers {
          database_accession
          database_name
        }
      }
    }
  }
}"""
    )
    link = "https://data.rcsb.org/graphql?query=" + pdb_q1
    link = "%5b".join(link.split("["))  # same as replace
    link = "%5d".join(link.split("]"))  # same as replace
    try:
        pdb_res = requests.get(requote_uri(link))

    except requests.exceptions.ConnectionError as e:
        print(e)
        return []
    # print(requote_uri(link))

    # print(pdb_res.content)
    try:
        m = json.loads(pdb_res.text)
    except json.decoder.JSONDecodeError as e:
        print(e)
        return []

    struct_list = []
    for pdbid in m["data"]["entries"]:
        if not pdbid["polymer_entities"]:
            break
        for entity in pdbid["polymer_entities"]:
            pid = entity["rcsb_id"]
            try:
                for db in entity["rcsb_polymer_entity_container_identifiers"][
                    "reference_sequence_identifiers"
                ]:
                    if db["database_name"] == "UniProt":
                        uni_id = db["database_accession"]
                        if uni_id == uniprot_id:
                            struct_list.append(pid)

            except TypeError as e:
                pass
                # print(e)

    return struct_list


def write_clean_raw_data(filename):
    """This function write a csv file containing all the structurale and activity data collected"""
    out = []
    with open(
        "/Users/nedramekni/Documents/PhD/Projects/NLRP3/skelrepo/InterGraph/data/csv/cleaned_raw_data.csv",
        "r",
    ) as f:
        for l in f.readlines()[1:]:
            l = l.split(",")
            target, target_pdb, pdb_lig, activity, smile, assay_chembl_id = (
                l[0].strip(),
                l[1].strip(),
                l[-1].strip(),
                l[4].strip(),
                l[3].strip(),
                l[-2].strip(),
            )

            if target not in chembl_pdb_target_dict.keys():
                if target_pdb == "None":
                    while 1:
                        print("Retrieving pdb from {}".format(target))
                        t = retrieve_pdb_from_target(target)
                        chembl_pdb_target_dict[target] = t
                        if t:
                            break
                else:
                    chembl_pdb_target_dict[target] = target_pdb

            if pdb_lig not in chembl_pdb_lig_dict.keys():
                print("Retrieving pdb_lig entries {}".format(pdb_lig))
                l_r = get_pdb_entries_with_pdb_comp(pdb_lig)
                chembl_pdb_lig_dict[pdb_lig] = l_r

            # l_r = get_pdb_entries_with_pdb_comp(pdb_lig)
            if l_r:
                if (
                    chembl_pdb_target_dict[target]
                    + " ".join(chembl_pdb_lig_dict[pdb_lig])
                    not in pdb_target_lig_dict.keys()
                ):
                    pdb_target_lig_dict[
                        chembl_pdb_target_dict[target]
                        + " ".join(chembl_pdb_lig_dict[pdb_lig])
                    ] = match_uniprot_from_pdbids(
                        chembl_pdb_lig_dict[pdb_lig], chembl_pdb_target_dict[target]
                    )

                if (
                    len(
                        pdb_target_lig_dict[
                            chembl_pdb_target_dict[target]
                            + " ".join(chembl_pdb_lig_dict[pdb_lig])
                        ]
                    )
                    > 0
                ):
                    pdb_target_lig = pdb_target_lig_dict[
                        chembl_pdb_target_dict[target]
                        + " ".join(chembl_pdb_lig_dict[pdb_lig])
                    ]
                    if (
                        "{}, {}, {}, {}, {}, {}, {}".format(
                            chembl_pdb_target_dict[target],
                            target,
                            " ".join(pdb_target_lig),
                            pdb_lig,
                            activity,
                            assay_chembl_id,
                            smile,
                        )
                        not in out
                    ):
                        out += [
                            "{}, {}, {}, {}, {}, {}, {}".format(
                                chembl_pdb_target_dict[target],
                                target,
                                " ".join(pdb_target_lig),
                                pdb_lig,
                                activity,
                                assay_chembl_id,
                                smile,
                            )
                        ]

    with open(filename, "w") as f:
        f.write(
            "target_uniprot_id, target_chembl_id, pdb_code, pdb_lig, activity, assay_chembl_id, smile\n"
        )
        for l in out:
            f.write(l + "\n")
    print("DONE")

    return f
