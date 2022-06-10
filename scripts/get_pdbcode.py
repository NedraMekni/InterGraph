import urllib

from requests.utils import requote_uri

import requests
import json


def retrieve_pdb_from_target(chemblID):
    url = "https://www.uniprot.org/uploadlists/"

    params = {
        "from": "CHEMBL_ID",
        "to": "ACC",
        "format": "tab",
        "query": chemblID,  # valorizziamolo noi
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


def get_pdb_entries_with_pdb_comp(pdb_comp_id):
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


def match_uniprot_from_pdbids(pdb_ids, uniprot_id):
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

    # pdb_res = requests.get("https://data.rcsb.org/graphql/index.html?query=" + pdb_q1)
    # print(link)
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


if __name__ == "__main__":

    chembl_pdb_target_dict = {}
    chembl_pdb_lig_dict = {}
    pdb_target_lig_dict = {}

    out = []
    with open("cleaned_raw_data.csv", "r") as f:
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

    with open("./data.csv", "w") as f:
        f.write(
            "target_uniprot_id, target_chembl_id, pdb_code, pdb_lig, activity, assay_chembl_id, smile\n"
        )
        for l in out:
            f.write(l + "\n")
    print("DONE")
