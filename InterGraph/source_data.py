import requests
import json
import string
import urllib
import tqdm
import re
import time
import zlib

from html.parser import HTMLParser
from requests.utils import requote_uri
from chembl_webresource_client.new_client import new_client
from InterGraph.target_req import chembl_comp_to_pdb_new
from xml.etree import ElementTree
from urllib.parse import urlparse, parse_qs, urlencode
from requests.adapters import HTTPAdapter, Retry


target = new_client.target
activity = new_client.activity
chembl_pdb_target_dict = {}
chembl_pdb_lig_dict = {}
pdb_target_lig_dict = {}
POLLING_INTERVAL = 3

""""Programmatic access to UniProt website retrieving protein UniProt id using ChEMBL id as input"""
########################

API_URL = "https://rest.uniprot.org"


retries = Retry(total=10, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))


def submit_id_mapping(from_db, to_db, ids):
    request = requests.post(
        f"{API_URL}/idmapping/run",
        data={"from": from_db, "to": to_db, "ids": ",".join(ids)},
    )
    request.raise_for_status()
    return request.json()["jobId"]


def get_next_link(headers):
    re_next_link = re.compile(r'<(.+)>; rel="next"')
    if "Link" in headers:
        match = re_next_link.match(headers["Link"])
        if match:
            return match.group(1)


def check_id_mapping_results_ready(job_id):
    while True:
        request = session.get(f"{API_URL}/idmapping/status/{job_id}")
        request.raise_for_status()
        j = request.json()
        if "jobStatus" in j:
            if j["jobStatus"] == "RUNNING":
                # print(f"Retrying in {POLLING_INTERVAL}s")
                time.sleep(POLLING_INTERVAL)
            else:
                raise Exception(request["jobStatus"])
        else:
            return bool(j["results"] or j["failedIds"])


def get_batch(batch_response, file_format, compressed):
    batch_url = get_next_link(batch_response.headers)
    while batch_url:
        batch_response = session.get(batch_url)
        batch_response.raise_for_status()
        yield decode_results(batch_response, file_format, compressed)
        batch_url = get_next_link(batch_response.headers)


def combine_batches(all_results, batch_results, file_format):
    if file_format == "json":
        for key in ("results", "failedIds"):
            if key in batch_results and batch_results[key]:
                all_results[key] += batch_results[key]
    elif file_format == "tsv":
        return all_results + batch_results[1:]
    else:
        return all_results + batch_results
    return all_results


def get_id_mapping_results_link(job_id):
    url = f"{API_URL}/idmapping/details/{job_id}"
    request = session.get(url)
    request.raise_for_status()
    return request.json()["redirectURL"]


def decode_results(response, file_format, compressed):
    if compressed:
        decompressed = zlib.decompress(response.content, 16 + zlib.MAX_WBITS)
        if file_format == "json":
            j = json.loads(decompressed.decode("utf-8"))
            return j
        elif file_format == "tsv":
            return [line for line in decompressed.decode("utf-8").split("\n") if line]
        elif file_format == "xlsx":
            return [decompressed]
        elif file_format == "xml":
            return [decompressed.decode("utf-8")]
        else:
            return decompressed.decode("utf-8")
    elif file_format == "json":
        return response.json()
    elif file_format == "tsv":
        return [line for line in response.text.split("\n") if line]
    elif file_format == "xlsx":
        return [response.content]
    elif file_format == "xml":
        return [response.text]
    return response.text


def get_xml_namespace(element):
    m = re.match(r"\{(.*)\}", element.tag)
    return m.groups()[0] if m else ""


def merge_xml_results(xml_results):
    merged_root = ElementTree.fromstring(xml_results[0])
    for result in xml_results[1:]:
        root = ElementTree.fromstring(result)
        for child in root.findall("{http://uniprot.org/uniprot}entry"):
            merged_root.insert(-1, child)
    ElementTree.register_namespace("", get_xml_namespace(merged_root[0]))
    return ElementTree.tostring(merged_root, encoding="utf-8", xml_declaration=True)


def print_progress_batches(batch_index, size, total):
    n_fetched = min((batch_index + 1) * size, total)
    # print(f"Fetched: {n_fetched} / {total}")


def get_id_mapping_results_search(url):
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    file_format = query["format"][0] if "format" in query else "json"
    if "size" in query:
        size = int(query["size"][0])
    else:
        size = 500
        query["size"] = size
    compressed = (
        query["compressed"][0].lower() == "true" if "compressed" in query else False
    )
    parsed = parsed._replace(query=urlencode(query, doseq=True))
    url = parsed.geturl()
    request = session.get(url)
    request.raise_for_status()
    results = decode_results(request, file_format, compressed)
    total = int(request.headers["x-total-results"])
    print_progress_batches(0, size, total)
    for i, batch in enumerate(get_batch(request, file_format, compressed), 1):
        results = combine_batches(results, batch, file_format)
        # print_progress_batches(i, size, total)
    if file_format == "xml":
        return merge_xml_results(results)
    return results


def get_id_mapping_results_stream(url):
    if "/stream/" not in url:
        url = url.replace("/results/", "/stream/")
    request = session.get(url)
    request.raise_for_status()
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    file_format = query["format"][0] if "format" in query else "json"
    compressed = (
        query["compressed"][0].lower() == "true" if "compressed" in query else False
    )
    return decode_results(request, file_format, compressed)


########################


def retrieve_pdb_data(compounds: str):
    """Given the ligand ChEBML id, this function writes a csv file adding the ligand PDBid to the information stored in data_from_chembl.csv"""
    f = open("../data/csv/raw_data.csv", "w")
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


def clear_raw_data(inputfile, outputfile):
    """This function cleans the raw_data.csv file from compounds whose activity values are not published"""
    with open(outputfile, "w") as f:
        with open(inputfile, "r") as g:
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


def retrieve_pdb_from_target(chemblID_list: list) -> dict:
    job_id = submit_id_mapping(from_db="ChEMBL", to_db="UniProtKB", ids=chemblID_list)
    if check_id_mapping_results_ready(job_id):
        link = get_id_mapping_results_link(job_id)
        results = get_id_mapping_results_search(link)

    results = results["results"]
    """
    return a dict where keys are chemblcode and value the corresponding uniprotid
    """
    results = {result["from"]: result["to"]["primaryAccession"] for result in results}
    return results


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
    url_encoded = requote_uri(
        "https://search.rcsb.org/rcsbsearch/v2/query?json=" + json.dumps(pdb_q)
    )
    pdb_res = requests.get(url_encoded)
    # print(requote_uri("https://search.rcsb.org/rcsbsearch/v1/query?json=" + json.dumps(pdb_q)))
    # exit()
    if len(pdb_res.text) > 0 and "result_set" in json.loads(pdb_res.text).keys():
        resp = json.loads(pdb_res.text)
        # print(resp)
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


def write_clean_raw_data(inputfile, outputfile):
    """This function write a csv file containing all the structurale and activity data collected"""
    out, targets = [], []
    with open(inputfile, "r") as f:
        for l in f.readlines()[1:]:
            l = l.split(",")
            targets.append(l[0].strip())

    chembl_pdb_target_dict = retrieve_pdb_from_target(targets)
    # print(chembl_pdb_target_dict)

    with open(
        inputfile,
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
                continue

            if pdb_lig not in chembl_pdb_lig_dict.keys():
                # print("Retrieving pdb_lig entries {}".format(pdb_lig))
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
                    # print('first if executed')
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

    with open(outputfile, "w") as f:
        f.write(
            "target_uniprot_id, target_chembl_id, pdb_code, pdb_lig, activity, assay_chembl_id, smile\n"
        )
        for l in out:
            f.write(l + "\n")
    print("DONE")
