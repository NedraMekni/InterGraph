import os
import pypdb
import urllib

# from prody import *

import openbabel
from openbabel import openbabel

pdb_raw_d = "pdb_raw"


if not os.path.exists("data"):
    os.makedirs("data")

if not os.path.exists("pdb_raw"):
    os.makedirs("pdb_raw")


def get_pdb_components(pdb_id):
    global pdb_raw_d
    pdb_r = urllib.request.urlretrieve(
        "https://files.rcsb.org/download/{}.pdb".format(pdb_id),
        "{}/{}.pdb".format(pdb_raw_d, pdb_id),
    )


def parsePDB(lig, filename):
    prot = []
    ligs = []

    with open(filename, "r") as f:
        for l in f:
            if l.split()[0] == "ATOM":
                prot += [l]
            if len(l.split()) > 3 and l.split()[3] == lig:
                ligs += [l]

    return prot, ligs


def write_pdb(prot, lig, filename):
    with open(filename, "w") as f:
        for l in prot:
            f.write(l)
        f.write("TER\n")
        for l in lig:
            f.write(l)


def write_ligand_pdb(lig, filename):

    with open(filename, "w") as f:
        for l in lig:
            l.split(",")
            f.write(l)


if __name__ == "__main__":

    # mol2_output_path =
    pdb_prot_downloaded = []

    with open("data.csv", "r") as f:
        lines = f.readlines()[1:]
        for l in lines:
            # print(l)
            pdb_prot_list, pdb_lig = (
                l.strip().split(",")[2].strip(),
                l.strip().split(",")[3].strip(),
            )
            pdb_prot_list = [x[:4] for x in pdb_prot_list.split()]

            # print(pdb_lig_list)

            # print(pdb_prot_list)

            for pdb_prot in pdb_prot_list:
                if pdb_prot not in pdb_prot_downloaded:
                    x = get_pdb_components(pdb_prot)
                    pdb_prot_downloaded += [pdb_prot]

                prot, lig = parsePDB(pdb_lig, pdb_raw_d + "/" + pdb_prot + ".pdb")

                if not os.path.exists("data/" + pdb_prot + "_" + pdb_lig):
                    os.makedirs("data/" + pdb_prot + "_" + pdb_lig)
                    write_pdb(
                        prot,
                        lig,
                        "data/"
                        + pdb_prot
                        + "_"
                        + pdb_lig
                        + "/"
                        + pdb_prot
                        + "_"
                        + pdb_lig
                        + ".pdb",
                    )
                    if not os.path.exists(
                        "data/" + pdb_prot + "_" + pdb_lig + "/" + pdb_lig + ".pdb"
                    ):
                        write_ligand_pdb(
                            lig,
                            "data/" + pdb_prot + "_" + pdb_lig + "/" + pdb_lig + ".pdb",
                        )

                        for file in os.listdir(
                            "data/" + pdb_prot + "_" + pdb_lig + "/"
                        ):
                            if file.endswith(f"{pdb_lig}.pdb"):
                                file_path = (
                                    "data/"
                                    + pdb_prot
                                    + "_"
                                    + pdb_lig
                                    + "/"
                                    + pdb_lig
                                    + ".pdb"
                                )
                                obConversion = openbabel.OBConversion()
                                obConversion.SetInAndOutFormats("pdb", "mol2")
                                mol = openbabel.OBMol()
                                obConversion.ReadFile(mol, file_path)
                                mol.AddHydrogens()
                                obConversion.WriteFile(
                                    mol,
                                    "data/"
                                    + pdb_prot
                                    + "_"
                                    + pdb_lig
                                    + "/"
                                    + f"{pdb_lig}.mol2",
                                )  # else:
                    # print("The write is skipped")
