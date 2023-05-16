"""Helper functions for the cli module"""

import datetime
import getpass
import os
import shutil
from glob import glob

import click
import yaml

from qibocal.config import log, raise_error


def folders_exists(folders):
    """Check if a list of folders exists

    Args:
        folders (list): list of absolute or relative path to folders
    """
    foldernames = []
    for foldername in folders:
        expanded = list(glob(foldername))
        if len(expanded) == 0 and "*" not in foldername:
            raise (click.BadParameter("file '{}' not found".format(foldername)))
        foldernames.extend(expanded)

    return foldernames


def check_folder_structure(folderList):
    """Check if a list of folders share structure between them

    Args:
        folders (list): list of absolute or relative path to folders
    """
    all_subdirList = []
    for folder in folderList:
        folder_subdirList = []
        for dirName, subdirList, fileList in os.walk(folder):
            folder_subdirList.append(subdirList)
        all_subdirList.append(folder_subdirList)

    return all(x == all_subdirList[0] for x in all_subdirList)


def update_meta(metadata, metadata_new, target_dir="qq-compare"):
    """Update meta.yml file

    Args:
        metadata (dict): dictionary with the meta.yml actual parameters and values
        metadata_new (dict): dictionary with the new parameters and values to update in the actual meta.yml
    """

    metadata["backend"] = metadata["backend"] + " , " + metadata_new["backend"]
    metadata["date"] = metadata["date"] + " , " + metadata_new["date"]
    metadata["end-time"] = metadata["end-time"] + " , " + metadata_new["end-time"]
    metadata["platform"] = metadata["platform"] + " , " + metadata_new["platform"]
    metadata["start-time"] = metadata["start-time"] + " , " + metadata_new["start-time"]
    metadata["title"] = metadata["title"] + " , " + metadata_new["title"]
    metadata["versions"]["numpy"] = (
        metadata["versions"]["numpy"] + " , " + metadata_new["versions"]["numpy"]
    )
    metadata["versions"]["qibo"] = (
        metadata["versions"]["qibo"] + " , " + metadata_new["versions"]["qibo"]
    )
    metadata["versions"]["qibocal"] = (
        metadata["versions"]["qibocal"] + " , " + metadata_new["versions"]["qibocal"]
    )
    metadata["versions"]["qibolab"] = (
        metadata["versions"]["qibolab"] + " , " + metadata_new["versions"]["qibolab"]
    )
    with open(f"{target_dir}/meta.yml", "w") as file:
        yaml.safe_dump(metadata, file)


def update_runcard(rundata, rundata_new, target_compare_dir):
    """Update runcard.yml file

    Args:
        rundata (dict): dictionary with the runcard.yml actual parameters and values
        rundata_new (dict): dictionary with the new parameters and values to update in the actual runcard.yml
    """

    rundata["platform"] = rundata["platform"] + " , " + rundata_new["platform"]
    unique = list(set(rundata["qubits"] + rundata_new["qubits"]))
    rundata["qubits"] = unique
    with open(f"{target_compare_dir}/runcard.yml", "w") as file:
        yaml.safe_dump(
            rundata,
            file,
            indent=4,
            allow_unicode=False,
            sort_keys=False,
            default_flow_style=None,
        )


def load_yaml(path):
    """Load yaml file from disk."""
    with open(path) as file:
        data = yaml.safe_load(file)
    return data


def generate_output_folder(folder, force):
    """Generation of qq output folder

    Args:
        folder (path): path for the output folder. If None it will be created a folder automatically
        force (bool): option to overwrite the output folder if it exists already.

    Returns:
        Output path.
    """
    if folder is None:
        e = datetime.datetime.now()
        user = getpass.getuser().replace(".", "-")
        date = e.strftime("%Y-%m-%d")
        folder = f"{date}-{'000'}-{user}"
        num = 0
        while os.path.exists(folder):
            log.info(f"Directory {folder} already exists.")
            num += 1
            folder = f"{date}-{str(num).rjust(3, '0')}-{user}"
            log.info(f"Trying to create directory {folder}")
    elif os.path.exists(folder) and not force:
        raise_error(RuntimeError, f"Directory {folder} already exists.")
    elif os.path.exists(folder) and force:
        log.warning(f"Deleting previous directory {folder}.")
        shutil.rmtree(os.path.join(os.getcwd(), folder))

    path = os.path.join(os.getcwd(), folder)
    log.info(f"Creating directory {folder}.")
    os.makedirs(path)
    return folder
