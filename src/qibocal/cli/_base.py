"""Adds global CLI options."""
import base64
import datetime
import json
import pathlib
import shutil
import socket
import subprocess
import uuid
from dataclasses import asdict
from urllib.parse import urljoin

import click
import yaml
from qibo.config import log, raise_error
from qibolab.serialize import dump_runcard

from ..auto.execute import Executor
from ..auto.history import add_timings_to_meta
from ..auto.mode import ExecutionMode
from ..auto.runcard import Runcard
from ..cli.builders import ReportBuilder
from .utils import (
    META,
    PLATFORM,
    RUNCARD,
    UPDATED_PLATFORM,
    create_qubits_dict,
    generate_meta,
    generate_output_folder,
)

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

# options for report upload
UPLOAD_HOST = (
    "qibocal@localhost"
    if socket.gethostname() == "saadiyat"
    else "qibocal@login.qrccluster.com"
)
TARGET_DIR = "qibocal-reports/"
ROOT_URL = "http://login.qrccluster.com:9000/"


@click.group()
def command():
    """Welcome to Qibocal!
    Qibo module to calibrate and characterize self-hosted QPUS.
    """


@command.command(context_settings=CONTEXT_SETTINGS)
@click.argument("runcard", metavar="RUNCARD", type=click.Path(exists=True))
@click.option(
    "folder",
    "-o",
    type=click.Path(),
    help="Output folder. If not provided a standard name will generated.",
)
@click.option(
    "force",
    "-f",
    is_flag=True,
    help="Use --force option to overwrite the output folder.",
)
@click.option(
    "--update/--no-update",
    default=True,
    help="Use --no-update option to avoid updating iteratively the platform."
    "With this option the new runcard will not be produced.",
)
def auto(runcard, folder, force, update):
    """Autocalibration

    Arguments:

     - RUNCARD: runcard with declarative inputs.
    """
    # load and initialize Runcard from file
    card = yaml.safe_load(pathlib.Path(runcard).read_text(encoding="utf-8"))
    runcard = Runcard.load(card)

    # generate output folder
    path = generate_output_folder(folder, force)
    # generate meta
    meta = generate_meta(runcard, path)

    # dump platform
    if runcard.backend == "qibolab":
        dump_runcard(runcard.platform_obj, path / PLATFORM)
    # dump action runcard
    (path / RUNCARD).write_text(yaml.dump(asdict(runcard)))
    # dump meta
    (path / META).write_text(json.dumps(meta, indent=4))

    # allocate qubits, runcard and executor
    qubits = create_qubits_dict(runcard)
    platform = runcard.platform_obj
    executor = Executor.load(runcard, path, platform, qubits, update)

    # connect and initialize platform
    if platform is not None:
        platform.connect()
        platform.setup()
        platform.start()

    # run protocols
    executor.run(mode=ExecutionMode.autocalibration)

    # stop and disconnect platform
    if platform is not None:
        platform.stop()
        platform.disconnect()

    # dump updated runcard
    if platform is not None:
        dump_runcard(platform, path / UPDATED_PLATFORM)

    # dump updated meta
    meta = add_timings_to_meta(meta, executor.history)
    e = datetime.datetime.now(datetime.timezone.utc)
    meta["end-time"] = e.strftime("%H:%M:%S")
    (path / META).write_text(json.dumps(meta, indent=4))


@command.command(context_settings=CONTEXT_SETTINGS)
@click.argument("runcard_path", metavar="RUNCARD", type=click.Path(exists=True))
@click.option(
    "folder",
    "-o",
    type=click.Path(),
    help="Output folder. If not provided a standard name will generated.",
)
@click.option(
    "force",
    "-f",
    is_flag=True,
    help="Use --force option to overwrite the output folder.",
)
def acquire(runcard_path, folder, force):
    """Data acquisition

    Arguments:

     - RUNCARD: runcard with declarative inputs.
    """
    # load and initialize Runcard from file
    card = yaml.safe_load(pathlib.Path(runcard_path).read_text(encoding="utf-8"))
    runcard = Runcard.load(card)

    # generate output folder
    path = generate_output_folder(folder, force)
    # generate meta
    meta = generate_meta(runcard, path)

    # dump platform
    if runcard.backend == "qibolab":
        dump_runcard(runcard.platform_obj, path / PLATFORM)

    # dump action runcard
    (path / RUNCARD).write_text(yaml.dump(asdict(runcard)))
    # dump meta
    (path / META).write_text(json.dumps(meta, indent=4))

    # allocate qubits, runcard and executor
    qubits = create_qubits_dict(runcard)
    platform = runcard.platform_obj
    executor = Executor.load(runcard, path, platform, qubits)

    # connect and initialize platform
    if platform is not None:
        platform.connect()
        platform.setup()
        platform.start()

    # run protocols
    executor.run(mode=ExecutionMode.acquire)

    # stop and disconnect platform
    if platform is not None:
        platform.stop()
        platform.disconnect()

    # dump updated meta
    meta = add_timings_to_meta(meta, executor.history)
    e = datetime.datetime.now(datetime.timezone.utc)
    meta["end-time"] = e.strftime("%H:%M:%S")
    (path / META).write_text(json.dumps(meta, indent=4))


@command.command(context_settings=CONTEXT_SETTINGS)
@click.argument("folder", metavar="folder", type=click.Path(exists=True))
def report(folder):
    """Report generation

    Arguments:

    - FOLDER: input folder.

    """
    # load path, meta and runcard
    path = pathlib.Path(folder)
    meta = yaml.safe_load((path / META).read_text())
    runcard = Runcard.load(yaml.safe_load((path / RUNCARD).read_text()))

    # load executor
    executor = Executor.load(runcard, path)

    # produce html
    builder = ReportBuilder(path, runcard.qubits, executor, meta)
    builder.run(path)


@command.command(context_settings=CONTEXT_SETTINGS)
@click.argument("folder", metavar="folder", type=click.Path(exists=True))
@click.option(
    "--update/--no-update",
    default=True,
    help="Use --no-update option to avoid updating iteratively the platform."
    "With this option the new runcard will not be produced.",
)
def fit(folder, update):
    """Post-processing analysis

    Arguments:

    - FOLDER: input folder.

    """
    # load path, meta, runcard and executor
    path = pathlib.Path(folder)
    meta = yaml.safe_load((path / META).read_text())
    runcard = Runcard.load(yaml.safe_load((path / RUNCARD).read_text()))
    executor = Executor.load(
        runcard, path, update=update, platform=runcard.platform_obj
    )

    # perform post-processing
    executor.run(mode=ExecutionMode.fit)

    # dump updated runcard
    if runcard.platform_obj is not None and update:
        dump_runcard(runcard.platform_obj, path / UPDATED_PLATFORM)

    # update time in meta.yml
    meta = add_timings_to_meta(meta, executor.history)
    e = datetime.datetime.now(datetime.timezone.utc)
    meta["end-time"] = e.strftime("%H:%M:%S")
    (path / META).write_text(json.dumps(meta, indent=4))


@command.command(context_settings=CONTEXT_SETTINGS)
@click.argument("folder", metavar="FOLDER", type=click.Path(exists=True))
def upload(folder):
    """Uploads output folder to server

    Arguments:

    - FOLDER: input folder.
    """

    output_path = pathlib.Path(folder)

    # check the rsync command exists.
    if not shutil.which("rsync"):
        raise_error(
            RuntimeError,
            "Could not find the rsync command. Please make sure it is installed.",
        )

    # check that we can authentica with a certificate
    ssh_command_line = (
        "ssh",
        "-o",
        "PreferredAuthentications=publickey",
        "-q",
        UPLOAD_HOST,
        "exit",
    )

    str_line = " ".join(repr(ele) for ele in ssh_command_line)

    log.info(f"Checking SSH connection to {UPLOAD_HOST}.")

    try:
        subprocess.run(ssh_command_line, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            (
                "Could not validate the SSH key. "
                "The command\n%s\nreturned a non zero exit status. "
                "Please make sure that your public SSH key is on the server."
            )
            % str_line
        ) from e
    except OSError as e:
        raise RuntimeError(
            "Could not run the command\n{}\n: {}".format(str_line, e)
        ) from e

    log.info("Connection seems OK.")

    # upload output
    randname = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode()
    newdir = TARGET_DIR + randname

    rsync_command = (
        "rsync",
        "-aLz",
        "--chmod=ug=rwx,o=rx",
        f"{output_path}/",
        f"{UPLOAD_HOST}:{newdir}",
    )

    log.info(f"Uploading output ({output_path}) to {UPLOAD_HOST}")
    try:
        subprocess.run(rsync_command, check=True)
    except subprocess.CalledProcessError as e:
        msg = f"Failed to upload output: {e}"
        raise RuntimeError(msg) from e

    url = urljoin(ROOT_URL, randname)
    log.info(f"Upload completed. The result is available at:\n{url}")
