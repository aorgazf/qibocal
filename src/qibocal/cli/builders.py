import datetime
import filecmp
import inspect
import os
import pathlib
import shutil

import yaml

from qibocal import calibrations
from qibocal.config import log, raise_error
from qibocal.data import Data, DataUnits
from qibocal.plots.utils import get_color, get_data_subfolders


def load_yaml(path):
    """Load yaml file from disk."""
    with open(path) as file:
        data = yaml.safe_load(file)
    return data


class ActionBuilder:
    """Class for parsing and executing runcards.
    Args:
        runcard (path): path containing the runcard.
        folder (path): path for the output folder.
        force (bool): option to overwrite the output folder if it exists already.
    """

    def __init__(self, runcard, folder=None, force=False, monitor=False):
        self.monitor = monitor
        self.runcard_path = runcard
        self.runcard = load_yaml(runcard)
        path, self.folder = self._generate_output_folder(folder, force)

        # Qibolab default backend if not provided in runcard.
        backend_name = self.runcard.get("backend", "qibolab")
        platform_name = self.runcard.get("platform", "dummy")
        platform_runcard = self.runcard.get("runcard", None)
        self.backend, self.platform = self._allocate_backend(
            backend_name, platform_name, path, platform_runcard
        )
        self.qubits = self.runcard["qubits"]
        self.format = self.runcard["format"]

        # Saving runcard
        shutil.copy(runcard, f"{path}/runcard.yml")
        self.save_meta(path, self.folder)

    def _generate_output_folder(self, folder, force):
        """Static method for generating the output folder. If self.monitor is True,
        it will inspect all directories containing a "runcard.yml" file. If
        the self.runcard is the same, it will use the same folder. Otherwise, it will
        create a new folder if the folder is not provided, and not forced.

        Args:
            folder (path): path for the output folder. If None it will be created a folder automatically
            force (bool): option to overwrite the output folder if it exists already.
        """
        current_directory = pathlib.Path.cwd()

        if self.monitor and folder is None:
            directories_with_runcard = [
                entry.path
                for entry in os.scandir(current_directory)
                if entry.is_dir() and "runcard.yml" in os.listdir(entry.path)
            ]
            for directory in directories_with_runcard:
                if filecmp.cmp(self.runcard_path, f"{directory}/runcard.yml"):
                    log.info(f"Found previous directory {directory}.")
                    return current_directory / directory, os.path.basename(directory)
        if folder is None:
            import getpass

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
            if self.monitor:
                if not os.path.exists(f"{folder}/runcard.yml") and not filecmp.cmp(
                    self.runcard_path, f"{folder}/runcard.yml"
                ):
                    raise_error(
                        ValueError,
                        f"Directory {folder} does not contain the same runcard.",
                    )
                log.info(f"Found previous directory {folder}.")
                return folder, os.path.basename(folder)
            else:
                raise_error(RuntimeError, f"Directory {folder} already exists.")

        elif os.path.exists(folder) and force:
            if not os.path.exists(f"{folder}/runcard.yml"):
                raise_error(
                    ValueError,
                    f"Directory {folder} might not be a qibocal directory. Aborting its deletion.",
                )
            log.warning(f"Deleting previous directory {folder}.")
            shutil.rmtree(os.path.join(current_directory, folder))

        path = os.path.join(current_directory, folder)
        log.info(f"Creating directory {folder}.")
        os.makedirs(path)
        return path, folder

    def _allocate_backend(self, backend_name, platform_name, path, platform_runcard):
        """Allocate the platform using Qibolab."""
        from qibo.backends import GlobalBackend, set_backend

        if backend_name == "qibolab":
            if platform_runcard is None:
                from qibolab.paths import qibolab_folder

                original_runcard = qibolab_folder / "runcards" / f"{platform_name}.yml"
            else:
                original_runcard = platform_runcard
            # copy of the original runcard that will stay unmodified
            shutil.copy(original_runcard, f"{path}/platform.yml")
            # copy of the original runcard that will be modified during calibration
            updated_runcard = f"{self.folder}/new_platform.yml"
            shutil.copy(original_runcard, updated_runcard)
            # allocate backend with updated_runcard
            set_backend(
                backend=backend_name, platform=platform_name, runcard=updated_runcard
            )
            backend = GlobalBackend()
            return backend, backend.platform
        else:
            set_backend(backend=backend_name, platform=platform_name)
            backend = GlobalBackend()
            return backend, None

    def save_meta(self, path, folder):
        import qibocal

        e = datetime.datetime.now(datetime.timezone.utc)
        meta = {}
        meta["title"] = folder
        meta["backend"] = str(self.backend)
        meta["platform"] = str(self.backend.platform)
        meta["date"] = e.strftime("%Y-%m-%d")
        meta["start-time"] = e.strftime("%H:%M:%S")
        meta["end-time"] = e.strftime("%H:%M:%S")
        meta["versions"] = self.backend.versions  # pylint: disable=E1101
        meta["versions"]["qibocal"] = qibocal.__version__

        with open(f"{path}/meta.yml", "w") as file:
            yaml.dump(meta, file)

    def _build_single_action(self, name):
        """Helper method to parse the actions in the runcard."""
        f = getattr(calibrations, name)
        path = os.path.join(self.folder, f"data/{name}/")
        os.makedirs(path, exist_ok=True)
        sig = inspect.signature(f)
        params = self.runcard["actions"][name]
        for param in list(sig.parameters)[2:-1]:
            if param not in params:
                raise_error(AttributeError, f"Missing parameter {param} in runcard.")

        return f, params, path

    def execute(self):
        """Method to execute sequentially all the actions in the runcard."""
        if self.platform is not None:
            self.platform.connect()
            self.platform.setup()
            self.platform.start()

        while True:
            for action in self.runcard["actions"]:
                routine, args, path = self._build_single_action(action)
                self._execute_single_action(routine, args, path)
                for qubit in self.qubits:
                    if self.platform is not None:
                        self.update_platform_runcard(qubit, action)
            if not self.monitor:
                break

        if self.platform is not None:
            self.platform.stop()
            self.platform.disconnect()

    def _execute_single_action(self, routine, arguments, path):
        """Method to execute a single action and retrieving the results.
        It will add a timestamp column to the results and append the data if the
        file already exists.
        """
        if self.format is None:
            raise_error(ValueError, f"Cannot store data using {self.format} format.")

        results = routine(self.platform, self.qubits, **arguments)

        if self.monitor:
            import datetime

            import pandas as pd

            # Generate all the data to only keep the data when the routine is completed
            datas = {}
            for result in results:
                if result.name not in datas:
                    datas[result.name] = result

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            for data in datas.values():
                data.options += ["timestamp"]
                if not os.path.isfile(
                    os.path.join(path + f"{data.name}.{self.format}")
                ):
                    data.df = data.df.assign(timestamp=current_time)
                    getattr(data, f"to_{self.format}")(path)
                else:
                    sub_folder = get_data_subfolders(self.folder)
                    if len(sub_folder) > 1:
                        raise_error(
                            ValueError, f"More than one subfolder found in {path}."
                        )
                    data_old = data.load_data(
                        self.folder,
                        sub_folder[0],
                        routine.__name__,
                        self.format,
                        data.name,
                    )
                    data.df = data.df.assign(timestamp=current_time)
                    data.df = pd.concat([data_old.df, data.df], ignore_index=True)
                    getattr(data, f"to_{self.format}")(path)
        else:
            for data in results:
                getattr(data, f"to_{self.format}")(path)

    def update_platform_runcard(self, qubit, routine):
        try:
            data_fit = Data.load_data(self.folder, "data", routine, self.format, "fits")
            data_fit.df = data_fit.df[data_fit.df["qubit"] == qubit]
        except:
            data_fit = Data()

        settings = load_yaml(f"{self.folder}/new_platform.yml")

        # Parameters to change in the platform runcard
        params = settings["characterization"]["single_qubit"][qubit].keys()

        for param in params:
            if param in data_fit.df.columns:
                settings["characterization"]["single_qubit"][qubit][param] = int(
                    data_fit.get_values(param)
                )

        with open(f"{self.folder}/new_platform.yml", "w") as file:
            yaml.dump(
                settings, file, sort_keys=False, indent=4, default_flow_style=None
            )

    def dump_report(self):
        from qibocal.web.report import create_report

        # update end time
        meta = load_yaml(f"{self.folder}/meta.yml")
        e = datetime.datetime.now(datetime.timezone.utc)
        meta["end-time"] = e.strftime("%H:%M:%S")
        with open(f"{self.folder}/meta.yml", "w") as file:
            yaml.dump(meta, file)

        create_report(self.folder)


class ReportBuilder:
    """Parses routines and plots to report and live plotting page.

    Args:
        path (str): Path to the data folder to generate report for.
    """

    def __init__(self, path):
        self.path = path
        self.metadata = load_yaml(os.path.join(path, "meta.yml"))

        # find proper path title
        base, self.title = os.path.join(os.getcwd(), path), ""
        while self.title in ("", "."):
            base, self.title = os.path.split(base)

        self.runcard = load_yaml(os.path.join(path, "runcard.yml"))
        self.format = self.runcard.get("format")
        self.qubits = self.runcard.get("qubits")

        # create calibration routine objects
        # (could be incorporated to :meth:`qibocal.cli.builders.ActionBuilder._build_single_action`)
        self.routines = []
        for action in self.runcard.get("actions"):
            if hasattr(calibrations, action):
                routine = getattr(calibrations, action)
            else:
                raise_error(ValueError, f"Undefined action {action} in report.")

            if not hasattr(routine, "plots"):
                routine.plots = []
            self.routines.append(routine)

    def get_routine_name(self, routine):
        """Prettify routine's name for report headers."""
        return routine.__name__.replace("_", " ").title()

    def get_figure(self, routine, method, qubit):
        """Get html figure for report.

        Args:
            routine (Callable): Calibration method.
            method (Callable): Plot method.
            qubit (int): Qubit id.
        """
        import tempfile

        figure = method(self.path, routine.__name__, qubit, self.format)
        with tempfile.NamedTemporaryFile() as temp:
            figure.write_html(temp.name, include_plotlyjs=False, full_html=False)
            fightml = temp.read().decode("utf-8")
        return fightml

    def get_live_figure(self, routine, method, qubit):
        """Get url to dash page for live plotting.

        This url is used by :meth:`qibocal.web.app.get_graph`.

        Args:
            routine (Callable): Calibration method.
            method (Callable): Plot method.
            qubit (int): Qubit id.
        """
        return os.path.join(
            method.__name__,
            self.path,
            routine.__name__,
            str(qubit),
            self.format,
        )
