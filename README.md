# Qibocal

This package provides Quantum Characterization Validation and Verification protocols using [Qibo](https://github.com/qiboteam/qibo) and [Qibolab](https://github.com/qiboteam/qibolab).

## Installation

The package can be installed by source:
```sh
git clone https://github.com/qiboteam/qibocal.git
cd qibocal
pip install .
```


### Developer instructions
For development make sure to install the package using [`poetry`](https://python-poetry.org/) and to install the pre-commit hooks:
```sh
git clone https://github.com/qiboteam/qibocal.git
cd qibocal
poetry install
pre-commit install
```

When installing qibocal poetry will also install [Qibolab](https://github.com/qiboteam/qibolab). Make sure to setup SSH authentication for your GitHub account
to avoid errors during installation. Here are the instructions on how to [generate](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) a new SSH key and to [add](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) it to your GitHub account.

If you are looking to test new features in Qibolab make sure to reinstall Qibolab in the same environment where qibocal is installed.

```sh
git clone git@github.com:qiboteam/qibolab.git
cd qibolab
git checkout <your_branch>
pip install -e .[tiiq]
```



## Minimal working example
The command for executing calibration routines is the following:
```sh
qq <runcard>
```
where:
- `<runcard>`: yaml file containing the calibration routines to be performed. For more information see the documentation or the runcard examples in the `runcards` folder.

### Uploading reports to server

In order to upload the report to a centralized server, send to the server administrators your public ssh key (from the machine(s) you are planning to upload the report) and then use the `qq-upload <output_folder>` command. This program will upload your report to the server and generate an unique URL.
