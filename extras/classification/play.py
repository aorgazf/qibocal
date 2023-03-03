from pathlib import Path

from qibocal.fitting.classifier import (
    linear_svm,
    naive_bayes,
    plots,
    random_forest,
    rbf_svm,
    run,
)

data_path = Path("calibrate_qubit_states/data.csv")
base_dir = Path("_results")
base_dir.mkdir()
qubit = 1
qubit_dir = base_dir / f"qubit{qubit}"
classifiers = [linear_svm]
table, y_test = run.train_qubit(data_path, base_dir, qubit)
run.dump_benchmarks_table(table, qubit_dir)
plots.plot_table(table, qubit_dir)
plots.plot_conf_matr(y_test, qubit_dir)
plots.plot_roc_curves(y_test, qubit_dir)
