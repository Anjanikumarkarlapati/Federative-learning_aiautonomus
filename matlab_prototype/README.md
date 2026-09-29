# MATLAB prototype

Same algorithm and data model as `../python_prototype`, in base MATLAB (no toolboxes).

    cd matlab_prototype
    results = fl_prototype(20, 30, 1);

Prints final accuracy for centralized / FedAvg / proposed, writes
`results/rounds.csv`, and saves `results/accuracy.png` when a desktop is available.
Random streams differ from the Python version, so numbers match only qualitatively.

Note: written without MATLAB available in the authoring environment, so it has
not been executed; please report any error on first run.
