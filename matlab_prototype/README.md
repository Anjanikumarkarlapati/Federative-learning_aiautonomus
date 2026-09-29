# MATLAB prototype

Same algorithm, data model, arms and outputs as `../python_prototype`, in base MATLAB
(no toolboxes; R2016b or newer, GNU Octave untested).

```matlab
cd matlab_prototype
fl_prototype                                   % defaults: 20 vehicles, 30 rounds, 3 seeds
r = fl_prototype('clients',40,'rounds',50,'repeats',10);
fl_prototype('verbose',false,'plot',false)     % summary only
```

Options: `clients`(20) `rounds`(30) `classes`(6) `dim`(16) `alpha`(0.4) `frac`(0.3)
`seed`(1) `repeats`(3) `verbose`(true) `plot`(true).

See `python_prototype/README.md` for the idea, what is simulated, and how to read the
results. MATLAB-specific outputs:

- Console: fleet table, per-round log, results table.
- `results/rounds.csv`, `results/summary.csv`, and `results/accuracy.png` (convergence and
  cumulative-upload panels, when a desktop is available).
- Return value `r`: struct with `arms`, `mean`, `std` (arms x [acc loss worst r90 KB dropped]) and `history` (per-round matrices `[round loss acc participants dropped KB]`).

Random streams differ from Python, so numbers agree only qualitatively.

**Untested:** written without MATLAB available in the authoring environment; please report
any error on first run.
