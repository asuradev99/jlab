# jlab

Physics lab data analysis project. Python scripts for curve fitting, histogram analysis, and spectrum plotting.

## Structure

- `util.py` — shared utilities: Gaussian/Poisson fit functions, chi-squared, histogram helpers
- `curve_fit/` — Gaussian curve fitting examples and tests
- `mca_plot/` — MCA spectrum plotting (e.g. uranium energy spectrum)
- `pendulum/` — pendulum experiment analysis
- `poisson/` — Poisson distribution fitting
- `initial_fit/` — initial fit experiments

## Environment

Uses a local `.venv`. Activate with:

```
source .venv/bin/activate.fish
```

Install dependencies:

```
pip install numpy matplotlib scipy pandas
```

## Notes

- `util.py` lives at the project root and is shared across subdirectories. Scripts in subdirectories insert the project root into `sys.path` to import it.
- LaTeX rendering (`usetex=True`) is used in plots — requires a working LaTeX installation.
