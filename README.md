# FairMetrics
This repository is the official implementation of the paper **"Rethinking Fairness Metrics for Multi-Class and Multi-Group Assessment: Not Everything is Binary."**

**Authors:**
- **Matteo Grazioso**¹*† ([matteo.grazioso@unive.it](mailto:matteo.grazioso@unive.it))
- **Fabio Naccarella**¹† ([900619@stud.unive.it](mailto:900619@stud.unive.it))
- **Chiara Gallese**² ([C.Gallese@tilburguniversity.edu](mailto:C.Gallese@tilburguniversity.edu))
- **Marco S. Nobile**¹ ³ ([marco.nobile@unive.it](mailto:marco.nobile@unive.it))

¹ *Department of Environmental Sciences, Informatics and Statistics, Ca’ Foscari University of Venice, Via Torino 155, Venezia Mestre, 30172, Italy*  
² *Tilburg Institute for Law, Technology, and Society, Tilburg University, Prof. Cobbenhagenlaan 221, Tilburg, 5037 DB, The Netherlands*  
³ *European Center for Living Technologies (ECLT), Venice, Italy*  

*\*Corresponding author*  
*†Equal contribution*

## Layout

```text
fairmetrics/
├── __init__.py
├── _utils.py
├── metrics.py
├── tests.py
├── independence.py
├── separation.py
├── sufficiency.py
└── accuracy.py

fairmetrics_experiments/
├── __init__.py
├── binarization.py
└── run_hpo.py

scripts/
└── multi_run.sh
```

The split is organized around:

- `fairmetrics/metrics.py`: standard fairness metrics
- `fairmetrics/tests.py`: notebook-style data/HPO/test helpers
- `fairmetrics_experiments/run_hpo.py`: paper-style HPO runner (`adult`/`students`, `mul`/`bin*`) + fairness audit

## Installation

```bash
pip install -e .
```

Optional experiment dependencies:

```bash
pip install -e '.[experiments]'
```

## Modes used in paper runs

The experiment runner supports four target strategies:

- `mul`
  - keep the original multiclass target as-is

- `bin0`
  - one-vs-rest binarization using class index 0 as positive
  - mapping: `y == class_0 -> 1`, else `0`

- `bin1`
  - one-vs-rest binarization using class index 1 as positive
  - mapping: `y == class_1 -> 1`, else `0`

- `bin2`
  - one-vs-rest binarization using class index 2 as positive
  - mapping: `y == class_2 -> 1`, else `0`

In short:
- `mul` = multiclass experiment
- `bin*` = binary one-vs-rest experiment for the selected class index

## Fairness configuration used in runs

The runner evaluates fairness with **intersectional sensitive attributes** and conditional strata:

- **Adult**
  - sensitive: `['Sex', 'Race_White']`
  - strata: `['Country_United-States']`

- **Students**
  - sensitive: `['Gender', 'International']`
  - strata: `['Displaced']`

The output JSON includes:
- HPO result (`best_params`, `best_score`)
- fairness metrics:
  - `demographic_parity`
  - `conditional_demographic_parity`
  - `equal_opportunity`
  - `predictive_equality`
  - `equalized_odds`
  - `predictive_parity`
  - `equal_accuracy`
- `intersectional_sensitive_groups`

## Single-run commands (paper style)

```bash
python -m fairmetrics_experiments.run_hpo --dataset adult --mode mul
python -m fairmetrics_experiments.run_hpo --dataset adult --mode bin0
python -m fairmetrics_experiments.run_hpo --dataset adult --mode bin1
python -m fairmetrics_experiments.run_hpo --dataset adult --mode bin2

python -m fairmetrics_experiments.run_hpo --dataset students --mode mul
python -m fairmetrics_experiments.run_hpo --dataset students --mode bin0
python -m fairmetrics_experiments.run_hpo --dataset students --mode bin1
python -m fairmetrics_experiments.run_hpo --dataset students --mode bin2
```

Save JSON output:

```bash
python -m fairmetrics_experiments.run_hpo --dataset adult --mode bin0 --output results/adult_bin0.json
```

## Batch execution
Run the full set of strategies:

```bash
bash scripts/multi_run.sh
```

It runs:

- Adult: `mul`, `bin0`, `bin1`, `bin2`
- Students: `mul`, `bin0`, `bin1`, `bin2`

and stores outputs in `results/`, e.g.:

- `results/adult_mul.json`
- `results/adult_bin0.json`
- `results/adult_bin1.json`
- `results/adult_bin2.json`
- `results/students_mul.json`
- `results/students_bin0.json`
- `results/students_bin1.json`
- `results/students_bin2.json`

## FRACTALS Research Group

**FRACTALS**  
*Fair and Responsible Algorithms for Complex-systems, Therapeutics, Astrophysics and Life-Sciences*

🔗 https://fractals.group
