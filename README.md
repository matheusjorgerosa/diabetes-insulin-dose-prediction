# Personalized meal-insulin dosing in Type 1 Diabetes

Course project for Machine Learning for Healthcare (Technion): predict the manual bolus dose for a meal from the patient's profile, how their body has responded to insulin on earlier meals, and what they ate and did in the previous hours.

| File | Content |
|---|---|
| `01_preprocessing.ipynb` | Downloads AZT1D, cleans it and builds one row per meal episode |
| `02_linear_regression.ipynb` | Linear regression of the meal bolus: per-patient time split, baselines, evaluation, residuals, coefficients |
| `plano-do-projeto.md` | Project plan and log (in Portuguese): datasets evaluated, decisions, next steps |
| `data/processed/azt1d_meal_episodes.csv` | Meal-episode table produced by notebook 1 and used by notebook 2 |

## Data

The raw data are not in this repository. `01_preprocessing.ipynb` downloads them (~776 MB) into `data/raw/` the first time it runs.

- **AZT1D:** Khamesian S. et al., *AZT1D: A Real-World Dataset for Type 1 Diabetes*, arXiv:2506.14789 (2025). Mendeley Data, DOI [10.17632/gk9m674wcx.1](https://doi.org/10.17632/gk9m674wcx.1), CC BY 4.0.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

To run only the regression, open `02_linear_regression.ipynb`: it reads the processed CSV that is already in the repository.
