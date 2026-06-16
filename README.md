# Incubation Portfolio Pipeline
[![DOI](https://zenodo.org/badge/1251479660.svg)](https://doi.org/10.5281/zenodo.20720293)

**A predictive modeling and multiobjective optimization toolkit for evidence-based incubation policy evaluation**

This repository contains the full replication code for the paper:

> *Designing Incubation Portfolios Aligned with Public Policy Objectives: Predictive Modeling and Multiobjective Optimization from a Brazilian University Incubator*

---

## Overview

University incubators face a dual mandate: cultivating high-potential ventures and demonstrating measurable impact against government-mandated performance frameworks (e.g., Brazil's CERNE accreditation system). This pipeline operationalizes that mandate as a data-driven decision-support tool.

It combines:

- **ElasticNet-penalized logistic regression** (saga solver, with l1_ratio tuned per outcome via nested cross-validation) with leave-one-out cross-validation (LOOCV) to predict five post-graduation outcomes from ex-ante venture attributes
- **Permutation testing** (Phipson–Smyth formula) to assess statistical significance of LOOCV metrics
- **Calibration assessment** (reliability curves, Brier decomposition, isotonic post-calibration) to evaluate probabilistic quality
- **Average Marginal Effects (AMEs)** with pairs-bootstrap confidence intervals to quantify each attribute's causal-descriptive influence on outcomes
- **ε-constraint MILP optimization** to trace Pareto frontiers over policy-relevant objectives under capacity constraints
- **Monte Carlo simulation** to propagate statistical uncertainty into portfolio selection

The result is a **replicable, policy-oriented framework** that any incubator or government evaluator can adapt to their own cohort data and priority objectives.

---

## Repository Structure

```
.
├── data/
│   ├── venture_cohort.csv             # Anonymized graduate cohort data (required)
│   └── venture_cohort_dictionary.json # Variable definitions and coding rules
├── incubation_portfolio_pipeline_elasticnet.ipynb  # Main analysis notebook (ElasticNet pipeline)
├── outputs_elasticnet/               # Generated tables/figures from the ElasticNet notebook (gitignored)
├── requirements.txt                 # Python dependencies
├── README.md
└── .gitignore
```

---

## Requirements

- Python 3.9+
- See `requirements.txt` for all dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Data Format

Place your cohort data at `data/venture_cohort.csv`. The pipeline expects a CSV with the following columns (adapt `OUTCOME_SPECS` in the notebook to match your variable names):

**Outcomes** (binary 0/1):

| Column | Description |
|---|---|
| `ip` | Intellectual property registered (patent/software) during or after graduation |
| `hub` | Post-graduation affiliation with a hub, science/technology park, or innovation space |
| `rev_high` | High-revenue bracket (annual turnover ≥ R$5M–R$10M) |
| `team_growth` | Team headcount bracket grew after graduation |
| `stage_growth` | Technology maturity stage advanced after graduation |

**Predictors**:

| Column | Type | Description |
|---|---|---|
| `tech_digital` | binary (0/1) | Tech domain: 1 = digital/platform/Industry-4.0; 0 = bio/agri |
| `founders` | integer | Number of founding partners at inception |
| `incub_years` | numeric | Incubation duration in years |
| `stage_in` | ordinal (1–4) | Technology maturity at entry: 1=Ideation, 2=MVP, 3=Business, 4=Scale-up |
| `team_size_in` | binary (0/1) | Entry team size: 0 = fewer than 5; 1 = 5–10 employees |
| `stage_adv` | binary (0/1) | Advanced stage at entry: `stage_in` > 2 (derived from `stage_in`) |

See `data/venture_cohort_dictionary.json` for full variable definitions and coding rules.

> **Privacy note**: The original dataset is not included in this repository to protect firm confidentiality. The pipeline has been validated on an anonymized cohort of graduated ventures from a Brazilian agricultural sciences university incubator.

---

## Usage

Open `incubation_portfolio_pipeline_elasticnet.ipynb` in Jupyter and run sections top-to-bottom. The notebook is organized as follows:

| Section | Content |
|---|---|
| 0 | Imports and configuration |
| 1 | Data loading and preprocessing |
| 2 | Helper functions (design matrix, LOOCV, AMEs, MILP) |
| 3 | Predictive modeling per outcome (LOOCV + permutation tests) |
| 4 | Calibration assessment |
| 5 | Average Marginal Effects (AMEs) |
| 6 | Portfolio optimization (ε-constraint sweep) |
| 7 | Monte Carlo uncertainty propagation |
| 8 | Results summary and export |

### Key configuration (Section 0)

```python
DATA_PATH   = Path("data/venture_cohort.csv")
B_PERM      = 2_000     # permutation draws
R_BOOT      = 2_000     # bootstrap replicates for AMEs
MC_DRAWS    = 2_000     # Monte Carlo draws
CAPACITY    = 8         # portfolio size (K)
ALPHAS      = [0.4, 0.6, 0.8, 1.0]        # Expand-vs-Admit gain scaling
EPS_VEC     = np.linspace(0.05, 0.95, 19)  # ε-floor sweep grid

USE_EVIDENCE_WEIGHTS = False  # set True to apply permutation-p-weighted objectives
```

Adapt `OUTCOME_SPECS` to your variable names and feature sets. Each entry now also configures the ElasticNet `l1_ratio` grid, optional interaction terms, and whether Ridge-based pre-screening is applied:

```python
OUTCOME_SPECS = {
    "ip":           {"bin_cols": ["tech_digital", "team_size_in"], "num_cols": ["founders", "incub_years"],
                      "cat_cols": ["stage_in"], "interact_num_cat": [...], "interact_num_bin": [...],
                      "l1_ratio_grid": [0.5], "prescreening": True},
    "hub":          {"bin_cols": [], "num_cols": ["incub_years"], "cat_cols": ["stage_in"],
                      "interact_num_cat": [("incub_years", "stage_in")],
                      "l1_ratio_grid": [0.1], "prescreening": False},
    "rev_high":     {"bin_cols": ["team_size_in"], "num_cols": ["incub_years"], "cat_cols": [],
                      "interact_num_bin": [("incub_years", "team_size_in")],
                      "l1_ratio_grid": [0.1], "prescreening": False},
    "team_growth":  {"bin_cols": ["tech_digital", "stage_adv"], "num_cols": [], "cat_cols": [],
                      "l1_ratio_grid": [0.1], "prescreening": False},
    "stage_growth": {"bin_cols": ["stage_adv", "team_size_in"], "num_cols": [], "cat_cols": [],
                      "l1_ratio_grid": [0.1], "prescreening": False},
}
```

See Section 0/1 of `incubation_portfolio_pipeline_elasticnet.ipynb` for the full specs (including all interaction terms) and the rationale behind each curated feature set.

---

## Policy Evaluation Context

The five outcome variables map directly to performance dimensions tracked by Brazil's **CERNE** (Reference Centre for Supporting New Enterprises) accreditation framework — the national quality standard for incubators administered by Anprotec/SEBRAE. They capture:

- **Technology transfer** (`ip`): alignment with innovation mandates under Brazil's Lei de Inovação
- **Ecosystem integration** (`hub`): linkage to public S&T infrastructure
- **Economic performance** (`rev_high`): revenue and commercialization thresholds
- **Employment generation** (`team_growth`): sustainable headcount growth
- **Technological advancement** (`stage_growth`): progress along the technology readiness ladder

The ε-constraint Pareto sweep allows policy evaluators to visualize trade-offs between these dimensions and identify portfolio configurations that satisfy multiple government-mandated targets simultaneously.

---

## Adapting to Other Contexts

This pipeline is designed to be context-portable. To apply it to a different incubator or national system:

1. Replace `venture_cohort.csv` with your cohort data, using the same column schema
2. Update `OUTCOME_SPECS` to match your variable names and predictor sets
3. Adjust `CAPACITY` to reflect your intake cohort size
4. Relabel the five objectives in Section 6 to match your policy framework's KPIs

---

## Citation

This repository accompanies a manuscript currently in press. Full citation details will be added once the paper is published.

---

## License

MIT License. See `LICENSE` for details.
