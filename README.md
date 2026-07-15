# Data Science Internship: 8-Week Progressive Project

## Overview
This repository tracks the progression of an 8-week Data Science internship. Instead of treating tasks as isolated scripts, this project is architected as an evolving, professional-grade software pipeline. The work progresses sequentially from raw data ingestion and exploratory data analysis (EDA), through statistical analysis, feature engineering, machine learning, and finally, deployment.

Currently, the pipeline addresses two foundational datasets:
1. **Titanic Dataset** (Classification)
2. **House Prices Dataset** (Regression)

## Project Progress
- **Weeks 1-2 (Completed):** Repository architecture setup, custom data profiling, structural cleaning (handling missing data while preventing leakage), and comprehensive EDA using a custom Data Journalism visualization API. 
  - ➡️ **[Read the detailed Week 1-2 Report here](reports/Week_1_2_README.md)**

## Repository Architecture
The repository follows a domain-driven notebook structure combined with a centralized, modular source code folder to ensure code reusability and clean architecture.

```text
analystlab-internship/
├── README.md               # Top-level executive summary
├── datasets/
│   ├── raw/                # Immutable original datasets (Do not edit manually)
│   └── processed/          # Cleaned, model-ready datasets 
├── notebooks/              # Domain-organized Jupyter notebooks
│   ├── titanic/            # Classification pipeline
│   │   ├── 01_data_profiling.ipynb
│   │   ├── 02_data_cleaning.ipynb
│   │   ├── 03_exploratory_analysis.ipynb
│   │   └── 04_summary.ipynb
│   └── housing/            # Regression pipeline
│       ├── 01_data_profiling.ipynb
│       ├── 02_data_cleaning.ipynb
│       ├── 03_exploratory_analysis.ipynb
│       └── 04_summary.ipynb
├── reports/                # Weekly logs, design decisions, and findings
├── figures/                # Exported visual assets and plots
├── src/                    # Reusable Python modules (DRY principle)
│   ├── data/               # Scripts for loading and saving data
│   ├── visualization/      # Standardized plotting functions
│   ├── utils/              # Helper functions (e.g., custom profilers)
│   └── config.py           # Global configuration and path management
└── requirements.txt        # Environment dependencies
```

## Workflow Strategy
- **Notebooks:** Kept lightweight. They are used for narration, visualization, and decision-making. 
- **Src Module:** Any code that is reused across multiple notebooks (like a custom data profiler or missing-value imputer) is abstracted into the `src/` directory.
- **Reports:** To avoid monolithic documentation, detailed design decisions and weekly progress summaries are maintained in the `reports/` directory.

## Setup Instructions
1. Clone the repository.
2. Install dependencies via: `pip install -r requirements.txt`
3. Ensure the raw data files are placed within `datasets/raw/`.
```