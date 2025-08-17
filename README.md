# Fantasy Sports Machine Learning Analytics

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Example Workflow](#example-workflow)
- [What I Learned](#what-i-learned)
- [Overall Impact](#overall-impact)
- [Contact](#contact)

## Overview

This project automates the collection, processing, analysis, and predictive modeling of fantasy football league data from Yahoo Fantasy Sports across multiple seasons (2017–2025). It integrates data engineering, custom analytics, and machine learning to build a reproducible, extensible pipeline that:

- Eliminates manual data collection by automating multi-season API retrieval
- Produces normalized, human-readable datasets for matchups, standings, and statistics
- Generates custom domain metrics (LuckIndex, Weighted Adjusted Averages, EWMA trends)
- Supports dynamic playoff ranking logic across various bracket sizes
- Delivers machine learning power ratings and predictions for remaining wins using feature engineering, Random Forest regression, and permutation importance analysis

## Key Features

- **End-to-End Data Pipeline** — From raw Yahoo API calls to structured CSVs and analytics outputs
- **Multi-Year Historical Tracking** — Maintains full-season data from 2017 onward, with scalable extension for future years
- **Automated Name/ID Resolution** — Resolves inconsistencies in team names and owner GUIDs across seasons
- **Advanced Metrics** — LuckIndex, weighted averages, volatility, streak tracking, and more
- **Dynamic Playoff Calculations** — Accurate ranking for 4, 6, 7, and 8-team brackets
- **ML Predictive Analytics** — Random Forest and XGBoost models to forecast remaining wins, with interpretable feature importance visualizations
- **Reproducible & Configurable** — Centralized configuration, OAuth authentication, and modular architecture for maintainability

## Technologies Used

### Languages & Core Libraries
- Python 3
- csv, os, sys, statistics, collections

### API Integration
- yahoo_oauth — Secure OAuth2 authentication
- yahoo_fantasy_api — Simplified Yahoo Fantasy API client

### Data Analysis & Visualization
- pandas, numpy
- Matplotlib, Seaborn

### Machine Learning
- scikit-learn (RandomForestRegressor, LinearRegression, RandomizedSearchCV, permutation_importance)
- XGBoost (XGBRegressor)

### Reproducibility & Performance
- Fixed random seeds for reproducibility
- Parallel training (n_jobs=-1)

## Installation & Setup

### 1. Clone Repository
``` bash
git clone https://github.com/yourusername/Fantasy-Sports-ML-Analytics.git
cd Fantasy-Sports-ML-Analytics
```

### 2. (Optional But Recommended) Create a Virtual Environment

Create and activate a virtual environment to isolate dependencies.

```bash
# Create the virtual environment
python -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Requirements
``` bash
pip install -r requirements.txt
```

### 4. Configure API Authentication

Obtain Yahoo API credentials and create `oauth2.json` in the project root.

File must contain the consumer key, secret, and tokens per `yahoo_oauth` format.

### 5. Update Configurations

`src/config.py` contains constants like `LEAGUE_IDS` and `MAX_WEEKS_BY_YEAR`.

Add or update league IDs as needed.

## Project Structure
``` bash
src/
  authenticate.py         # Initializes and refreshes OAuth2 tokens for Yahoo API access
  authenticate.py         # Initializes and refreshes OAuth2 tokens for Yahoo API access
  utils.py                # League/owner metadata automation
  matchup_utils.py        # Name normalization, result determination, data extraction
  collect_matchups.py     # Full-season matchup collection
  update_matchups.py      # Incremental weekly matchup updates
  standings_utils.py      # Win %, playoff ranking logic
  generate_standings.py   # Season/all-time standings generation
  generate_stats.py       # Core & advanced statistics (LuckIndex, EWMA, etc.)
  regression_utils.py     # Scaling and regression helpers
  ml_plots.py             # Functions to visualize regression modeling output
  ml_plots.py             # Functions to visualize regression modeling output
  ml_regression.py        # Predictive modeling pipeline
data/
  matchup_data.csv        # Master matchup dataset
standings/                # Standings directory
  <year>/                 # One folder per season
    regular.csv
    raw.csv
    final.csv
  all_time.csv            # All-time standings
statistics/               # Statistics directory
  <year>/                 # One folder per season
    stats.csv
regression/               # Predictive modeling results/visualizations
  regression_results.csv  # Regression model results
  figures/                # Regression model visualizations
    <figures>
.gitignore                # Lists files/folders Git should ignore
requirements.txt          # Lists Python packages and versions needed to run the project
regression/               # Predictive modeling results/visualizations
  regression_results.csv  # Regression model results
  figures/                # Regression model visualizations
    <figures>
.gitignore                # Lists files/folders Git should ignore
requirements.txt          # Lists Python packages and versions needed to run the project
  ```

## Data Pipeline

### 1. Metadata & Owner Resolution — utils.py
Automates retrieval of league IDs and owner GUIDs from Yahoo, across all seasons.  
**Outputs:**  
- `LEAGUE_IDS` mapping (season → ID)  
- `valid_owner_map.csv` and `manual_owner_map.csv`  

**Key features:**  
- Secure OAuth2 authentication  
- Robust error handling  
- Scalable for new seasons with config updates  

---

### 2. Name Normalization & Team Data Extraction — matchup_utils.py
Centralized helper functions to:  
- Map GUIDs/team names → consistent display names  
- Calculate matchup results ("Win", "Loss", "Tie", "N/A")  
- Extract actual & projected scores from API responses  

*Ensures data consistency across the entire project.*  

---

### 3. Full-Season Matchup Collection — collect_matchups.py
Retrieves all matchups for all weeks/seasons defined in config.  
- Owner names resolved with `matchup_utils`  
- Playoff weeks flagged automatically  
- Normalized CSV structure for downstream processing  

---

### 4. Incremental Weekly Updates — update_matchups.py
Targets a single week in a specific season:  
- Updates scores and results without overwriting other data  
- CLI-based input validation for safety  
- Supports in-season maintenance  

---

### 5. Standings Calculation Utilities — standings_utils.py
Functions to compute:  
- Win % with tie adjustment  
- Playoff rankings for 4, 6, 7, or 8-team brackets  
- Handles consolation games and dynamic playoff weeks  

---

### 6. Season & All-Time Standings — generate_standings.py
Produces:  
- Regular season standings  
- Raw (full season) standings  
- Playoff-adjusted final standings  

*All outputs saved to CSV for reporting and visualization.*  

---

### 7. Team Statistics Generation — generate_stats.py
Calculates:  
- GP, Avg, StDev, High/Low, Streak  
- `LuckIndex` — performance vs. expectation  
- `Weighted Adjusted Average` — recent performance emphasis  
- `EWMA` — smoothed performance trends  

*Outputs per-season stats CSVs.*  

---

### 8. Machine Learning Regression — ml_regression.py, ml_plots.py, regression_utils.py
Predicts remaining regular-season wins using:  
- Random Forest (primary) & XGBoost (experimental)  
- Leave-one-season-out cross-validation  
- Extensive feature engineering (core + interaction terms)  
- Feature scaling (z-score → min-max)
- Hyperparameter tuning (RandomizedSearchCV)

**Automated Workflow:**
- Saves regression results to `regression/regression_results.csv`  
- Generates and stores visualizations in `regression/figures/`:
  - Model performance across weeks  
  - Feature usage matrix  
  - Average and top-feature permutation importance  
  - Feature stability across thresholds  

All outputs are reproducible, organized, and ready for analysis. 

---

## Example Workflow
``` bash
# 1. Collect historical matchups
python src/collect_matchups.py

# 2. Update latest week's matchups
python src/update_matchups.py <year> <week>

# 3. Generate season standings
python src/generate_standings.py <year>

# 4. Produce statistics
python src/generate_stats.py <year>

# 5. Run machine learning regression
python src/ml_regression.py
```

## What I Learned

Building this project required working across the full data engineering and data science stack, and each stage introduced its own technical and strategic challenges. Key takeaways include:

### API Integration & Automation
- Learned how to work with OAuth2 authentication and the Yahoo Fantasy Sports API, including edge-case handling for historical season data.
- Gained experience in structuring reusable, modular scripts to make API workflows repeatable and easy to maintain.
- Gained experience in structuring reusable, modular scripts to make API workflows repeatable and easy to maintain.

### Data Cleaning & Normalization at Scale
- Developed practical strategies for resolving entity resolution issues (e.g., team/owner name changes, missing GUIDs).
- Built centralized mapping utilities that ensure consistency across years without repetitive code.

### Feature Engineering & Advanced Metrics
- Deepened understanding of designing domain-specific sports metrics like LuckIndex, weighted averages, volatility measures, and exponentially weighted moving averages (EWMA).
- Learned to combine domain knowledge with statistical principles to improve predictive features.
- Developed strategies to identify and remove uninformative/misleading features, and experimented with feature interactions to enhance model performance.
- Developed strategies to identify and remove uninformative/misleading features, and experimented with feature interactions to enhance model performance.

### Predictive Modeling Best Practices
- Applied leave-one-season-out cross-validation to realistically evaluate time-series generalization.
- Balanced model complexity with small dataset constraints to avoid overfitting.
- Used permutation importance for more reliable feature interpretability compared to impurity-based methods.

### Hyperparameter Tuning
- Explored randomized search and cross-validation (RandomizedSearchCV) to optimize model hyperparameters.
- Learned to balance model complexity and performance, improving generalization across seasons.

### Reproducibility & Maintainability
- Implemented reproducibility safeguards via fixed random seeds and parallel execution.
- Designed the project architecture so new seasons, metrics, or models can be integrated with minimal changes.

### Automation & Visualization
- Gained experience in automating end-to-end ML workflows, including storing results and figures in structured folders.  
- Learned to design pipelines that integrate model training, evaluation, and visualization with minimal manual intervention.  
- Improved ability to communicate complex model insights through well-organized plots and summary outputs.

## Overall Impact
This project strengthened my ability to connect data engineering and machine learning in a complete workflow, treating data as a product that delivers reliable, interpretable results. I gained confidence in cleaning complex datasets, creating domain-informed metrics, building accurate, explainable models, and communicating insights effectively through visuals and narrative. Overall, this experience significantly enhanced my problem-solving, critical thinking, and ability to manage end-to-end data projects independently.

## Contact

For questions, feedback, or collaboration opportunities, feel free to reach out:

- **Email:** abhi.salitri@gmail.com  
- **LinkedIn:** [linkedin.com/in/abhishek-salitri](https://www.linkedin.com/in/abhishek-salitri)  

I’m happy to connect with fellow sports enthusiasts, data scientists, or anyone interested in machine learning and analytics!
