# CivicCare — Online Complaint Management and Sentiment Analysis System

BCA final-year academic prototype for a three-member group: Shiladitya Barua, Anup Das and Suromya Ghosh Dastidar.

## Features
- Customer registration/login, complaint submission and unique reference IDs
- Public reference-based complaint tracking
- Separate staff and administrator roles
- Staff complaint queue, admin assignment, status updates, response notes and category correction
- Separate TF-IDF complaint-category and sentiment classifiers
- Comparison of Multinomial Naive Bayes, Logistic Regression and Linear SVM
- Analytics dashboard with live complaint KPIs, monthly trends, department performance, category and sentiment charts
- Historical CSV analysis separated from live complaint records
- Model evaluation dashboard with algorithm comparison, classification report and confusion matrix
- SQLite persistence, password hashing and basic role-based access controls

## Important academic data note
The bundled NLP examples are **synthetic demonstration examples** used to make the prototype runnable immediately. They are not a real-world complaint dataset and must not be presented as representative research data. Before final model evaluation, select and document a suitable public dataset, verify its licence/access terms, inspect fields and class balance, anonymise personal information, and create a leakage-safe train/test split. The handbook suggests the Kaggle CFPB Consumer Complaints resource, but access/participation conditions must be checked. Do not treat category labels as sentiment labels; annotate a separate sample or source a separate labelled sentiment dataset.

The metrics in `models/training_results.json` are only a software demonstration of the training pipeline on synthetic examples. They are not valid claims of real-world model performance.

## Requirements
- Python 3.10 or newer recommended
- Internet connection for Bootstrap/Chart.js CDN assets (or download assets locally for offline demo)

## Setup (Windows / macOS / Linux)
```bash
cd complaint-management-system
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

The application loads the model files in `models/` and creates `complaints.db` if it does not exist. It does not automatically retrain or overwrite models on startup.

## Demo accounts
- Admin: `admin@demo.local` / `Admin@12345`
- Staff: `staff@demo.local` / `Staff@12345`
- Customer: register through the website

Change demo passwords and set a strong `SECRET_KEY` before any deployment. This is an academic prototype, not a production public grievance service. Use HTTPS, CSRF protection, rate limiting, email verification, audit controls, backups and a privacy review before real deployment. Never upload real sensitive complaints to a public demo.

## Suggested team division
- **Shiladitya Barua:** Flask backend, database schema, authentication/roles and integration
- **Anup Das:** dataset sourcing, cleaning, EDA, category/sentiment model training and evaluation
- **Suromya Ghosh Dastidar:** frontend, dashboard charts, test cases, screenshots and report/presentation coordination

All members should understand the complete workflow and be able to explain the model, database and limitations.

## Suggested next milestones
1. Confirm the three-member group is accepted by the department (the handbook's general requirement says 4–5 members).
2. Select a permitted complaint dataset and document exact title, URL, licence, fields and sample count.
3. Replace synthetic examples with real labelled training data; keep category and sentiment targets separate.
4. Add EDA, model evaluation tables, confusion matrices and error analysis to the report.
5. Add CSRF protection, password reset, stronger validation and deployment configuration before public hosting.
6. Prepare use-case, architecture and ER diagrams, test evidence, final report and live demonstration.

## Analytics dashboard notes
- **Live operational analytics** are calculated from the CivicCare SQLite database: complaint totals, status distribution, monthly submissions, department resolution rates and average time to resolution.
- **Historical dataset analysis** is stored in `models/dataset_analysis.json` and describes the supplied `data/complaints_processed.csv` dataset. The supplied CSV has `product` and `narrative` columns; it has no date column, so it cannot support historical monthly trend analysis.
- **Category model evaluation** is read from `models/category_training_results.json`. Scores are calculated on a held-out test split before the selected model is refitted on the full labelled dataset.
- **Sentiment predictions are experimental** because the bundled sentiment model was trained on synthetic demonstration examples. Do not use its dashboard distribution as validated public sentiment research.
- Historical dataset rows are not imported as live citizen complaints.


## CSV upload and staff queue behavior

When an administrator uploads a labelled CSV (`product`, `narrative`), CivicCare evaluates/retrains the category model, saves historical category and inferred-sentiment analytics, and imports each valid row as an unassigned `Submitted` complaint for the staff queue. Imported rows use a dedicated system customer and are tagged `dataset_import`; staff can review, assign, update, and resolve them. The staff queue is paginated (50 rows per page). Re-uploading the same file skips rows with matching deterministic references. Since the source dataset has no date field, imported tickets use the import time, not an asserted original complaint date. Sentiment analytics are model-inferred and experimental because the CSV has no sentiment labels.

**Operational caution:** importing a large dataset creates a large number of active staff tickets. Upload only when you intend those historical records to be processed as work items.
