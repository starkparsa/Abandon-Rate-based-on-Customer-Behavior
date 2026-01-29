# Abandon Rate Based on Customer Behavior Metrics

## Problem Statement
Identify which customers are most likely to abandon their cart during a **new product launch** and determine **data-driven interventions** to reduce abandonment **without eroding profit margins**.

This project frames cart abandonment as a **binary classification problem**, using aggregated behavioral, engagement, and psychographic signals while strictly preventing target leakage.

---

## Project Objective
- Predict **cart abandonment likelihood**
- Understand **key drivers** behind abandonment behavior
- Enable **actionable recommendations** (pricing, engagement, UX, incentives)
- Build an **end-to-end, production-ready ML pipeline**

---

## Process Overview

### Phase 1 — Setup (Completed)

**Environment & Project Structure**
- Designed a **scalable folder structure** aligned with production ML workflows
- Initialized:
  - `data/` (raw, processed)
  - `src/` (data, features, models, app)
  - `configs/`, `tests/`, `scripts/`
  - `mlruns/`, `artifacts/`, `docker/`, CI/CD scaffolding

**Development Environment**
- Created and activated a **Python virtual environment**
- Installed core libraries for:
  - Data processing
  - Modeling
  - Experiment tracking
  - Validation

**Version Control**
- Initialized **Git repository**
- Configured `.gitignore` for:
  - Virtual environments
  - Artifacts
  - MLflow runs
  - Temporary data files

---

### Phase 2 — Problem Framing & Feature Strategy (Completed)

**Target Variable**
- `cart_abandonment_flag` (Binary Classification)

**Key Decisions**
- Explicitly removed **target leakage variables** (e.g., conversion rates, return rates)
- Aggregated raw behavioral metrics into **stable, interpretable indices**
- Reduced feature space to improve **generalization and inference stability**

**Final Feature Categories**
- Demographics & preferences
- Engagement intensity
- Advertising responsiveness
- Purchase intent
- Discount sensitivity
- Loyalty & advocacy
- Stress & lifestyle impact
- Shopping consistency

---

### Phase 3 — Feature Engineering (Completed)

**Highlights**
- Converted granular event-level data into:
  - Normalized scores
  - Ratios
  - Buckets
- Dropped raw columns post-aggregation to:
  - Reduce noise
  - Prevent overfitting
  - Improve SHAP interpretability

**Outcome**
- Compact, production-ready feature set (~25 features + target)

---

## Current Status
- [x] Dataset cleaned and schema finalized  
- [x] Target variable locked  
- [x] Leakage risks mitigated  
- [x] Feature engineering strategy documented  

---

## Next Steps (Planned)
- Implement **baseline classification models**
- Optimize using **balanced precision–recall objective (Optuna)**
- Address **class imbalance**
- Add **MLflow experiment tracking**
- Build **FastAPI inference endpoint**
- Create **executive-friendly explanations (SHAP)**
- Containerize and prepare for **deployment**

---

## Why This Project Matters
This project mirrors **real-world e-commerce ML systems**, emphasizing:
- Business-driven metrics
- Production constraints
- Interpretability
- Profit-aware decision making

---

📌 *Daily progress is intentionally documented to reflect an industry-style ML workflow rather than a notebook-only project.*
