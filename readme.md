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

## Why This Project Matters
This project mirrors **real-world e-commerce ML systems**, emphasizing:
- Business-driven metrics
- Production constraints
- Interpretability
- Profit-aware decision making

---


## Phase 4 — Evaluation Framing & Metric Strategy (Completed)

### Why Metrics Matter for Cart Abandonment
Cart abandonment during a **new product launch** is asymmetric:

- Missing a true abandoner → **direct revenue loss**
- Flagging a non-abandoner → **minor incentive cost**

Because these errors have unequal business impact, **accuracy alone is misleading**.

---

### Business-Aligned Metric Framing

**Key Outcomes**
- Correctly flagged abandoner → potential sale recovery
- Incorrectly flagged user → small margin impact
- Missed abandoner → lost customer and revenue

**Metric Strategy**
- Emphasize **recall** to protect revenue
- Maintain **precision** to control incentive waste
- Use **F1-score** to enforce balanced, cost-aware decisions

**Target Ranges**
- Recall: 75–85%
- Precision: 65–75%
- F1-score: 0.75–0.85

This framing ensures the model favors revenue protection without unchecked discounting.

---

## Phase 5 — Correlation Analysis & Signal Validation (Completed)

### Purpose
Correlation analysis was used to:
- Validate feature signal quality
- Detect dominance or leakage
- Guide model selection
- Avoid demographic over-reliance

---

### Key Insights

**1. No Dominant Features**
- All features show weak individual correlations  
- Abandonment is driven by **interacting signals**, not single rules

**2. Behavior > Demographics**
- Behavioral and psychographic signals carry more value
- Demographic features contribute minimal standalone signal

**3. Engagement ≠ Conversion for New Products**
- Loyalty and past engagement show weak relevance
- Familiarity with existing products does not guarantee adoption of new ones

---

### Modeling Implications (Pre-Training)

**Feature Strategy**
- Retain weak predictors for interaction learning
- Avoid correlation-based feature elimination
- Preserve aggregated features for interpretability

**Model Direction**
- Tree-based ensembles preferred for non-linear interactions
- Neural networks considered for weak-signal composition
- Linear models deemed insufficient

---

📌 *These phases establish evaluation rigor and signal validity **before training**, mirroring real-world ML system design and preventing metric-driven bias.*
