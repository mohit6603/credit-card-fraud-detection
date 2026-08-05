# Beginner's Guide — This Project as an ML Life Cycle

A plain-words, step-by-step walkthrough of the whole project, with every important number.
(For the technical version, see [RESULTS.md](RESULTS.md).)

---

## Step 1 — Define the problem

A bank must decide in a split second: **is this card payment real or stolen?**

- Missing a fraud → the bank loses that money.
- Blocking a real customer → the customer gets angry, calls support, maybe leaves.

So we need a model that gives a **fraud probability (0 to 1)** for every transaction —
not just a yes/no. A probability lets the bank tune how strict to be without retraining.

## Step 2 — Collect the data

The famous Kaggle credit-card dataset:

- **284,807 rows** (transactions) × **31 columns**
- Columns: `Time`, `V1–V28` (anonymized features), `Amount`, `Class` (0 = genuine, 1 = fraud)
- Only **492 frauds** in the whole dataset = **0.172%** → for every 1 fraud there are
  **578 genuine** transactions. This imbalance is THE central problem of the project.

## Step 3 — Clean & validate the data

Automated checks found:

| Check | Result |
|---|---|
| Missing values | **0** ✅ (real data is rarely this clean) |
| Impossible values (negative amounts etc.) | **0** ✅ |
| Duplicate rows | **1,081** ❌ → deleted |

**Left over after cleaning: 283,726 rows** (473 frauds).

Key detail: duplicates were deleted **before** splitting the data. If the same row lands in
both train and test, the model "cheats" by seeing the answer in advance (data leakage).

## Step 4 — Explore the data (EDA)

What the plots told us (all saved in `reports/images/`):

- Fraud amounts are usually **small** (median ≈ $9) — thieves test cards with tiny payments.
- Fraud **rate spikes at night (~2–5 AM)** — cardholders are asleep.
- A few features (V14, V12, V10) look very different for fraud vs genuine — the signal exists.

## Step 5 — Feature engineering

- Added **`Hour`** (0–23, from `Time`) → captures the night-fraud pattern.
- Added **`Amount_log`** → squashes huge amounts so they don't dominate.
- Dropped raw `Time` — "seconds since recording started" is meaningless for future data.

Final input to the model: **30 features**.

## Step 6 — Split the data (three ways)

| Split | Rows | Frauds | Used for |
|---|---|---|---|
| Train (64%) | **181,584** | 302 | teaching the model |
| Validation (16%) | **45,396** | 76 | choosing the best model & threshold |
| Test (20%) | **56,746** | 95 | final exam — touched only ONCE at the end |

**Stratified** splitting keeps the same 0.17% fraud rate in every split. With only 473
frauds, a careless split could leave the test set with almost nothing to check against.

## Step 7 — Handle the imbalance + train models

**5 models × 4 imbalance tricks = 17 experiments** (~18 minutes of training).
The tricks: give fraud rows extra weight (`class_weight`), create synthetic fraud rows
(`SMOTE`), delete genuine rows (`undersampling`), SMOTE + boundary cleanup (`SMOTE-Tomek`).

**Winner: XGBoost + class_weight** (validation PR-AUC 0.8785).

Three honest lessons from the losers:

1. Deleting genuine rows always made things worse — you throw away 99.8% of your information.
2. Synthetic fraud rows (SMOTE) didn't beat simple weighting.
3. LightGBM completely broke under extreme weighting (scored ~0) — same trick, different
   library, different result. Never assume libraries are interchangeable.

## Step 8 — Evaluate (and the accuracy trap)

Accuracy of the final model:

| Where | Accuracy |
|---|---|
| Training | **100.000%** |
| Validation | **99.971%** |
| Test | **99.954%** |

Looks amazing — **it's a trap.** A "model" that just says *"everything is genuine"* scores
**99.83%** here while catching **zero** frauds. When 99.8% of answers are "genuine",
accuracy is nearly useless. (Also: 100% on training = the model memorized its homework;
only the test number counts.)

So we judge with fraud-focused numbers on the **test set**:

| Question | Answer |
|---|---|
| Of 95 real frauds, how many caught? | **72 → recall = 75.8%** |
| Of 75 alarms raised, how many were real fraud? | **72 → precision = 96.0%** |
| False alarms among 56,651 genuine payments | **only 3** |
| Frauds missed | 23 (they look like normal spending) |
| PR-AUC (quality score for rare-event problems) | **0.821** (random guessing = 0.002) |

## Step 9 — Pick the decision threshold

The model outputs a probability; someone must choose the cut-off for shouting "FRAUD!".
Default is 0.5 — instead we computed real money: *a missed fraud costs its amount, a false
alarm costs $5*. Cheapest cut-off on validation: **0.669**.

**Money result (test set):** doing nothing loses **$14,766** to fraud; with the model the
total cost is **$3,827** → **74.1% of fraud money saved**.

## Step 10 — Explain the model

SHAP analysis shows the model mostly relies on **V14, V4, V12, V10** — exactly the features
EDA flagged in Step 4. The model learned what we expected it to learn. ✅

## Step 11 — Deploy

- Reusable predictor: transaction in → probability + GENUINE/FRAUD + LOW/MEDIUM/HIGH risk.
- **Streamlit web app** (single + batch scoring): `streamlit run app/streamlit_app.py`
- **39 automated tests**, all passing.
- Full audit trail: `logs/`, `artifacts/`, `reports/`, git history.

## Problems faced (interviewers love these)

1. **Extreme imbalance (578:1)** — solved with class weighting + PR-AUC + threshold tuning.
2. **1,081 duplicate rows** — a hidden cheating risk; removed before splitting.
3. **LightGBM collapsed** under extreme class weight — caught by comparing everything.
4. **Hyperparameter tuning made the model slightly worse** (0.876 vs 0.878) — we kept the
   simpler model. Tuning is not magic.
5. **Validation score (0.879) vs test score (0.821)** — normal randomness with only 95 test
   frauds; reported honestly instead of hidden.
6. **SMOTE-Tomek took 60× longer for zero improvement** — fancier ≠ better.
7. **V1–V28 are anonymized** — we can't know what they physically mean; a real limitation.

---

**One-line summary to memorize:**

> "284,807 transactions, only 0.17% fraud — so accuracy is a lie; I compared 17
> model/imbalance combinations, picked XGBoost by PR-AUC, tuned the threshold by dollar
> cost, and on unseen data caught 76% of frauds with only 3 false alarms, saving 74% of
> fraud losses."
