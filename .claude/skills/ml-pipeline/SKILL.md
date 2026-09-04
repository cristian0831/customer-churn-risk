---
name: ml-pipeline
description: Scaffold the standard small-ML-project notebook pipeline (Extracting Data, EDA, Preprocessing, Training, Model Evaluation) into a Jupyter notebook. Use when the user asks to insert/add the pipeline structure, scaffold a notebook, or add a section from it (e.g. "add the EDA section", "scaffold feature scaling").
---

# ML pipeline notebook scaffold

Inserts the user's standard small-ML-project section structure into a Jupyter
notebook, as markdown headers plus minimal, ready-to-run starter code cells.
This is a scaffold, not finished analysis — code cells are deliberately small
and generic so the user fills in the real logic.

## Full section structure

1. Extracting Data
2. EDA
   2.1 Missing Data
   2.2 Target Variable
   2.3 Visualizing Data
   2.4 Data Cleaning
   2.5 Feature Engineering
   2.6 Feature Selection
3. Preprocessing Data
   3.1 Feature Scaling
   3.2 Encoding Categorical Variables
4. Training
5. Model Evaluation

## How to run this skill

1. **Pick the target notebook.** If the user names a path, use it. Otherwise
   use the notebook currently open in the IDE (check for an
   `ide_opened_file`/`ide_selection` reference in context), or ask if neither
   is available.
2. **Read the notebook first** with the Read tool — NotebookEdit requires
   this. Look at existing cells to learn:
   - the dataframe variable name (default `df` if none exists yet)
   - the target/label column name (default `target`)
   - which sections already exist, so you don't duplicate them
   - what's already imported, so you don't re-add redundant imports
3. **Scope the insert.** By default, insert every section that isn't already
   present. If the user names specific sections/subsections (e.g. "add 2.1
   and 2.2", "scaffold preprocessing", "just feature scaling"), insert only
   those, matched by the numbering/names above.
4. **Insert, don't overwrite.** Use `NotebookEdit` with `edit_mode: insert`,
   appending after the last existing cell (or after the relevant prior
   section if inserting into the middle). Never `replace` a cell that
   already has user content.
5. **Cell pattern per section:** one `markdown` cell with the header (use
   `#` for top-level sections 1/2/3/4/5, `##` for numbered subsections),
   immediately followed by one or more `code` cells with the starter code
   below. Adapt variable/column names to what you found in step 2.
6. **Split before fitting anything, to avoid data leakage.** `train_test_split`
   must run as the *first* code cell under "3. Preprocessing Data", before
   3.1/3.2. Any fitted transform (`StandardScaler`, encoders, imputers, etc.)
   must `fit` on `X_train` only and `transform` (never `fit_transform`) on
   `X_test`. Section "4. Training" then only fits the model — it does not
   split. This applies even for a partial insert: if the user asks for just
   "3.1 Feature Scaling" and no split cell exists yet in the notebook, insert
   the split first.
7. After inserting, briefly tell the user which sections were added and
   which were skipped (already present), in one or two sentences — don't
   re-explain the whole structure back to them.

## Starter code per section

Use these as templates; adapt column names to the real dataframe. Keep
imports minimal and only add ones not already present in the notebook.

**1. Extracting Data**
```python
DIR = "./data/"
FILENAME = "<dataset>.csv"

df = pd.read_csv(DIR + FILENAME)
df.head()
```

**2.1 Missing Data**
```python
df.isnull().sum().sort_values(ascending=False)
```

**2.2 Target Variable**
```python
df["target"].value_counts(normalize=True)
```
```python
df["target"].value_counts().plot(kind="bar", title="Target distribution")
```

**2.3 Visualizing Data**
```python
df.hist(figsize=(12, 8), bins=30)
plt.tight_layout()
```
```python
import seaborn as sns

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
```

**2.4 Data Cleaning**
```python
# duplicates
print("Duplicate rows:", df.duplicated().sum())

# TODO: handle duplicates / outliers / invalid values found above
```

**2.5 Feature Engineering**
```python
# TODO: derive new features here
```

**2.6 Feature Selection**
```python
df.corr(numeric_only=True)["target"].sort_values(ascending=False)
```

**3. Preprocessing Data (split — insert first, before 3.1/3.2)**
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train = X_train.copy()
X_test = X_test.copy()
```

**3.1 Feature Scaling**
```python
from sklearn.preprocessing import StandardScaler

num_cols = X_train.select_dtypes(include="number").columns
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
```

**3.2 Encoding Categorical Variables**
```python
cat_cols = X_train.select_dtypes(include="object").columns
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
```

**4. Training**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

**5. Model Evaluation**
```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
```
```python
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
```

## Notes

- If `matplotlib`/`seaborn` aren't imported yet, add
  `import matplotlib.pyplot as plt` / `import seaborn as sns` once, near the
  top, not repeated per section.
- Don't invent columns the dataframe doesn't have — inspect it first.
- If the user's model choice differs (e.g. tree-based, XGBoost), swap the
  Training/Evaluation code accordingly rather than forcing LogisticRegression.
- Never fit a scaler/encoder/imputer on the full `df` before splitting — that
  leaks test-set statistics into training. Split first (see "3. Preprocessing
  Data" split cell above), fit on `X_train` only, transform `X_test`.
