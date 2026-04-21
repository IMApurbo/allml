# AllML — All-in-One Machine Learning Library

> One import to rule them all.
> Load a CSV, engineer your features, pick your columns, choose a method, train, predict. Done.

```python
from allml import AllML

ml = AllML("your_data.csv")
ml.col_to_feed(["feature1", "feature2", "feature3"])
ml.col_to_pred(["target"])
ml.method("random_forest_regressor")
ml.split(70, 20, 10)
ml.train(n_estimators=200)
ml.show()
ml.save("models/my_model")

result = ml.pred(...)
print(result)
```

---

## What is AllML?

AllML is a unified machine learning interface that wraps every major
scikit-learn algorithm — plus XGBoost, LightGBM, and CatBoost when installed —
behind a single consistent API.

No boilerplate. No manual preprocessing. No separate scalers, encoders,
or train/test split code. You give it a CSV file and tell it what to do.
AllML handles everything else automatically.

- String and categorical columns are encoded automatically
- Raw string values can be passed directly into `ml.pred()`
- Missing values are reported on load and handled with one line
- Saving a model bundles everything needed to predict later — no CSV required at inference time

It automatically detects whether your task is:

- **Regression** — predicting a decimal number
- **Binary Classification** — predicting 0 or 1
- **Multi-class Classification** — predicting one of many categories
- **Multi-Output Regression** — predicting several numbers at once
- **Multi-Output Classification** — predicting several categories at once

---

## Installation

```bash
pip install allml
```

Optional boosting libraries for extra algorithms:

```bash
pip install xgboost lightgbm catboost
```

---

## API Reference

### `AllML(csv_filepath=None)`

Load your dataset. Providing a CSV is required for training.
For loading a saved model and predicting, no CSV is needed.

```python
# Training mode — CSV required
ml = AllML("your_data.csv")

# Inference mode — no CSV needed
ml = AllML()
ml.load("models/my_model.allml")
```

When a CSV is loaded, AllML automatically reports how many
missing values exist and which columns are affected.

---

### `ml.engineer`

Feature engineering module attached to every AllML instance.
All operations are applied directly to the loaded DataFrame
and logged so you can inspect every step with `ml.engineer.report()`.

#### Fill Missing Values

```python
ml.engineer.fill_missing("column_name", strategy="mean")
ml.engineer.fill_missing("column_name", strategy="median")
ml.engineer.fill_missing("column_name", strategy="constant", fill_value=0)
ml.engineer.fill_missing(strategy="mode")           # all columns
ml.engineer.fill_missing(strategy="ffill")          # forward fill
ml.engineer.fill_missing(strategy="bfill")          # backward fill
ml.engineer.fill_missing(strategy="knn", knn_neighbors=5)
ml.engineer.fill_missing(strategy="iterative")      # MICE imputation
```

| Strategy | Description |
|---|---|
| `mean` | Replace with column mean |
| `median` | Replace with column median |
| `mode` | Replace with most frequent value |
| `constant` | Replace with a fixed value |
| `ffill` | Forward fill from the previous row |
| `bfill` | Backward fill from the next row |
| `knn` | K-Nearest Neighbours imputation |
| `iterative` | MICE iterative imputation |

#### Drop Columns and Rows

```python
ml.engineer.drop_columns(["col1", "col2"])
ml.engineer.drop_duplicates()
ml.engineer.drop_duplicates(subset=["id_column"])
ml.engineer.drop_missing_rows()
ml.engineer.drop_missing_rows(columns=["col1", "col2"])
ml.engineer.drop_missing_rows(threshold=0.5)
```

#### Encode Categorical Columns

```python
ml.engineer.encode("column_name", method="label")
ml.engineer.encode(["col1", "col2"], method="onehot")
ml.engineer.encode("column_name", method="ordinal",
                   categories=["low", "medium", "high"])
ml.engineer.encode("column_name", method="binary")
```

| Method | Description |
|---|---|
| `label` | Integer label encoding — 0, 1, 2, ... |
| `onehot` | One-hot encoding — creates new columns |
| `ordinal` | Ordered integer encoding using a provided category list |
| `binary` | Maps exactly 2 unique values to 0 and 1 |

> **Note:** You do not need to manually encode columns before training.
> AllML encodes all string columns in `col_to_feed()` and `col_to_pred()`
> automatically at train time and applies the same encoding at predict time.
> Use `ml.engineer.encode()` only when you want explicit control over
> how a column is encoded.

#### Scale and Normalise

```python
ml.engineer.scale("column_name", method="standard")
ml.engineer.scale("column_name", method="minmax")
ml.engineer.scale("column_name", method="robust")
ml.engineer.scale("column_name", method="log")
ml.engineer.scale("column_name", method="sqrt")
ml.engineer.scale(["col1", "col2"], method="standard")
```

| Method | Description |
|---|---|
| `standard` | Zero mean, unit variance — z-score |
| `minmax` | Scale to range [0, 1] |
| `robust` | Scale using median and IQR — resistant to outliers |
| `log` | Natural log transform — log(x + 1) |
| `sqrt` | Square root transform |

#### Handle Outliers

```python
ml.engineer.clip_outliers("column_name", method="iqr", factor=1.5)
ml.engineer.clip_outliers("column_name", method="zscore", z_threshold=3.0)
ml.engineer.clip_outliers(["col1", "col2"], method="iqr")
```

| Method | Description |
|---|---|
| `iqr` | Clip values outside Q1 - factor * IQR and Q3 + factor * IQR |
| `zscore` | Clip values where the absolute z-score exceeds the threshold |

#### Create New Features

```python
# Polynomial features
ml.engineer.polynomial("column_name", degree=2)
ml.engineer.polynomial(["col1", "col2"], degree=3)

# Interaction feature — multiplies two columns together
ml.engineer.interaction("col_a", "col_b")
ml.engineer.interaction("col_a", "col_b", new_name="custom_name")

# Bin a continuous column into discrete intervals
ml.engineer.bin("column_name", bins=5)
ml.engineer.bin("column_name", bins=[0, 100, 500, 1000],
                labels=["Low", "Medium", "High"])

# Extract features from a datetime column
ml.engineer.datetime_features("datetime_column")
ml.engineer.datetime_features("datetime_column",
                               features=["year", "month", "dayofweek"])

# Apply a custom function to a column
ml.engineer.transform("column_name", lambda x: x / 1000,
                       new_name="new_column_name")

# Rename columns
ml.engineer.rename({"old_name": "new_name"})
```

#### Inspect and Visualise

```python
ml.engineer.report()
ml.engineer.plot_missing()
ml.engineer.plot_distribution("column_name")
ml.engineer.plot_distribution(["col1", "col2", "col3"])
```

#### Chaining

All engineer operations return `self` so they can be chained:

```python
(ml.engineer
   .fill_missing("column_name", strategy="mean")
   .drop_duplicates()
   .clip_outliers("column_name", method="iqr")
   .encode("column_name", method="label")
   .scale("column_name", method="minmax")
   .report())
```

---

### `ml.col_to_feed(columns)`

Set the input feature columns. String and categorical columns
are accepted without any manual encoding.

```python
ml.col_to_feed(["col1", "col2", "col3"])
```

---

### `ml.col_to_pred(columns)`

Set the target column(s) to predict.

```python
# Single target
ml.col_to_pred(["target_column"])

# Multiple targets
ml.col_to_pred(["target1", "target2", "target3"])
```

AllML detects the task type automatically:

| Target column content | Task detected |
|---|---|
| Float values | Regression |
| 2 unique values | Binary Classification |
| 3 or more unique values | Multi-class Classification |
| Multiple float columns | Multi-Output Regression |
| Multiple category columns | Multi-Output Classification |

---

### `ml.method(name)`

Choose which algorithm(s) to use.

```python
# Single method
ml.method("method_name")

# Multiple methods trained and compared together
ml.method(["method1", "method2", "method3"])

# Every applicable method for your task type
ml.method("all")
```

If you select a method that does not match your task type, AllML skips it
with a clear explanation and suggests the correct alternatives.

#### All Available Methods

**Regression**

| Method Name | Algorithm |
|---|---|
| `linear_regression` | Linear Regression |
| `ridge` | Ridge Regression |
| `lasso` | Lasso Regression |
| `elastic_net` | Elastic Net |
| `bayesian_ridge` | Bayesian Ridge |
| `sgd_regressor` | SGD Regressor |
| `huber_regressor` | Huber Regressor |
| `ransac_regressor` | RANSAC |
| `theilsen_regressor` | Theil-Sen |
| `ard_regression` | ARD Regression |
| `passive_aggressive_regressor` | Passive Aggressive |
| `decision_tree_regressor` | Decision Tree |
| `extra_tree_regressor` | Extra Tree |
| `random_forest_regressor` | Random Forest |
| `extra_trees_regressor` | Extra Trees |
| `gradient_boosting_regressor` | Gradient Boosting |
| `adaboost_regressor` | AdaBoost |
| `bagging_regressor` | Bagging |
| `svr` | Support Vector Regression |
| `linear_svr` | Linear SVR |
| `nu_svr` | Nu SVR |
| `knn_regressor` | K-Nearest Neighbors |
| `mlp_regressor` | Neural Network (MLP) |
| `gaussian_process_regressor` | Gaussian Process |
| `pls_regression` | PLS Regression |
| `isotonic_regression` | Isotonic Regression (1 feature only) |
| `lars` | LARS |
| `lasso_lars` | Lasso LARS |
| `xgboost_regressor` | XGBoost * |
| `lightgbm_regressor` | LightGBM * |
| `catboost_regressor` | CatBoost * |

**Classification**

| Method Name | Algorithm |
|---|---|
| `logistic_regression` | Logistic Regression |
| `ridge_classifier` | Ridge Classifier |
| `sgd_classifier` | SGD Classifier |
| `passive_aggressive_classifier` | Passive Aggressive |
| `perceptron` | Perceptron |
| `decision_tree_classifier` | Decision Tree |
| `extra_tree_classifier` | Extra Tree |
| `random_forest_classifier` | Random Forest |
| `extra_trees_classifier` | Extra Trees |
| `gradient_boosting_classifier` | Gradient Boosting |
| `adaboost_classifier` | AdaBoost |
| `bagging_classifier` | Bagging |
| `svc` | Support Vector Machine |
| `linear_svc` | Linear SVC |
| `nu_svc` | Nu SVC |
| `knn_classifier` | K-Nearest Neighbors |
| `gaussian_nb` | Gaussian Naive Bayes |
| `bernoulli_nb` | Bernoulli Naive Bayes |
| `mlp_classifier` | Neural Network (MLP) |
| `gaussian_process_classifier` | Gaussian Process |
| `lda` | Linear Discriminant Analysis |
| `qda` | Quadratic Discriminant Analysis |
| `xgboost_classifier` | XGBoost * |
| `lightgbm_classifier` | LightGBM * |
| `catboost_classifier` | CatBoost * |

**Clustering**

| Method Name | Algorithm |
|---|---|
| `kmeans` | K-Means |
| `dbscan` | DBSCAN |
| `agglomerative_clustering` | Agglomerative Clustering |
| `mean_shift` | Mean Shift |
| `spectral_clustering` | Spectral Clustering |
| `birch` | BIRCH |
| `mini_batch_kmeans` | Mini-Batch K-Means |

`*` Requires the optional library to be installed.

---

### `ml.split(train_pct, test_pct, val_pct=0)`

Split your data into training, test, and optional validation sets.
Percentages must sum to 100.

```python
ml.split(train_pct, test_pct)
ml.split(train_pct, test_pct, val_pct)
```

---

### `ml.train(...)`

Train the selected model(s). All parameters are optional
and are applied to every model where they are applicable.

```python
ml.train(
    epochs         = 100,
    learning_rate  = 0.01,
    n_estimators   = 100,
    max_depth      = None,
    n_neighbors    = 5,
    kernel         = "rbf",
    n_clusters     = 3,
    alpha          = 1.0,
)
```

| Parameter | Applies to |
|---|---|
| `epochs` | Iterative models — max iterations |
| `learning_rate` | Gradient-based models |
| `n_estimators` | Ensemble models — number of trees |
| `max_depth` | Tree-based models |
| `n_neighbors` | KNN models |
| `kernel` | SVM models — `"rbf"`, `"linear"`, `"poly"` |
| `n_clusters` | Clustering models |
| `alpha` | Regularised linear models |

---

### `ml.show()`

Print a complete summary including dataset info, data splits,
training configuration, all performance metrics, missing value
report, and the full list of feature engineering steps applied.

```python
ml.show()
```

---

### `ml.show_graph(graph_type)`

Plot one, several, or all available graphs.

```python
ml.show_graph("graph_name")
ml.show_graph(["graph1", "graph2", "graph3"])
ml.show_graph("all")
```

| Graph | Best for |
|---|---|
| `confusion_matrix` | Classification |
| `roc_curve` | Binary Classification |
| `precision_recall_curve` | Binary Classification |
| `feature_importance` | Tree and linear models |
| `learning_curve` | All tasks |
| `residuals` | Regression |
| `prediction_vs_actual` | Regression |
| `error_distribution` | Regression |
| `model_comparison` | All tasks — bar chart of the main metric |
| `correlation_heatmap` | All tasks |
| `distribution` | All tasks |
| `boxplot` | All tasks |
| `pairplot` | All tasks |
| `cluster_plot` | Clustering |

---

### `ml.save(path)`

Save trained model(s) to disk.

```python
ml.save("path/to/model_name")
```

- Single method → `model_name.allml`
- Multiple methods → `model_name_<method>.allml` for each model
- Always writes `model_name_meta.json` — a human-readable summary

Each `.allml` file bundles everything needed to make predictions later:
the trained model, fitted scalers, label encoders for all string columns,
column names, task type, metrics, and hyperparameters.

---

### `ml.load(path)`

Load a previously saved model. No CSV required.
After loading, `ml.pred()` works immediately.

```python
ml = AllML()
ml.load("path/to/model_name.allml")
```

---

### `ml.pred(*values)`

Run a prediction. Pass values in the same order as `col_to_feed()`.
String values are accepted directly for categorical columns.

```python
result = ml.pred(val1, val2, val3, ...)
result = ml.pred(val1, val2, val3, ..., method="method_name")
```

When multiple models are trained, `ml.pred()` returns a dictionary
mapping each method name to its prediction.

---

## Examples

### Regression — Predict a house price

```python
from allml import AllML

ml = AllML("house_prices.csv")

ml.engineer.fill_missing(strategy="mean")
ml.engineer.clip_outliers("price_usd", method="iqr")
ml.engineer.report()

ml.col_to_feed(["area_sqft", "bedrooms", "bathrooms",
                "year_built", "garage_cars", "condition"])
ml.col_to_pred(["price_usd"])
ml.method("gradient_boosting_regressor")
ml.split(70, 20, 10)
ml.train(n_estimators=200, learning_rate=0.05)
ml.show()
ml.show_graph(["prediction_vs_actual", "feature_importance",
               "residuals", "model_comparison"])
ml.save("models/house_model")

price = ml.pred(1360, 2, 1, 1981, 1, "Excellent")
print(f"Predicted price: ${price:,.2f}")
```

---

### Binary Classification — Predict heart disease risk

```python
from allml import AllML

ml = AllML("heart_disease.csv")

ml.engineer.fill_missing(strategy="median")
ml.engineer.clip_outliers(["cholesterol", "blood_pressure"])
ml.engineer.report()

ml.col_to_feed(["age", "sex", "cholesterol", "blood_pressure",
                "max_heart_rate", "smoking"])
ml.col_to_pred(["heart_disease"])
ml.method(["random_forest_classifier",
           "gradient_boosting_classifier",
           "logistic_regression"])
ml.split(70, 20, 10)
ml.train(n_estimators=100)
ml.show()
ml.show_graph(["confusion_matrix", "roc_curve",
               "precision_recall_curve", "model_comparison"])
ml.save("models/heart_model")

result = ml.pred(55, 1, 230, 140, 150, 1)
print(f"Heart disease: {'High risk' if result == 1 else 'Low risk'}")
```

---

### Multi-class Classification — Predict a flower species

```python
from allml import AllML

ml = AllML("flower_species.csv")

ml.col_to_feed(["sepal_length_cm", "sepal_width_cm",
                "petal_length_cm", "petal_width_cm"])
ml.col_to_pred(["species"])
ml.method("all")
ml.split(70, 20, 10)
ml.train(n_estimators=100)
ml.show()
ml.show_graph(["confusion_matrix", "feature_importance",
               "model_comparison"])
ml.save("models/flower_model")

species = ml.pred(5.1, 3.5, 1.4, 0.2)
print(f"Predicted species: {species}")
```

---

### Multi-Output Regression — Predict weather conditions

```python
from allml import AllML

ml = AllML("weather_forecast.csv")

ml.engineer.fill_missing(strategy="mean")

ml.col_to_feed(["month", "day_of_year", "humidity_pct",
                "pressure_hpa", "wind_speed_kmh",
                "cloud_cover_pct", "altitude_m"])
ml.col_to_pred(["temperature_c", "humidity_out_pct", "rainfall_mm"])
ml.method(["random_forest_regressor", "gradient_boosting_regressor"])
ml.split(70, 20, 10)
ml.train(n_estimators=150)
ml.show()
ml.show_graph(["model_comparison", "correlation_heatmap"])
ml.save("models/weather_model")

temp, humidity, rain = ml.pred(7, 200, 65.0, 1013.0, 15.0, 40.0, 100.0)
print(f"Temperature : {temp:.1f} C")
print(f"Humidity    : {humidity:.1f} %")
print(f"Rainfall    : {rain:.1f} mm")
```

---

### Compare All Methods — Loan approval

```python
from allml import AllML

ml = AllML("loan_approval.csv")

ml.engineer.fill_missing(strategy="median")
ml.engineer.clip_outliers("annual_income", method="iqr")

ml.col_to_feed(["annual_income", "loan_amount", "credit_score",
                "employment_years", "existing_debt_pct", "loan_purpose"])
ml.col_to_pred(["approved"])
ml.method("all")
ml.split(70, 20, 10)
ml.train(n_estimators=50, epochs=100)
ml.show()
ml.show_graph("model_comparison")
ml.save("models/loan_all")
```

---

### Load and Predict — No CSV Required

```python
from allml import AllML

ml = AllML()
ml.load("models/house_model.allml")

price = ml.pred(1360, 2, 1, 1981, 1, "Excellent")
print(f"Price: ${price:,.2f}")
```

---

### Chained API

```python
from allml import AllML

ml = AllML("your_data.csv")

(ml.engineer
   .fill_missing(strategy="mean")
   .drop_duplicates()
   .clip_outliers("target_column", method="iqr")
   .encode("category_column", method="label")
   .report())

(ml
   .col_to_feed(["col1", "col2", "col3", "category_column"])
   .col_to_pred(["target_column"])
   .method("random_forest_regressor")
   .split(70, 20, 10)
   .train(n_estimators=100))

ml.show()
ml.show_graph("all")
ml.save("models/my_model")
```

---

## How String Columns Work

AllML fully supports string and categorical values in both
feature columns and prediction inputs — no manual encoding required.

**At training time** — AllML detects non-numeric columns and fits a
label encoder on each one automatically.

**At prediction time** — the same encoder is applied so raw string
values like `"Excellent"` or `"Engineering"` are converted correctly.

**At save and load time** — all encoders are bundled inside the `.allml`
file so they are always available when you load and predict.

The only restriction is that a value passed to `ml.pred()` must have
been present in the training data. Passing an unseen value raises a
clear error showing the valid options.

---

## Automatic Behaviours

| Behaviour | Details |
|---|---|
| Task detection | Inferred from target column type and unique value count |
| String column encoding | All non-numeric columns are encoded automatically |
| String pred inputs | Raw strings can be passed directly to `ml.pred()` |
| Encoder persistence | Label encoders are saved inside `.allml` files and restored on load |
| Feature scaling | All features are standardised before training |
| Target scaling | Regression targets are scaled and inverse-transformed for predictions |
| Multi-output wrapping | Models are wrapped in MultiOutputRegressor or MultiOutputClassifier as needed |
| Wrong method warning | Picking a classifier for a regression task is caught with a suggestion |
| NuSVC auto-tuning | `nu` is computed from the class distribution — always feasible |
| Parameter mapping | All hyperparameters are mapped to the correct internal argument per model |
| Missing value warning | AllML reports missing value counts and affected columns on CSV load |

---

## Supported Metrics

| Task | Metrics |
|---|---|
| Regression | MSE, RMSE, MAE, R2 |
| Classification | Accuracy, F1 (weighted), Precision, Recall, ROC-AUC, Confusion Matrix |
| Clustering | Silhouette Score, Davies-Bouldin Score |

---

## Saved File Structure

```
models/
  my_model.allml                              # single model bundle
  my_model_meta.json                          # human-readable summary

  my_model_random_forest_classifier.allml     # one file per method
  my_model_gradient_boosting_classifier.allml
  my_model_logistic_regression.allml
  my_model_meta.json
```

---

## Notes

- `isotonic_regression` requires exactly one feature column.
- `pls_regression` automatically caps `n_components` to the number of features.
- `nu_svc` and `nu_svr` automatically select a feasible `nu` value based on training data.
- `gaussian_process_regressor` and `gaussian_process_classifier` can be slow on large datasets.
- The `kernel` parameter applies to SVM models only.
- String values in `ml.pred()` must match values seen during training.
- Calling `col_to_feed()`, `col_to_pred()`, or `split()` without a CSV raises a clear error with instructions.

---

## License

MIT License. Free to use, modify, and distribute.

---

## Contributing

Pull requests are welcome.
If a model produces unexpected errors or a parameter is not mapped
correctly, open an issue with the method name, dataset shape,
and the full error message.
---

## Author
**IMApurbo**  
GitHub: [@IMApurbo](https://github.com/IMApurbo)
