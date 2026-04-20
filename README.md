# AllML — All-in-One Machine Learning Library

> **One import to rule them all.**
> Load a CSV, pick your columns, choose a method, train, predict. Done.

```python
from allml import AllML

ml = AllML("data.csv")
ml.col_to_feed(["feature1", "feature2", "feature3"])
ml.col_to_pred(["target"])
ml.method("random_forest_classifier")
ml.split(70, 20, 10)
ml.train(n_estimators=100)
ml.show()
ml.save("models/my_model")

result = ml.pred(5.1, 3.2, 1.4)
```

---

## What is AllML?

AllML is a unified machine learning interface that wraps every major
scikit-learn algorithm (plus XGBoost, LightGBM, CatBoost when installed)
behind a single consistent API.

No boilerplate. No manual preprocessing. No separate scalers, encoders,
or train/test split code. You give it a CSV file and tell it what to do.
AllML handles the rest.

It automatically detects whether your task is:

- Regression (predicting a number)
- Binary classification (predicting 0 or 1)
- Multi-class classification (predicting one of many categories)
- Multi-output regression (predicting several numbers at once)
- Multi-output classification (predicting several categories at once)

---

## Installation

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

Optional boosting libraries (recommended):

```bash
pip install xgboost lightgbm catboost
```

Then drop `allml.py` into your project folder.

---

## Quick Start

```python
from allml import AllML

ml = AllML("house_prices.csv")
ml.col_to_feed(["area_sqft", "bedrooms", "bathrooms", "year_built"])
ml.col_to_pred(["price_usd"])
ml.method("random_forest_regressor")
ml.split(70, 20, 10)
ml.train(n_estimators=150)
ml.show()
ml.show_graph("all")
ml.save("models/house_model")

price = ml.pred(2000, 3, 2, 2010)
print(f"Predicted price: ${price:,.2f}")
```

---

## API Reference

### `AllML(csv_filepath)`

Load your dataset. This is always the first step.

```python
ml = AllML("data.csv")
```

---

### `ml.col_to_feed(columns)`

Set the input feature columns. Pass a list of column names.

```python
ml.col_to_feed(["age", "income", "credit_score"])
```

---

### `ml.col_to_pred(columns)`

Set the target column(s) to predict.
Pass a single column for standard tasks,
or multiple columns for multi-output tasks.

```python
# Single target
ml.col_to_pred(["price"])

# Multiple targets
ml.col_to_pred(["temperature", "humidity", "rainfall"])
```

AllML automatically detects the task type from the target column:

| Target column type | Task detected |
|---|---|
| Float values | Regression |
| 2 unique integer values | Binary Classification |
| 3+ unique integer / string values | Multi-class Classification |
| Multiple float columns | Multi-Output Regression |
| Multiple category columns | Multi-Output Classification |

---

### `ml.method(name)`

Choose which algorithm(s) to use.

```python
# Single method
ml.method("random_forest_classifier")

# Multiple methods
ml.method(["logistic_regression", "svc", "gradient_boosting_classifier"])

# All applicable methods for your task type
ml.method("all")
```

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
| `ransac_regressor` | RANSAC Regressor |
| `theilsen_regressor` | Theil-Sen Regressor |
| `ard_regression` | ARD Regression |
| `passive_aggressive_regressor` | Passive Aggressive Regressor |
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
| `nu_svc` | Nu SVC (auto-tunes nu) |
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

`*` Only available if the optional library is installed.

If you select a method that does not match your task type,
AllML will skip it with an explanation and suggest correct alternatives.

---

### `ml.split(train_pct, test_pct, val_pct=0)`

Split your data into training, test, and optional validation sets.
Percentages must sum to 100.

```python
# 80% train, 20% test
ml.split(80, 20)

# 70% train, 20% test, 10% validation
ml.split(70, 20, 10)
```

---

### `ml.train(...)`

Train the selected model(s). All parameters are optional
and will be applied where applicable.

```python
ml.train(
    epochs         = 100,    # max iterations for iterative models
    learning_rate  = 0.01,   # step size for gradient-based models
    n_estimators   = 100,    # trees for ensemble models
    max_depth      = None,   # max tree depth (None = unlimited)
    n_neighbors    = 5,      # k for KNN
    kernel         = "rbf",  # kernel for SVM  ("rbf", "linear", "poly")
    n_clusters     = 3,      # clusters for clustering models
    alpha          = 1.0,    # regularisation strength
)
```

AllML maps each parameter to the correct argument name for every model.
You do not need to worry about which model uses which parameter name.

---

### `ml.show()`

Print a detailed summary of the model, data splits, training config,
and all performance metrics.

```python
ml.show()
```

Output includes:

- Dataset file, shape, feature and target columns
- Detected task type
- Train / test / validation sample counts
- All training hyperparameters
- Per-model metrics (accuracy, R2, RMSE, F1, confusion matrix, etc.)
- Missing value report

---

### `ml.show_graph(graph_type)`

Plot one, several, or all available graphs.

```python
# Single graph
ml.show_graph("confusion_matrix")

# Multiple graphs
ml.show_graph(["roc_curve", "feature_importance", "model_comparison"])

# Every applicable graph
ml.show_graph("all")
```

#### Available Graphs

| Graph | Task |
|---|---|
| `confusion_matrix` | Classification |
| `roc_curve` | Binary Classification |
| `precision_recall_curve` | Binary Classification |
| `feature_importance` | Tree & linear models |
| `learning_curve` | All |
| `residuals` | Regression |
| `prediction_vs_actual` | Regression |
| `error_distribution` | Regression |
| `model_comparison` | All (bar chart of main metric) |
| `correlation_heatmap` | All |
| `distribution` | All |
| `boxplot` | All |
| `pairplot` | All |
| `cluster_plot` | Clustering |

---

### `ml.save(path)`

Save the trained model(s) to disk.

```python
ml.save("models/my_model")
```

- Single method  → saves `my_model.allml`
- Multiple methods → saves `my_model_<method_name>.allml` for each
- Always saves `my_model_meta.json` with a human-readable summary

The `.allml` file bundles the trained model, fitted scaler,
label encoders, and all metadata needed to predict later.

---

### `ml.load(path)`

Load a previously saved model. After loading, `ml.pred()` works
immediately without any other setup.

```python
ml.load("models/my_model.allml")
result = ml.pred(5.1, 3.2, 1.4)
```

---

### `ml.pred(*values)`

Run a prediction. Pass values in the same order as `col_to_feed()`.

```python
# Regression
price = ml.pred(2000, 3, 2, 2010)

# Classification
label = ml.pred(6.5, 85.0, 72.0, 7.5)

# Multi-output (returns list)
temp, humidity, rain = ml.pred(7, 200, 65.0, 1013.0, 15.0, 40.0, 100.0)

# Predict using a specific method when multiple were trained
price = ml.pred(2000, 3, 2, 2010, method="ridge")
```

When multiple models are trained, `ml.pred()` returns a dictionary
mapping each method name to its prediction.

---

## Task Examples

### Regression — Predict a number

```python
from allml import AllML

ml = AllML("employee_salary.csv")
ml.col_to_feed(["age", "years_experience", "education_level",
                "performance_score", "certifications"])
ml.col_to_pred(["annual_salary"])
ml.method("gradient_boosting_regressor")
ml.split(70, 20, 10)
ml.train(n_estimators=200, learning_rate=0.05)
ml.show()
ml.show_graph(["prediction_vs_actual", "feature_importance",
               "residuals", "model_comparison"])
ml.save("models/salary_model")

salary = ml.pred(35, 10, 3, 4, 2)
print(f"Predicted salary: ${salary:,.0f}")
```

---

### Binary Classification — Predict 0 or 1

```python
from allml import AllML

ml = AllML("heart_disease.csv")
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
print(f"Heart disease risk: {'High' if result == 1 else 'Low'}")
```

---

### Multi-class Classification — Predict a category

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

### Multi-Output Regression — Predict several numbers

```python
from allml import AllML

ml = AllML("weather_forecast.csv")
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

### Train All Methods and Compare

```python
from allml import AllML

ml = AllML("loan_approval.csv")
ml.col_to_feed(["annual_income", "loan_amount", "credit_score",
                "employment_years", "existing_debt_pct"])
ml.col_to_pred(["approved"])
ml.method("all")          # trains every classification algorithm
ml.split(70, 20, 10)
ml.train(n_estimators=50, epochs=100)
ml.show()
ml.show_graph("model_comparison")   # bar chart of all model accuracies
ml.save("models/loan_all")          # saves one file per trained model
```

---

### Load and Predict (No Training Required)

```python
from allml import AllML

ml = AllML("any_file.csv")     # CSV required for initialisation only
ml.load("models/salary_model.allml")

salary = ml.pred(40, 15, 4, 5, 3)
print(f"Salary: ${salary:,.0f}")
```

---

### Chained API

Every method returns `self`, so calls can be chained.

```python
from allml import AllML

result = (
    AllML("data.csv")
        .col_to_feed(["f1", "f2", "f3"])
        .col_to_pred(["target"])
        .method("random_forest_regressor")
        .split(80, 20)
        .train(n_estimators=100)
)
result.show()
```

---

## Automatic Behaviours

These things happen without any extra code from you.

| Behaviour | Details |
|---|---|
| Task detection | Inferred from the target column type and unique value count |
| Categorical encoding | String columns are label-encoded automatically |
| Feature scaling | All features are standardised before training |
| Target scaling | Regression targets are scaled and inverse-transformed for predictions |
| Multi-output wrapping | Models are wrapped in `MultiOutputRegressor` or `MultiOutputClassifier` as needed |
| Wrong method warning | If you pick a classifier for a regression task, AllML warns you and suggests alternatives |
| NuSVC auto-tuning | The `nu` parameter is computed from your class distribution so it is always feasible |
| Parameter mapping | `learning_rate`, `alpha`, `kernel`, etc. are mapped to the correct argument for each model |

---

## Supported Metrics

| Task | Metrics |
|---|---|
| Regression | MSE, RMSE, MAE, R2 |
| Classification | Accuracy, F1 (weighted), Precision, Recall, ROC-AUC, Confusion Matrix |
| Clustering | Silhouette Score, Davies-Bouldin Score |

---

## File Format

Saved models use the `.allml` extension.
Each file is a Python pickle bundle containing:

- The trained model object
- The fitted `StandardScaler` for features
- The fitted `StandardScaler` for targets (regression)
- All label encoders for categorical columns
- Metadata (column names, task type, metrics, hyperparameters)

A companion `_meta.json` file is also written with human-readable info.

```
models/
  salary_model.allml          # single model bundle
  salary_model_meta.json      # human-readable summary

  loan_all_random_forest_classifier.allml
  loan_all_gradient_boosting_classifier.allml
  loan_all_logistic_regression.allml
  ...
  loan_all_meta.json
```

---

## Notes

- `isotonic_regression` requires exactly one feature column.
- `pls_regression` automatically caps `n_components` to the number of features.
- `nu_svc` and `nu_svr` automatically select a feasible `nu` value.
- `gaussian_process_regressor` and `gaussian_process_classifier` can be slow on large datasets.
- The `kernel` parameter applies to SVM models only. Gaussian Process models use their own internal kernel.
- Categorical feature columns and target columns are encoded automatically. You do not need to pre-process them.

---

## License

MIT License. Free to use, modify, and distribute.

---

## Contributing

Pull requests are welcome.
If a model produces unexpected errors or a parameter is not being
mapped correctly, please open an issue with the method name,
dataset shape, and the full error message.

## Author
**IMApurbo**  
GitHub: [@IMApurbo](https://github.com/IMApurbo)
