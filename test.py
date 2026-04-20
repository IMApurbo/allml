# test_allml_complete.py
# Complete Test Suite for AllML Library
# Run: python test_allml_complete.py

import os
import sys
import time
import traceback
import numpy as np

# ============================================================
# Colors
# ============================================================
class C:
    G  = '\033[92m'
    Y  = '\033[93m'
    CY = '\033[96m'
    R  = '\033[91m'
    B  = '\033[1m'
    E  = '\033[0m'
    M  = '\033[35m'
    W  = '\033[97m'
    BL = '\033[94m'

def passed(text):  print(f"  {C.G}{C.B}✅ PASS{C.E} → {C.G}{text}{C.E}")
def failed(text):  print(f"  {C.R}{C.B}❌ FAIL{C.E} → {C.R}{text}{C.E}")
def skipped(text): print(f"  {C.Y}{C.B}⚠️  SKIP{C.E} → {C.Y}{text}{C.E}")
def info(text):    print(f"  {C.CY}ℹ️  {text}{C.E}")
def warn(text):    print(f"  {C.Y}⚠️  {text}{C.E}")

def section(title):
    print(f"\n{C.BL}{C.B}{'═'*65}{C.E}")
    print(f"{C.BL}{C.B}  🧪 {title}{C.E}")
    print(f"{C.BL}{C.B}{'═'*65}{C.E}")

def subsection(title):
    print(f"\n{C.M}{C.B}  ┌─ {title} {'─'*(55-len(title))}{C.E}")

def result_box(passed_n, failed_n, skipped_n, total, elapsed):
    status = C.G if failed_n == 0 else C.R
    print(f"""
{status}{C.B}
╔══════════════════════════════════════════════════════╗
║              TEST RESULTS SUMMARY                    ║
╠══════════════════════════════════════════════════════╣
║  ✅ Passed  : {str(passed_n).ljust(38)}║
║  ❌ Failed  : {str(failed_n).ljust(38)}║
║  ⚠️  Skipped : {str(skipped_n).ljust(38)}║
║  📊 Total   : {str(total).ljust(38)}║
║  ⏱️  Time    : {f'{elapsed:.2f}s'.ljust(38)}║
╚══════════════════════════════════════════════════════╝
{C.E}""")

# ============================================================
# Test Tracker
# ============================================================
class TestTracker:
    def __init__(self):
        self.passed  = 0
        self.failed  = 0
        self.skipped = 0
        self.errors  = []

    def ok(self, name):
        self.passed += 1
        passed(name)

    def fail(self, name, error=""):
        self.failed += 1
        self.errors.append((name, str(error)))
        failed(f"{name}  [{error}]")

    def skip(self, name, reason=""):
        self.skipped += 1
        skipped(f"{name}  [{reason}]")

    @property
    def total(self):
        return self.passed + self.failed + self.skipped

tracker = TestTracker()
DATA    = "test_datasets"
MODELS  = "test_models"
os.makedirs(MODELS, exist_ok=True)

start_time = time.time()

# ============================================================
# Import AllML
# ============================================================
section("IMPORTING AllML")
try:
    from allml import AllML
    tracker.ok("Import AllML")
except Exception as e:
    tracker.fail("Import AllML", e)
    print(f"\n{C.R}Cannot import AllML. Exiting.{C.E}")
    sys.exit(1)

# ============================================================
# Helper
# ============================================================
def safe_test(name, fn):
    """Run a test safely and track result."""
    try:
        fn()
        tracker.ok(name)
        return True
    except AssertionError as e:
        tracker.fail(name, f"AssertionError: {e}")
        return False
    except Exception as e:
        tracker.fail(name, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def make_ml(csv_name):
    """Create a fresh AllML instance."""
    return AllML(f"{DATA}/{csv_name}")

# ════════════════════════════════════════════════════════════
# TEST BLOCK 1 — LOADING & INITIALIZATION
# ════════════════════════════════════════════════════════════
section("TEST BLOCK 1 — LOADING & INITIALIZATION")

subsection("1.1 Valid CSV Loading")

def t1_1():
    ml = make_ml("house_prices.csv")
    assert ml._df is not None,          "DataFrame is None"
    assert ml._df.shape[0] > 0,         "No rows loaded"
    assert ml._df.shape[1] > 0,         "No columns loaded"
    assert ml._filepath.endswith(".csv"),"Filepath not set"

safe_test("Load house_prices.csv", t1_1)

def t1_2():
    ml = make_ml("student_performance.csv")
    assert ml._df.shape[0] > 0

safe_test("Load student_performance.csv", t1_2)

def t1_3():
    ml = make_ml("weather_forecast.csv")
    assert ml._df.shape[0] > 0

safe_test("Load weather_forecast.csv", t1_3)

subsection("1.2 Invalid File Error")

def t1_4():
    try:
        ml = AllML("nonexistent_file.csv")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass  # expected

safe_test("FileNotFoundError on missing CSV", t1_4)

subsection("1.3 DataFrame Content Check")

def t1_5():
    ml = make_ml("house_prices.csv")
    assert "price_usd"  in ml._df.columns
    assert "area_sqft"  in ml._df.columns
    assert "bedrooms"   in ml._df.columns

safe_test("house_prices.csv columns present", t1_5)

def t1_6():
    ml = make_ml("heart_disease.csv")
    assert "heart_disease" in ml._df.columns
    assert "age"            in ml._df.columns

safe_test("heart_disease.csv columns present", t1_6)

# ════════════════════════════════════════════════════════════
# TEST BLOCK 2 — col_to_feed & col_to_pred
# ════════════════════════════════════════════════════════════
section("TEST BLOCK 2 — col_to_feed & col_to_pred")

subsection("2.1 Single Column Feed")

def t2_1():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft"])
    assert ml._feed_cols == ["area_sqft"]

safe_test("Single feature column", t2_1)

subsection("2.2 Multiple Columns Feed")

def t2_2():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft", "bedrooms", "bathrooms", "year_built"])
    assert len(ml._feed_cols) == 4

safe_test("Multiple feature columns", t2_2)

subsection("2.3 Single Target Column")

def t2_3():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft", "bedrooms"])
    ml.col_to_pred(["price_usd"])
    assert ml._pred_cols == ["price_usd"]

safe_test("Single target column", t2_3)

subsection("2.4 Multiple Target Columns")

def t2_4():
    ml = make_ml("weather_forecast.csv")
    ml.col_to_feed(["month", "humidity_pct", "pressure_hpa"])
    ml.col_to_pred(["temperature_c", "humidity_out_pct", "rainfall_mm"])
    assert len(ml._pred_cols) == 3

safe_test("Multiple target columns", t2_4)

subsection("2.5 Invalid Column Names")

def t2_5():
    ml = make_ml("house_prices.csv")
    try:
        ml.col_to_feed(["nonexistent_col"])
        assert False, "Should raise ValueError"
    except ValueError:
        pass

safe_test("ValueError on invalid feed column", t2_5)

def t2_6():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft"])
    try:
        ml.col_to_pred(["not_a_column"])
        assert False, "Should raise ValueError"
    except ValueError:
        pass

safe_test("ValueError on invalid pred column", t2_6)

subsection("2.6 Task Type Detection")

def t2_7():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft", "bedrooms"])
    ml.col_to_pred(["price_usd"])
    assert ml._task_type == "regression", f"Got: {ml._task_type}"

safe_test("Task = regression (house prices)", t2_7)

def t2_8():
    ml = make_ml("student_performance.csv")
    ml.col_to_feed(["study_hours_per_day", "attendance_pct"])
    ml.col_to_pred(["passed"])
    assert ml._task_type == "binary_classification", f"Got: {ml._task_type}"

safe_test("Task = binary_classification (student)", t2_8)

def t2_9():
    ml = make_ml("flower_species.csv")
    ml.col_to_feed(["sepal_length_cm", "sepal_width_cm"])
    ml.col_to_pred(["species"])
    assert ml._task_type == "multiclass_classification", f"Got: {ml._task_type}"

safe_test("Task = multiclass_classification (iris)", t2_9)

def t2_10():
    ml = make_ml("weather_forecast.csv")
    ml.col_to_feed(["month", "humidity_pct"])
    ml.col_to_pred(["temperature_c", "rainfall_mm"])
    assert ml._task_type == "multi_output_regression", f"Got: {ml._task_type}"

safe_test("Task = multi_output_regression (weather)", t2_10)

# ════════════════════════════════════════════════════════════
# TEST BLOCK 3 — method()
# ════════════════════════════════════════════════════════════
section("TEST BLOCK 3 — method()")

subsection("3.1 Single Method Selection")

def t3_1():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft"])
    ml.col_to_pred(["price_usd"])
    ml.method("linear_regression")
    assert ml._method_names == ["linear_regression"]

safe_test("Single method: linear_regression", t3_1)

def t3_2():
    ml = make_ml("student_performance.csv")
    ml.col_to_feed(["study_hours_per_day", "attendance_pct"])
    ml.col_to_pred(["passed"])
    ml.method("random_forest_classifier")
    assert ml._method_names == ["random_forest_classifier"]

safe_test("Single method: random_forest_classifier", t3_2)

subsection("3.2 Multiple Methods")

def t3_3():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft", "bedrooms"])
    ml.col_to_pred(["price_usd"])
    ml.method(["linear_regression", "ridge", "lasso"])
    assert len(ml._method_names) == 3

safe_test("Multiple methods: [linear, ridge, lasso]", t3_3)

def t3_4():
    ml = make_ml("student_performance.csv")
    ml.col_to_feed(["study_hours_per_day", "attendance_pct"])
    ml.col_to_pred(["passed"])
    ml.method(["random_forest_classifier", "svc", "knn_classifier"])
    assert len(ml._method_names) == 3

safe_test("Multiple methods: [rf, svc, knn] for classifier", t3_4)

subsection("3.3 method('all') Selection")

def t3_5():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft", "bedrooms"])
    ml.col_to_pred(["price_usd"])
    ml.method("all")
    assert len(ml._method_names) > 10, f"Expected >10, got {len(ml._method_names)}"

safe_test("method('all') for regression", t3_5)

def t3_6():
    ml = make_ml("student_performance.csv")
    ml.col_to_feed(["study_hours_per_day", "attendance_pct"])
    ml.col_to_pred(["passed"])
    ml.method("all")
    assert len(ml._method_names) > 10

safe_test("method('all') for binary classification", t3_6)

subsection("3.4 Invalid Method")

def t3_7():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft"])
    ml.col_to_pred(["price_usd"])
    try:
        ml.method("fake_method_xyz")
        assert False, "Should raise ValueError"
    except ValueError:
        pass

safe_test("ValueError on invalid method name", t3_7)

subsection("3.5 Wrong Method for Task (suggestion test)")

def t3_8():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft", "bedrooms"])
    ml.col_to_pred(["price_usd"])
    # This should not raise but should print a warning when training
    ml.method("random_forest_classifier")   # classifier on regression task
    assert "random_forest_classifier" in ml._method_names

safe_test("Wrong method stored (warning shown at train time)", t3_8)

# ════════════════════════════════════════════════════════════
# TEST BLOCK 4 — split()
# ════════════════════════════════════════════════════════════
section("TEST BLOCK 4 — split()")

subsection("4.1 Train/Test Split (no validation)")

def t4_1():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft", "bedrooms", "bathrooms"])
    ml.col_to_pred(["price_usd"])
    ml.split(80, 20)
    assert ml._X_train is not None
    assert ml._X_test  is not None
    assert ml._X_val   is None
    assert ml._split_done

safe_test("80/20 split (no validation)", t4_1)

subsection("4.2 Train/Test/Validation Split")

def t4_2():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft", "bedrooms", "bathrooms"])
    ml.col_to_pred(["price_usd"])
    ml.split(70, 20, 10)
    assert ml._X_train is not None
    assert ml._X_test  is not None
    assert ml._X_val   is not None

safe_test("70/20/10 split (with validation)", t4_2)

subsection("4.3 Sample Count Sanity Check")

def t4_3():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft", "bedrooms"])
    ml.col_to_pred(["price_usd"])
    ml.split(70, 20, 10)
    total = ml._X_train.shape[0] + ml._X_test.shape[0] + ml._X_val.shape[0]
    original = ml._df.shape[0]
    ratio = total / original
    assert 0.95 <= ratio <= 1.05, f"Split ratio off: {ratio}"

safe_test("Total samples match original after split", t4_3)

subsection("4.4 Invalid Percentages")

def t4_4():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft"])
    ml.col_to_pred(["price_usd"])
    try:
        ml.split(50, 20, 10)  # sums to 80
        assert False, "Should raise ValueError"
    except ValueError:
        pass

safe_test("ValueError on percentages not summing to 100", t4_4)

subsection("4.5 Split Before col_to_feed (error check)")

def t4_5():
    ml = make_ml("house_prices.csv")
    try:
        ml.split(70, 20, 10)
        assert False, "Should raise ValueError"
    except (ValueError, RuntimeError):
        pass

safe_test("Error if split called before col_to_feed/pred", t4_5)

# ════════════════════════════════════════════════════════════
# TEST BLOCK 5 — REGRESSION TRAINING
# ════════════════════════════════════════════════════════════
section("TEST BLOCK 5 — REGRESSION TRAINING")

REGRESSION_QUICK_METHODS = [
    "linear_regression",
    "ridge",
    "lasso",
    "elastic_net",
    "decision_tree_regressor",
    "random_forest_regressor",
    "gradient_boosting_regressor",
    "svr",
    "knn_regressor",
    "mlp_regressor",
    "adaboost_regressor",
    "bagging_regressor",
    "extra_trees_regressor",
    "sgd_regressor",
    "bayesian_ridge",
]

subsection("5.1 Individual Regression Methods — House Prices")

for method_name in REGRESSION_QUICK_METHODS:
    def make_reg_test(m):
        def _test():
            ml = make_ml("house_prices.csv")
            ml.col_to_feed(["area_sqft", "bedrooms", "bathrooms",
                            "year_built", "garage_cars"])
            ml.col_to_pred(["price_usd"])
            ml.method(m)
            ml.split(70, 20, 10)
            ml.train(n_estimators=20, epochs=50)
            assert ml._is_trained
            assert m in ml._models
            assert m in ml._metrics
        return _test
    safe_test(f"Regression: {method_name}", make_reg_test(method_name))

subsection("5.2 Regression Metric Checks")

def t5_reg_metrics():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft", "bedrooms", "bathrooms", "year_built"])
    ml.col_to_pred(["price_usd"])
    ml.method("random_forest_regressor")
    ml.split(70, 20, 10)
    ml.train(n_estimators=30)
    m = ml._metrics["random_forest_regressor"]
    assert "MSE"  in m, "MSE missing"
    assert "RMSE" in m, "RMSE missing"
    assert "MAE"  in m, "MAE missing"
    assert "R2"   in m, "R2 missing"
    assert isinstance(m["R2"], float)
    assert -10 < m["R2"] <= 1.0, f"Unexpected R2: {m['R2']}"

safe_test("Regression metrics (MSE, RMSE, MAE, R2) present", t5_reg_metrics)

subsection("5.3 Multiple Regression Methods Together")

def t5_multi_reg():
    ml = make_ml("employee_salary.csv")
    ml.col_to_feed(["age", "years_experience", "education_level",
                    "performance_score", "certifications"])
    ml.col_to_pred(["annual_salary"])
    ml.method(["linear_regression", "ridge", "random_forest_regressor"])
    ml.split(70, 20, 10)
    ml.train(n_estimators=20)
    assert len(ml._models) >= 2

safe_test("Multiple regression methods trained together", t5_multi_reg)

subsection("5.4 Energy Consumption Regression")

def t5_energy():
    ml = make_ml("energy_consumption.csv")
    ml.col_to_feed(["temperature_c", "humidity_pct", "num_occupants",
                    "building_area_m2", "hour_of_day", "num_appliances"])
    ml.col_to_pred(["energy_kwh"])
    ml.method("gradient_boosting_regressor")
    ml.split(75, 25)
    ml.train(n_estimators=30, learning_rate=0.1)
    assert ml._is_trained

safe_test("Energy consumption regression", t5_energy)

# ════════════════════════════════════════════════════════════
# TEST BLOCK 6 — BINARY CLASSIFICATION TRAINING
# ════════════════════════════════════════════════════════════
section("TEST BLOCK 6 — BINARY CLASSIFICATION TRAINING")

BINARY_QUICK_METHODS = [
    "logistic_regression",
    "random_forest_classifier",
    "gradient_boosting_classifier",
    "decision_tree_classifier",
    "svc",
    "knn_classifier",
    "mlp_classifier",
    "adaboost_classifier",
    "bagging_classifier",
    "extra_trees_classifier",
    "gaussian_nb",
    "sgd_classifier",
    "ridge_classifier",
    "lda",
]

subsection("6.1 Individual Binary Classification Methods — Student")

for method_name in BINARY_QUICK_METHODS:
    def make_bin_test(m):
        def _test():
            ml = make_ml("student_performance.csv")
            ml.col_to_feed(["study_hours_per_day", "attendance_pct",
                            "previous_score", "sleep_hours", "assignments_done"])
            ml.col_to_pred(["passed"])
            ml.method(m)
            ml.split(70, 20, 10)
            ml.train(n_estimators=20, epochs=50)
            assert ml._is_trained
            assert m in ml._models
        return _test
    safe_test(f"Binary: {method_name}", make_bin_test(method_name))

subsection("6.2 Classification Metric Checks")

def t6_cls_metrics():
    ml = make_ml("student_performance.csv")
    ml.col_to_feed(["study_hours_per_day", "attendance_pct",
                    "previous_score", "sleep_hours"])
    ml.col_to_pred(["passed"])
    ml.method("random_forest_classifier")
    ml.split(70, 20, 10)
    ml.train(n_estimators=30)
    m = ml._metrics["random_forest_classifier"]
    assert "accuracy"    in m, "accuracy missing"
    assert "f1_weighted" in m, "f1 missing"
    assert "precision"   in m, "precision missing"
    assert "recall"      in m, "recall missing"
    assert "conf_matrix" in m, "conf_matrix missing"
    assert 0 <= m["accuracy"] <= 1.0

safe_test("Binary classification metrics present", t6_cls_metrics)

subsection("6.3 Heart Disease Binary Classification")

def t6_heart():
    ml = make_ml("heart_disease.csv")
    ml.col_to_feed(["age", "sex", "cholesterol", "blood_pressure",
                    "max_heart_rate", "fasting_blood_sugar", "chest_pain_type"])
    ml.col_to_pred(["heart_disease"])
    ml.method(["random_forest_classifier", "gradient_boosting_classifier"])
    ml.split(70, 20, 10)
    ml.train(n_estimators=25)
    assert len(ml._models) >= 1

safe_test("Heart disease binary classification", t6_heart)

subsection("6.4 Loan Approval Binary Classification")

def t6_loan():
    ml = make_ml("loan_approval.csv")
    ml.col_to_feed(["annual_income", "loan_amount", "credit_score",
                    "loan_term_years", "employment_years",
                    "existing_debt_pct", "num_defaults"])
    ml.col_to_pred(["approved"])
    ml.method("gradient_boosting_classifier")
    ml.split(75, 25)
    ml.train(n_estimators=30, learning_rate=0.1)
    assert ml._is_trained

safe_test("Loan approval binary classification", t6_loan)

# ════════════════════════════════════════════════════════════
# TEST BLOCK 7 — MULTICLASS CLASSIFICATION
# ════════════════════════════════════════════════════════════
section("TEST BLOCK 7 — MULTICLASS CLASSIFICATION")

MULTICLASS_METHODS = [
    "random_forest_classifier",
    "gradient_boosting_classifier",
    "decision_tree_classifier",
    "svc",
    "knn_classifier",
    "mlp_classifier",
    "logistic_regression",
    "extra_trees_classifier",
    "gaussian_nb",
    "lda",
    "qda",
]

subsection("7.1 Flower Species Multiclass")

for method_name in MULTICLASS_METHODS:
    def make_mc_test(m):
        def _test():
            ml = make_ml("flower_species.csv")
            ml.col_to_feed(["sepal_length_cm", "sepal_width_cm",
                            "petal_length_cm", "petal_width_cm"])
            ml.col_to_pred(["species"])
            ml.method(m)
            ml.split(70, 20, 10)
            ml.train(n_estimators=20, epochs=50)
            assert ml._is_trained
        return _test
    safe_test(f"Multiclass: {method_name}", make_mc_test(method_name))

subsection("7.2 Wine Quality Multiclass")

def t7_wine():
    ml = make_ml("wine_quality.csv")
    ml.col_to_feed(["fixed_acidity", "volatile_acidity", "citric_acid",
                    "chlorides", "alcohol_pct", "sulphates", "pH"])
    ml.col_to_pred(["quality"])
    ml.method("random_forest_classifier")
    ml.split(70, 20, 10)
    ml.train(n_estimators=30)
    assert ml._is_trained
    assert "accuracy" in ml._metrics["random_forest_classifier"]

safe_test("Wine quality multiclass", t7_wine)

subsection("7.3 All Classification Methods on Flower Species")

def t7_all_cls():
    ml = make_ml("flower_species.csv")
    ml.col_to_feed(["sepal_length_cm", "sepal_width_cm",
                    "petal_length_cm", "petal_width_cm"])
    ml.col_to_pred(["species"])
    ml.method("all")
    ml.split(70, 20, 10)
    ml.train(n_estimators=15, epochs=50)
    assert ml._is_trained
    assert len(ml._models) > 5, f"Expected >5 models, got {len(ml._models)}"

safe_test("method('all') multiclass classification", t7_all_cls)

# ════════════════════════════════════════════════════════════
# TEST BLOCK 8 — MULTI-OUTPUT REGRESSION
# ════════════════════════════════════════════════════════════
section("TEST BLOCK 8 — MULTI-OUTPUT REGRESSION")

MULTI_OUT_REG_METHODS = [
    "random_forest_regressor",
    "gradient_boosting_regressor",
    "decision_tree_regressor",
    "extra_trees_regressor",
    "knn_regressor",
    "mlp_regressor",
    "linear_regression",
    "ridge",
    "lasso",
]

subsection("8.1 Weather Forecast Multi-Output Regression")

for method_name in MULTI_OUT_REG_METHODS:
    def make_mo_test(m):
        def _test():
            ml = make_ml("weather_forecast.csv")
            ml.col_to_feed(["month", "day_of_year", "humidity_pct",
                            "pressure_hpa", "wind_speed_kmh",
                            "cloud_cover_pct", "altitude_m"])
            ml.col_to_pred(["temperature_c", "rainfall_mm"])
            ml.method(m)
            ml.split(70, 20, 10)
            ml.train(n_estimators=20, epochs=50)
            assert ml._is_trained
        return _test
    safe_test(f"Multi-Output Reg: {method_name}", make_mo_test(method_name))

subsection("8.2 Car Specs Multi-Output Regression")

def t8_car():
    ml = make_ml("car_specs.csv")
    ml.col_to_feed(["engine_cc", "horsepower", "torque_nm",
                    "weight_kg", "cylinders", "car_age_years",
                    "has_turbo", "is_electric", "brand_tier"])
    ml.col_to_pred(["price_usd", "fuel_economy_mpg"])
    ml.method("random_forest_regressor")
    ml.split(70, 20, 10)
    ml.train(n_estimators=30)
    assert ml._is_trained

safe_test("Car specs multi-output regression", t8_car)

subsection("8.3 Multi-Output with Multiple Methods")

def t8_multi():
    ml = make_ml("weather_forecast.csv")
    ml.col_to_feed(["month", "humidity_pct", "pressure_hpa", "wind_speed_kmh"])
    ml.col_to_pred(["temperature_c", "humidity_out_pct"])
    ml.method(["random_forest_regressor", "gradient_boosting_regressor"])
    ml.split(70, 20, 10)
    ml.train(n_estimators=20)
    assert len(ml._models) >= 1

safe_test("Multi-output with multiple methods", t8_multi)

# ════════════════════════════════════════════════════════════
# TEST BLOCK 9 — method("all") FULL TESTS
# ════════════════════════════════════════════════════════════
section("TEST BLOCK 9 — method('all') FULL TESTS")

subsection("9.1 All Methods: Regression")

def t9_all_reg():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft", "bedrooms", "bathrooms", "year_built"])
    ml.col_to_pred(["price_usd"])
    ml.method("all")
    ml.split(70, 20, 10)
    ml.train(n_estimators=10, epochs=30)
    assert ml._is_trained
    assert len(ml._models) >= 5
    info(f"  Trained {len(ml._models)} regression models")

safe_test("method('all') on regression task", t9_all_reg)

subsection("9.2 All Methods: Binary Classification")

def t9_all_bin():
    ml = make_ml("heart_disease.csv")
    ml.col_to_feed(["age", "sex", "cholesterol", "blood_pressure",
                    "max_heart_rate", "smoking"])
    ml.col_to_pred(["heart_disease"])
    ml.method("all")
    ml.split(70, 20, 10)
    ml.train(n_estimators=10, epochs=30)
    assert ml._is_trained
    assert len(ml._models) >= 5
    info(f"  Trained {len(ml._models)} classification models")

safe_test("method('all') on binary classification", t9_all_bin)

subsection("9.3 All Methods: Multiclass Classification")

def t9_all_mc():
    ml = make_ml("wine_quality.csv")
    ml.col_to_feed(["fixed_acidity", "volatile_acidity", "alcohol_pct",
                    "sulphates", "pH", "chlorides"])
    ml.col_to_pred(["quality"])
    ml.method("all")
    ml.split(70, 20, 10)
    ml.train(n_estimators=10, epochs=30)
    assert ml._is_trained
    assert len(ml._models) >= 5

safe_test("method('all') on multiclass classification", t9_all_mc)

# ════════════════════════════════════════════════════════════
# TEST BLOCK 10 — pred() PREDICTIONS
# ════════════════════════════════════════════════════════════
section("TEST BLOCK 10 — pred() PREDICTIONS")

subsection("10.1 Regression Prediction Output Type")

def t10_1():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft", "bedrooms", "bathrooms", "year_built"])
    ml.col_to_pred(["price_usd"])
    ml.method("linear_regression")
    ml.split(70, 20, 10)
    ml.train()
    result = ml.pred(2000, 3, 2, 2005)
    assert result is not None
    val = float(result) if not hasattr(result, "__iter__") else float(list(result)[0])
    assert val > 0, f"Expected positive price, got {val}"

safe_test("Regression pred returns positive number", t10_1)

subsection("10.2 Binary Classification Prediction (0 or 1)")

def t10_2():
    ml = make_ml("student_performance.csv")
    ml.col_to_feed(["study_hours_per_day", "attendance_pct",
                    "previous_score", "sleep_hours", "assignments_done"])
    ml.col_to_pred(["passed"])
    ml.method("random_forest_classifier")
    ml.split(70, 20, 10)
    ml.train(n_estimators=20)
    result = ml.pred(6.0, 80.0, 70.0, 7.0, 8)
    val = int(result) if not hasattr(result, "__iter__") else int(list(result)[0])
    assert val in [0, 1], f"Expected 0 or 1, got {val}"

safe_test("Binary classification pred returns 0 or 1", t10_2)

subsection("10.3 Multiclass Prediction (named class)")

def t10_3():
    ml = make_ml("flower_species.csv")
    ml.col_to_feed(["sepal_length_cm", "sepal_width_cm",
                    "petal_length_cm", "petal_width_cm"])
    ml.col_to_pred(["species"])
    ml.method("random_forest_classifier")
    ml.split(70, 20, 10)
    ml.train(n_estimators=20)
    result = ml.pred(5.1, 3.5, 1.4, 0.2)
    assert result is not None

safe_test("Multiclass pred returns a class value", t10_3)

subsection("10.4 Multi-Output Regression Prediction")

def t10_4():
    ml = make_ml("weather_forecast.csv")
    ml.col_to_feed(["month", "day_of_year", "humidity_pct",
                    "pressure_hpa", "wind_speed_kmh",
                    "cloud_cover_pct", "altitude_m"])
    ml.col_to_pred(["temperature_c", "rainfall_mm"])
    ml.method("random_forest_regressor")
    ml.split(70, 20, 10)
    ml.train(n_estimators=20)
    result = ml.pred(7, 200, 65.0, 1013.0, 15.0, 40.0, 100.0)
    assert result is not None

safe_test("Multi-output pred returns multiple values", t10_4)

subsection("10.5 Wrong Number of Inputs")

def t10_5():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft", "bedrooms", "bathrooms"])
    ml.col_to_pred(["price_usd"])
    ml.method("linear_regression")
    ml.split(70, 20, 10)
    ml.train()
    try:
        result = ml.pred(2000, 3)   # missing one input
        assert False, "Should raise ValueError"
    except ValueError:
        pass

safe_test("ValueError on wrong number of pred inputs", t10_5)

subsection("10.6 pred() Before Training")

def t10_6():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft", "bedrooms"])
    ml.col_to_pred(["price_usd"])
    try:
        ml.pred(2000, 3)
        assert False, "Should raise RuntimeError"
    except RuntimeError:
        pass

safe_test("RuntimeError pred before training", t10_6)

subsection("10.7 pred() with method= selector")

def t10_7():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft", "bedrooms", "bathrooms"])
    ml.col_to_pred(["price_usd"])
    ml.method(["linear_regression", "ridge"])
    ml.split(70, 20, 10)
    ml.train()
    result = ml.pred(2000, 3, 2, method="ridge")
    assert result is not None

safe_test("pred() with method= selector", t10_7)

subsection("10.8 pred() All Models Return Dict")

def t10_8():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft", "bedrooms", "bathrooms"])
    ml.col_to_pred(["price_usd"])
    ml.method(["linear_regression", "ridge", "lasso"])
    ml.split(70, 20, 10)
    ml.train()
    result = ml.pred(2000, 3, 2)
    # multiple models → returns dict
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert len(result) == 3

safe_test("pred() multiple models returns dict", t10_8)

# ════════════════════════════════════════════════════════════
# TEST BLOCK 11 — save() & load()
# ════════════════════════════════════════════════════════════
section("TEST BLOCK 11 — save() & load()")

subsection("11.1 Save Single Model")

def t11_1():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft", "bedrooms", "bathrooms"])
    ml.col_to_pred(["price_usd"])
    ml.method("linear_regression")
    ml.split(70, 20, 10)
    ml.train()
    ml.save(f"{MODELS}/test_single_reg")
    assert os.path.exists(f"{MODELS}/test_single_reg.allml"), "Save file not found"

safe_test("Save single regression model", t11_1)

subsection("11.2 Save Multiple Models")

def t11_2():
    ml = make_ml("student_performance.csv")
    ml.col_to_feed(["study_hours_per_day", "attendance_pct",
                    "previous_score", "sleep_hours"])
    ml.col_to_pred(["passed"])
    ml.method(["random_forest_classifier", "gradient_boosting_classifier"])
    ml.split(70, 20, 10)
    ml.train(n_estimators=20)
    ml.save(f"{MODELS}/test_multi_cls")
    # Should create 2 files (one per model)
    saved = [f for f in os.listdir(MODELS) if "test_multi_cls" in f and f.endswith(".allml")]
    assert len(saved) >= 1, f"Expected saved files, got {saved}"

safe_test("Save multiple classification models", t11_2)

subsection("11.3 Load Single Model")

def t11_3():
    ml2 = make_ml("house_prices.csv")
    ml2.load(f"{MODELS}/test_single_reg.allml")
    assert ml2._is_trained
    assert ml2._feed_cols  == ["area_sqft", "bedrooms", "bathrooms"]
    assert ml2._pred_cols  == ["price_usd"]
    assert ml2._task_type  == "regression"

safe_test("Load single regression model", t11_3)

subsection("11.4 Load → Predict")

def t11_4():
    ml2 = make_ml("house_prices.csv")
    ml2.load(f"{MODELS}/test_single_reg.allml")
    result = ml2.pred(2000, 3, 2)
    assert result is not None
    val = float(result) if not hasattr(result, "__iter__") else float(list(result)[0])
    assert val != 0

safe_test("Load model and predict", t11_4)

subsection("11.5 Load Non-Existent File")

def t11_5():
    ml2 = make_ml("house_prices.csv")
    try:
        ml2.load("nonexistent_model.allml")
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError:
        pass

safe_test("FileNotFoundError on bad load path", t11_5)

subsection("11.6 Save Classifier → Load → Classify")

def t11_6():
    ml = make_ml("student_performance.csv")
    ml.col_to_feed(["study_hours_per_day", "attendance_pct",
                    "previous_score", "sleep_hours", "assignments_done"])
    ml.col_to_pred(["passed"])
    ml.method("random_forest_classifier")
    ml.split(70, 20, 10)
    ml.train(n_estimators=20)
    ml.save(f"{MODELS}/test_cls_save")

    ml2 = make_ml("student_performance.csv")
    ml2.load(f"{MODELS}/test_cls_save.allml")
    result = ml2.pred(6.0, 80.0, 70.0, 7.0, 8)
    val = int(result) if not hasattr(result, "__iter__") else int(list(result)[0])
    assert val in [0, 1]

safe_test("Save classifier → load → predict 0/1", t11_6)

subsection("11.7 Meta JSON Created")

def t11_7():
    assert os.path.exists(f"{MODELS}/test_single_reg_meta.json"), \
        "Meta JSON not created"

safe_test("Meta JSON file created on save", t11_7)

# ════════════════════════════════════════════════════════════
# TEST BLOCK 12 — show() & show_graph()
# ════════════════════════════════════════════════════════════
section("TEST BLOCK 12 — show() & show_graph()")

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no window pops up)
import matplotlib.pyplot as plt

subsection("12.1 show() Doesn't Crash")

def t12_1():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft", "bedrooms", "bathrooms"])
    ml.col_to_pred(["price_usd"])
    ml.method("random_forest_regressor")
    ml.split(70, 20, 10)
    ml.train(n_estimators=20)
    ml.show()   # should not raise

safe_test("show() runs without error", t12_1)

subsection("12.2 Regression Graphs")

REGRESSION_GRAPHS = [
    "correlation_heatmap",
    "distribution",
    "boxplot",
    "feature_importance",
    "prediction_vs_actual",
    "residuals",
    "error_distribution",
    "learning_curve",
    "model_comparison",
    "pairplot",
]

def _setup_reg_ml():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft", "bedrooms", "bathrooms", "year_built"])
    ml.col_to_pred(["price_usd"])
    ml.method(["random_forest_regressor", "linear_regression"])
    ml.split(70, 20, 10)
    ml.train(n_estimators=20)
    return ml

_reg_ml = _setup_reg_ml()

for graph in REGRESSION_GRAPHS:
    def make_graph_test(g, ml_instance):
        def _test():
            plt.close("all")
            ml_instance.show_graph(g)
            plt.close("all")
        return _test
    safe_test(f"Regression graph: {graph}", make_graph_test(graph, _reg_ml))

subsection("12.3 Classification Graphs")

CLS_GRAPHS = [
    "confusion_matrix",
    "roc_curve",
    "precision_recall_curve",
    "feature_importance",
    "learning_curve",
    "model_comparison",
    "correlation_heatmap",
    "distribution",
]

def _setup_cls_ml():
    ml = make_ml("student_performance.csv")
    ml.col_to_feed(["study_hours_per_day", "attendance_pct",
                    "previous_score", "sleep_hours", "assignments_done"])
    ml.col_to_pred(["passed"])
    ml.method(["random_forest_classifier", "gradient_boosting_classifier"])
    ml.split(70, 20, 10)
    ml.train(n_estimators=20)
    return ml

_cls_ml = _setup_cls_ml()

for graph in CLS_GRAPHS:
    def make_cls_graph_test(g, ml_instance):
        def _test():
            plt.close("all")
            ml_instance.show_graph(g)
            plt.close("all")
        return _test
    safe_test(f"Classification graph: {graph}", make_cls_graph_test(graph, _cls_ml))

subsection("12.4 show_graph('all') on Regression")

def t12_all_reg():
    plt.close("all")
    _reg_ml.show_graph("all")
    plt.close("all")

safe_test("show_graph('all') regression", t12_all_reg)

subsection("12.5 show_graph('all') on Classification")

def t12_all_cls():
    plt.close("all")
    _cls_ml.show_graph("all")
    plt.close("all")

safe_test("show_graph('all') classification", t12_all_cls)

subsection("12.6 show_graph List of Multiple Graphs")

def t12_multi_graph():
    plt.close("all")
    _reg_ml.show_graph(["distribution", "boxplot", "correlation_heatmap"])
    plt.close("all")

safe_test("show_graph(['dist','box','heatmap']) list input", t12_multi_graph)

subsection("12.7 Unknown Graph Warning (no crash)")

def t12_unk():
    plt.close("all")
    _reg_ml.show_graph("totally_fake_graph")
    plt.close("all")

safe_test("Unknown graph type doesn't crash", t12_unk)

# ════════════════════════════════════════════════════════════
# TEST BLOCK 13 — EDGE CASES & STRESS TESTS
# ════════════════════════════════════════════════════════════
section("TEST BLOCK 13 — EDGE CASES & STRESS TESTS")

subsection("13.1 Train Without Split")

def t13_1():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft"])
    ml.col_to_pred(["price_usd"])
    ml.method("linear_regression")
    try:
        ml.train()
        assert False, "Should raise RuntimeError"
    except RuntimeError:
        pass

safe_test("RuntimeError train without split", t13_1)

subsection("13.2 Train Without method()")

def t13_2():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft"])
    ml.col_to_pred(["price_usd"])
    ml.split(70, 20, 10)
    try:
        ml.train()
        assert False, "Should raise RuntimeError"
    except RuntimeError:
        pass

safe_test("RuntimeError train without method", t13_2)

subsection("13.3 Save Without Training")

def t13_3():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft"])
    ml.col_to_pred(["price_usd"])
    try:
        ml.save("dummy_path")
        assert False, "Should raise RuntimeError"
    except RuntimeError:
        pass

safe_test("RuntimeError save without training", t13_3)

subsection("13.4 Small Dataset (50 rows)")

def t13_4():
    import pandas as pd, numpy as np
    small = pd.DataFrame({
        "x1": np.random.rand(50),
        "x2": np.random.rand(50),
        "y" : np.random.rand(50) * 100,
    })
    small.to_csv(f"{DATA}/tiny.csv", index=False)
    ml = AllML(f"{DATA}/tiny.csv")
    ml.col_to_feed(["x1", "x2"])
    ml.col_to_pred(["y"])
    ml.method("linear_regression")
    ml.split(70, 20, 10)
    ml.train()
    assert ml._is_trained
    result = ml.pred(0.5, 0.3)
    assert result is not None

safe_test("Small dataset (50 rows) end-to-end", t13_4)

subsection("13.5 Dataset with Categorical Features")

def t13_5():
    ml = make_ml("employee_salary.csv")   # has 'department' categorical
    ml.col_to_feed(["age", "years_experience", "education_level",
                    "department", "performance_score"])
    ml.col_to_pred(["annual_salary"])
    ml.method("random_forest_regressor")
    ml.split(70, 20, 10)
    ml.train(n_estimators=20)
    assert ml._is_trained
    result = ml.pred(35, 10, 3, "Engineering", 4)
    assert result is not None

safe_test("Categorical features auto-encoded", t13_5)

subsection("13.6 method('all') Regression — All Models Compared")

def t13_6():
    ml = make_ml("energy_consumption.csv")
    ml.col_to_feed(["temperature_c", "num_occupants",
                    "building_area_m2", "insulation_grade"])
    ml.col_to_pred(["energy_kwh"])
    ml.method("all")
    ml.split(70, 20, 10)
    ml.train(n_estimators=10, epochs=30)
    plt.close("all")
    ml.show_graph("model_comparison")
    plt.close("all")
    assert len(ml._models) >= 5

safe_test("method('all') regression + model_comparison graph", t13_6)

subsection("13.7 Chained API calls")

def t13_7():
    result = (
        AllML(f"{DATA}/house_prices.csv")
            .col_to_feed(["area_sqft", "bedrooms", "bathrooms"])
            .col_to_pred(["price_usd"])
            .method("linear_regression")
            .split(70, 20, 10)
            .train()
    )
    assert result._is_trained

safe_test("Chained API: AllML().feed().pred().method().split().train()", t13_7)

subsection("13.8 Hyperparameter Variations")

def t13_8():
    configs = [
        dict(epochs=10,  n_estimators=5,  learning_rate=0.001),
        dict(epochs=50,  n_estimators=20, learning_rate=0.05),
        dict(epochs=100, n_estimators=50, learning_rate=0.1),
    ]
    for cfg in configs:
        ml = make_ml("student_performance.csv")
        ml.col_to_feed(["study_hours_per_day", "attendance_pct", "previous_score"])
        ml.col_to_pred(["passed"])
        ml.method("gradient_boosting_classifier")
        ml.split(70, 20, 10)
        ml.train(**cfg)
        assert ml._is_trained

safe_test("Multiple hyperparameter configurations", t13_8)

subsection("13.9 repr() doesn't crash")

def t13_9():
    ml = make_ml("house_prices.csv")
    ml.col_to_feed(["area_sqft"])
    ml.col_to_pred(["price_usd"])
    r = repr(ml)
    assert isinstance(r, str)
    assert "AllML" in r

safe_test("repr(ml) returns correct string", t13_9)

subsection("13.10 Save → Load → Graph (no crash)")

def t13_10():
    ml = make_ml("heart_disease.csv")
    ml.col_to_feed(["age", "sex", "cholesterol", "blood_pressure"])
    ml.col_to_pred(["heart_disease"])
    ml.method("gradient_boosting_classifier")
    ml.split(70, 20, 10)
    ml.train(n_estimators=20)
    ml.save(f"{MODELS}/test_reload_graph")

    ml2 = make_ml("heart_disease.csv")
    ml2.load(f"{MODELS}/test_reload_graph.allml")
    ml2.show()

safe_test("Save → Load → show() without crash", t13_10)

# ════════════════════════════════════════════════════════════
# TEST BLOCK 14 — FULL PIPELINE END-TO-END TESTS
# ════════════════════════════════════════════════════════════
section("TEST BLOCK 14 — FULL PIPELINE END-TO-END")

subsection("14.1 Complete House Price Pipeline")

def t14_1():
    ml = AllML(f"{DATA}/house_prices.csv")
    ml.col_to_feed(["area_sqft", "bedrooms", "bathrooms",
                    "year_built", "garage_cars", "has_pool"])
    ml.col_to_pred(["price_usd"])
    ml.method(["random_forest_regressor",
               "gradient_boosting_regressor",
               "linear_regression"])
    ml.split(70, 20, 10)
    ml.train(n_estimators=30, epochs=100)
    ml.show()
    plt.close("all")
    ml.show_graph(["prediction_vs_actual",
                   "feature_importance",
                   "model_comparison"])
    plt.close("all")
    ml.save(f"{MODELS}/e2e_house")
    pred = ml.pred(2500, 4, 2, 2010, 2, 1)
    assert pred is not None
    info(f"  House price prediction: {pred}")

safe_test("E2E: House price complete pipeline", t14_1)

subsection("14.2 Complete Student Pass/Fail Pipeline")

def t14_2():
    ml = AllML(f"{DATA}/student_performance.csv")
    ml.col_to_feed(["study_hours_per_day", "attendance_pct",
                    "previous_score", "sleep_hours",
                    "assignments_done", "extra_classes"])
    ml.col_to_pred(["passed"])
    ml.method(["random_forest_classifier",
               "gradient_boosting_classifier",
               "logistic_regression",
               "svc"])
    ml.split(70, 20, 10)
    ml.train(n_estimators=30, epochs=100)
    ml.show()
    plt.close("all")
    ml.show_graph(["confusion_matrix", "roc_curve",
                   "feature_importance", "model_comparison"])
    plt.close("all")
    ml.save(f"{MODELS}/e2e_student")
    pred = ml.pred(7.0, 90.0, 80.0, 8.0, 10, 1)
    info(f"  Pass prediction: {pred}")

safe_test("E2E: Student pass/fail complete pipeline", t14_2)

subsection("14.3 Complete Flower Species Pipeline")

def t14_3():
    ml = AllML(f"{DATA}/flower_species.csv")
    ml.col_to_feed(["sepal_length_cm", "sepal_width_cm",
                    "petal_length_cm", "petal_width_cm"])
    ml.col_to_pred(["species"])
    ml.method(["random_forest_classifier",
               "gradient_boosting_classifier",
               "knn_classifier"])
    ml.split(70, 20, 10)
    ml.train(n_estimators=30)
    ml.show()
    plt.close("all")
    ml.show_graph(["confusion_matrix",
                   "feature_importance",
                   "model_comparison"])
    plt.close("all")
    ml.save(f"{MODELS}/e2e_flower")
    pred = ml.pred(5.1, 3.5, 1.4, 0.2)
    info(f"  Species prediction: {pred}")

safe_test("E2E: Flower species multiclass pipeline", t14_3)

subsection("14.4 Complete Weather Multi-Output Pipeline")

def t14_4():
    ml = AllML(f"{DATA}/weather_forecast.csv")
    ml.col_to_feed(["month", "day_of_year", "humidity_pct",
                    "pressure_hpa", "wind_speed_kmh",
                    "cloud_cover_pct", "altitude_m", "latitude"])
    ml.col_to_pred(["temperature_c", "humidity_out_pct", "rainfall_mm"])
    ml.method(["random_forest_regressor",
               "gradient_boosting_regressor"])
    ml.split(70, 20, 10)
    ml.train(n_estimators=30)
    ml.show()
    plt.close("all")
    ml.show_graph(["model_comparison", "correlation_heatmap"])
    plt.close("all")
    ml.save(f"{MODELS}/e2e_weather")
    pred = ml.pred(7, 200, 65.0, 1013.0, 15.0, 40.0, 100.0, 28.0)
    info(f"  Weather prediction: {pred}")

safe_test("E2E: Weather multi-output regression pipeline", t14_4)

subsection("14.5 Complete Loan Approval Pipeline")

def t14_5():
    ml = AllML(f"{DATA}/loan_approval.csv")
    ml.col_to_feed(["annual_income", "loan_amount", "credit_score",
                    "loan_term_years", "employment_years",
                    "existing_debt_pct", "num_defaults"])
    ml.col_to_pred(["approved"])
    ml.method(["random_forest_classifier",
               "gradient_boosting_classifier",
               "logistic_regression"])
    ml.split(70, 20, 10)
    ml.train(n_estimators=30)
    ml.show()
    plt.close("all")
    ml.show_graph(["confusion_matrix", "roc_curve", "model_comparison"])
    plt.close("all")
    ml.save(f"{MODELS}/e2e_loan")
    pred = ml.pred(75000, 200000, 720, 15, 5, 20.0, 0)
    info(f"  Loan approved: {pred}")

safe_test("E2E: Loan approval complete pipeline", t14_5)

subsection("14.6 Complete Car Specs Multi-Output Pipeline")

def t14_6():
    ml = AllML(f"{DATA}/car_specs.csv")
    ml.col_to_feed(["engine_cc", "horsepower", "torque_nm",
                    "weight_kg", "cylinders", "car_age_years",
                    "has_turbo", "is_electric", "brand_tier"])
    ml.col_to_pred(["price_usd", "fuel_economy_mpg"])
    ml.method(["random_forest_regressor", "decision_tree_regressor"])
    ml.split(70, 20, 10)
    ml.train(n_estimators=25)
    ml.show()
    plt.close("all")
    ml.show_graph("model_comparison")
    plt.close("all")
    ml.save(f"{MODELS}/e2e_car")
    pred = ml.pred(2000, 180, 320, 1500, 4, 3, 1, 0, 2)
    info(f"  Car pred (price, fuel): {pred}")

safe_test("E2E: Car specs multi-output pipeline", t14_6)

# ════════════════════════════════════════════════════════════
# TEST BLOCK 15 — DATASET INTEGRITY CHECKS
# ════════════════════════════════════════════════════════════
section("TEST BLOCK 15 — DATASET INTEGRITY CHECKS")

import pandas as pd

DATASETS = {
    "house_prices.csv"       : {"rows": 900,  "col": "price_usd"},
    "student_performance.csv": {"rows": 1000, "col": "passed"},
    "flower_species.csv"     : {"rows": 150,  "col": "species"},
    "employee_salary.csv"    : {"rows": 1000, "col": "annual_salary"},
    "heart_disease.csv"      : {"rows": 900,  "col": "heart_disease"},
    "weather_forecast.csv"   : {"rows": 1500, "col": "temperature_c"},
    "wine_quality.csv"       : {"rows": 1200, "col": "quality"},
    "car_specs.csv"          : {"rows": 1000, "col": "price_usd"},
    "loan_approval.csv"      : {"rows": 1000, "col": "approved"},
    "energy_consumption.csv" : {"rows": 1500, "col": "energy_kwh"},
}

for fname, specs in DATASETS.items():
    def make_data_test(fn, sp):
        def _test():
            df = pd.read_csv(f"{DATA}/{fn}")
            assert df.shape[0] >= sp["rows"],  f"Rows: {df.shape[0]} < {sp['rows']}"
            assert sp["col"] in df.columns,     f"Col '{sp['col']}' missing"
            missing_rate = df.isnull().mean().max()
            assert missing_rate < 0.5,          f"Too many nulls: {missing_rate:.2%}"
        return _test
    safe_test(f"Dataset integrity: {fname}", make_data_test(fname, specs))

# ════════════════════════════════════════════════════════════
# FINAL REPORT
# ════════════════════════════════════════════════════════════
elapsed = time.time() - start_time
result_box(tracker.passed, tracker.failed, tracker.skipped, tracker.total, elapsed)

if tracker.errors:
    print(f"{C.R}{C.B}Failed Tests Detail:{C.E}")
    for name, err in tracker.errors:
        print(f"  {C.R}• {name}{C.E}")
        print(f"    {C.Y}{err}{C.E}")

print(f"\n{C.CY}{C.B}📁 Saved models in: {MODELS}/{C.E}")
print(f"{C.CY}{C.B}📁 Test datasets in: {DATA}/{C.E}\n")
