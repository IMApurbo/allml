# generate_test_datasets.py
# Generates multiple logical CSV datasets for testing AllML library

import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_classification, make_regression,
    make_blobs, load_iris, load_wine, load_breast_cancer
)
import os

np.random.seed(42)

# ============================================================
# Color output
# ============================================================
class C:
    G = '\033[92m'
    Y = '\033[93m'
    C = '\033[96m'
    R = '\033[91m'
    B = '\033[1m'
    E = '\033[0m'
    M = '\033[35m'

def title(text):
    print(f"\n{C.C}{C.B}{'='*60}{C.E}")
    print(f"{C.C}{C.B}  {text}{C.E}")
    print(f"{C.C}{C.B}{'='*60}{C.E}")

def ok(text):
    print(f"  {C.G}✅ {text}{C.E}")

def info(text):
    print(f"  {C.Y}ℹ️  {text}{C.E}")

# ============================================================
# Output folder
# ============================================================
OUT = "test_datasets"
os.makedirs(OUT, exist_ok=True)

datasets_info = []

# ============================================================
# 1. HOUSE PRICES  →  Regression (predict decimal)
# ============================================================
title("1. House Prices Dataset  →  Regression")

n = 1000
area          = np.random.randint(500,  5000, n)
bedrooms      = np.random.randint(1,    6,    n)
bathrooms     = np.random.randint(1,    4,    n)
year_built    = np.random.randint(1950, 2023, n)
garage        = np.random.randint(0,    4,    n)
floors        = np.random.randint(1,    4,    n)
lot_size      = np.random.randint(1000, 20000, n)
distance_city = np.random.uniform(1,    50,   n)
has_pool      = np.random.randint(0,    2,    n)
condition     = np.random.choice(["Excellent","Good","Fair","Poor"], n)
condition_map = {"Excellent": 4, "Good": 3, "Fair": 2, "Poor": 1}
condition_num = np.array([condition_map[c] for c in condition])

# Logical price formula
price = (
      area          * 120
    + bedrooms      * 15000
    + bathrooms     * 10000
    + (year_built - 1950) * 800
    + garage        * 8000
    + floors        * 5000
    + lot_size      * 2
    - distance_city * 1200
    + has_pool      * 25000
    + condition_num * 12000
    + np.random.normal(0, 15000, n)        # noise
)
price = np.clip(price, 50000, 3000000)

house_df = pd.DataFrame({
    "area_sqft"       : area,
    "bedrooms"        : bedrooms,
    "bathrooms"       : bathrooms,
    "year_built"      : year_built,
    "garage_cars"     : garage,
    "floors"          : floors,
    "lot_size_sqft"   : lot_size,
    "distance_to_city": np.round(distance_city, 2),
    "has_pool"        : has_pool,
    "condition"       : condition,
    "price_usd"       : np.round(price, 2),
})

path = f"{OUT}/house_prices.csv"
house_df.to_csv(path, index=False)
ok(f"Saved → {path}  |  Shape: {house_df.shape}")
info("Task: REGRESSION  |  Predict: price_usd")
info("Feed: area_sqft, bedrooms, bathrooms, year_built, garage_cars, floors, lot_size_sqft, distance_to_city, has_pool, condition")
datasets_info.append({
    "file"   : path,
    "task"   : "Regression",
    "feed"   : ["area_sqft","bedrooms","bathrooms","year_built","garage_cars"],
    "predict": ["price_usd"],
})

# ============================================================
# 2. STUDENT PERFORMANCE  →  Binary Classification (pass/fail)
# ============================================================
title("2. Student Performance Dataset  →  Binary Classification (0/1)")

n = 1200
study_hours   = np.random.uniform(0,   10, n)
attendance    = np.random.uniform(40, 100, n)
prev_score    = np.random.uniform(30, 100, n)
sleep_hours   = np.random.uniform(3,   10, n)
assignments   = np.random.randint(0,   11, n)
extra_class   = np.random.randint(0,    2, n)
parent_edu    = np.random.randint(1,    5, n)
internet      = np.random.randint(0,    2, n)

# Logical score
score = (
      study_hours * 4.5
    + attendance  * 0.3
    + prev_score  * 0.4
    + sleep_hours * 1.2
    + assignments * 1.5
    + extra_class * 5
    + parent_edu  * 1.0
    + internet    * 2.0
    + np.random.normal(0, 5, n)
)
passed = (score >= 60).astype(int)

student_df = pd.DataFrame({
    "study_hours_per_day" : np.round(study_hours, 2),
    "attendance_pct"      : np.round(attendance,  2),
    "previous_score"      : np.round(prev_score,  2),
    "sleep_hours"         : np.round(sleep_hours, 2),
    "assignments_done"    : assignments,
    "extra_classes"       : extra_class,
    "parent_education"    : parent_edu,
    "has_internet"        : internet,
    "passed"              : passed,
})

path = f"{OUT}/student_performance.csv"
student_df.to_csv(path, index=False)
ok(f"Saved → {path}  |  Shape: {student_df.shape}")
info("Task: BINARY CLASSIFICATION  |  Predict: passed (0 or 1)")
info("Feed: study_hours_per_day, attendance_pct, previous_score, sleep_hours, assignments_done")
datasets_info.append({
    "file"   : path,
    "task"   : "Binary Classification",
    "feed"   : ["study_hours_per_day","attendance_pct","previous_score","sleep_hours","assignments_done"],
    "predict": ["passed"],
})

# ============================================================
# 3. IRIS EXTENDED  →  Multi-class Classification
# ============================================================
title("3. Flower Species Dataset  →  Multi-class Classification (3 classes)")

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=[
    "sepal_length_cm","sepal_width_cm",
    "petal_length_cm","petal_width_cm"
])
iris_df["petal_area"]    = np.round(iris_df["petal_length_cm"] * iris_df["petal_width_cm"], 3)
iris_df["sepal_area"]    = np.round(iris_df["sepal_length_cm"] * iris_df["sepal_width_cm"], 3)
iris_df["length_ratio"]  = np.round(iris_df["petal_length_cm"] / iris_df["sepal_length_cm"], 3)
species_names            = {0:"Setosa", 1:"Versicolor", 2:"Virginica"}
iris_df["species"]       = [species_names[i] for i in iris.target]

# Add noise samples
extra_n  = 50
extra_df = pd.DataFrame({
    "sepal_length_cm": np.random.uniform(4.3, 7.9, extra_n),
    "sepal_width_cm" : np.random.uniform(2.0, 4.4, extra_n),
    "petal_length_cm": np.random.uniform(1.0, 6.9, extra_n),
    "petal_width_cm" : np.random.uniform(0.1, 2.5, extra_n),
})
extra_df["petal_area"]   = np.round(extra_df["petal_length_cm"] * extra_df["petal_width_cm"], 3)
extra_df["sepal_area"]   = np.round(extra_df["sepal_length_cm"] * extra_df["sepal_width_cm"], 3)
extra_df["length_ratio"] = np.round(extra_df["petal_length_cm"] / extra_df["sepal_length_cm"], 3)
extra_df["species"]      = np.random.choice(["Setosa","Versicolor","Virginica"], extra_n)
iris_df = pd.concat([iris_df, extra_df], ignore_index=True)

path = f"{OUT}/flower_species.csv"
iris_df.to_csv(path, index=False)
ok(f"Saved → {path}  |  Shape: {iris_df.shape}")
info("Task: MULTI-CLASS CLASSIFICATION  |  Predict: species (Setosa/Versicolor/Virginica)")
info("Feed: sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm")
datasets_info.append({
    "file"   : path,
    "task"   : "Multiclass Classification",
    "feed"   : ["sepal_length_cm","sepal_width_cm","petal_length_cm","petal_width_cm"],
    "predict": ["species"],
})

# ============================================================
# 4. EMPLOYEE SALARY  →  Regression (predict salary)
# ============================================================
title("4. Employee Salary Dataset  →  Regression")

n = 1500
experience    = np.random.randint(0,   30, n)
age           = experience + np.random.randint(22, 30, n)
age           = np.clip(age, 22, 65)
education     = np.random.randint(1,    5, n)   # 1=HS, 2=BS, 3=MS, 4=PhD
dept_num      = np.random.randint(0,    5, n)   # 0-4 departments
performance   = np.random.randint(1,    6, n)   # 1-5 rating
certifications= np.random.randint(0,    8, n)
is_manager    = np.random.randint(0,    2, n)
remote_pct    = np.random.randint(0,  101, n)
dept_names    = ["Engineering","Marketing","HR","Finance","Operations"]
department    = [dept_names[d] for d in dept_num]

salary = (
      experience     * 2500
    + education      * 8000
    + performance    * 3500
    + certifications * 1200
    + is_manager     * 20000
    + age            * 300
    + dept_num       * 4000
    - remote_pct     * 50
    + 30000
    + np.random.normal(0, 5000, n)
)
salary = np.clip(salary, 25000, 350000)

emp_df = pd.DataFrame({
    "age"              : age,
    "years_experience" : experience,
    "education_level"  : education,
    "department"       : department,
    "performance_score": performance,
    "certifications"   : certifications,
    "is_manager"       : is_manager,
    "remote_work_pct"  : remote_pct,
    "annual_salary"    : np.round(salary, 2),
})

path = f"{OUT}/employee_salary.csv"
emp_df.to_csv(path, index=False)
ok(f"Saved → {path}  |  Shape: {emp_df.shape}")
info("Task: REGRESSION  |  Predict: annual_salary")
info("Feed: age, years_experience, education_level, performance_score, certifications, is_manager")
datasets_info.append({
    "file"   : path,
    "task"   : "Regression",
    "feed"   : ["age","years_experience","education_level","performance_score","certifications","is_manager"],
    "predict": ["annual_salary"],
})

# ============================================================
# 5. HEART DISEASE  →  Binary Classification (disease: yes/no)
# ============================================================
title("5. Heart Disease Dataset  →  Binary Classification")

n = 1100
age_h         = np.random.randint(29, 77, n)
sex           = np.random.randint(0,  2,  n)     # 0=F, 1=M
cholesterol   = np.random.randint(150, 400, n)
blood_pressure= np.random.randint(90,  200, n)
heart_rate    = np.random.randint(60,  200, n)
blood_sugar   = np.random.randint(0,   2,  n)    # >120 fasting
chest_pain    = np.random.randint(0,   4,  n)    # 0-3 type
smoking       = np.random.randint(0,   2,  n)
exercise      = np.random.randint(0,   2,  n)    # angina on exercise
bmi           = np.random.uniform(18, 40,  n)

risk_score = (
      (age_h - 29)      * 0.05
    + cholesterol        * 0.03
    + blood_pressure     * 0.04
    - heart_rate         * 0.01
    + blood_sugar        * 1.5
    + chest_pain         * 1.2
    + smoking            * 2.0
    + exercise           * 1.8
    + bmi                * 0.08
    + sex                * 0.5
    + np.random.normal(0, 2, n)
)
threshold     = np.percentile(risk_score, 55)
heart_disease = (risk_score >= threshold).astype(int)

heart_df = pd.DataFrame({
    "age"                : age_h,
    "sex"                : sex,
    "cholesterol"        : cholesterol,
    "blood_pressure"     : blood_pressure,
    "max_heart_rate"     : heart_rate,
    "fasting_blood_sugar": blood_sugar,
    "chest_pain_type"    : chest_pain,
    "smoking"            : smoking,
    "exercise_angina"    : exercise,
    "bmi"                : np.round(bmi, 2),
    "heart_disease"      : heart_disease,
})

path = f"{OUT}/heart_disease.csv"
heart_df.to_csv(path, index=False)
ok(f"Saved → {path}  |  Shape: {heart_df.shape}")
info("Task: BINARY CLASSIFICATION  |  Predict: heart_disease (0 or 1)")
info("Feed: age, sex, cholesterol, blood_pressure, max_heart_rate, fasting_blood_sugar, chest_pain_type, smoking")
datasets_info.append({
    "file"   : path,
    "task"   : "Binary Classification",
    "feed"   : ["age","sex","cholesterol","blood_pressure","max_heart_rate","fasting_blood_sugar","chest_pain_type","smoking"],
    "predict": ["heart_disease"],
})

# ============================================================
# 6. WEATHER FORECAST  →  Multi-Output Regression
# ============================================================
title("6. Weather Forecast Dataset  →  Multi-Output Regression")

n = 2000
month         = np.random.randint(1,  13, n)
day_of_year   = np.random.randint(1, 366, n)
humidity_in   = np.random.uniform(20, 90, n)
pressure_in   = np.random.uniform(980, 1050, n)
wind_speed    = np.random.uniform(0,  80,  n)
cloud_cover   = np.random.uniform(0,  100, n)
altitude      = np.random.uniform(0,  3000, n)
latitude      = np.random.uniform(-60, 60, n)

# Temperature logic
temp_out = (
    25
    - np.abs(latitude)   * 0.4
    - altitude           * 0.006
    + np.sin(day_of_year / 365 * 2 * np.pi) * 12
    - cloud_cover        * 0.05
    + np.random.normal(0, 3, n)
)

# Humidity out
humidity_out = (
    humidity_in
    + cloud_cover   * 0.2
    - wind_speed    * 0.1
    + np.random.normal(0, 4, n)
)
humidity_out = np.clip(humidity_out, 0, 100)

# Rainfall mm
rainfall = np.where(
    humidity_out > 70,
    (humidity_out - 70) * 1.2 + cloud_cover * 0.3 + np.random.exponential(2, n),
    np.random.exponential(0.5, n)
)
rainfall = np.clip(rainfall, 0, 200)

weather_df = pd.DataFrame({
    "month"           : month,
    "day_of_year"     : day_of_year,
    "humidity_pct"    : np.round(humidity_in,  2),
    "pressure_hpa"    : np.round(pressure_in,  2),
    "wind_speed_kmh"  : np.round(wind_speed,   2),
    "cloud_cover_pct" : np.round(cloud_cover,  2),
    "altitude_m"      : np.round(altitude,     2),
    "latitude"        : np.round(latitude,     2),
    "temperature_c"   : np.round(temp_out,     2),
    "humidity_out_pct": np.round(humidity_out, 2),
    "rainfall_mm"     : np.round(rainfall,     2),
})

path = f"{OUT}/weather_forecast.csv"
weather_df.to_csv(path, index=False)
ok(f"Saved → {path}  |  Shape: {weather_df.shape}")
info("Task: MULTI-OUTPUT REGRESSION  |  Predict: temperature_c, humidity_out_pct, rainfall_mm")
info("Feed: month, day_of_year, humidity_pct, pressure_hpa, wind_speed_kmh, cloud_cover_pct, altitude_m, latitude")
datasets_info.append({
    "file"   : path,
    "task"   : "Multi-Output Regression",
    "feed"   : ["month","day_of_year","humidity_pct","pressure_hpa","wind_speed_kmh","cloud_cover_pct","altitude_m","latitude"],
    "predict": ["temperature_c","humidity_out_pct","rainfall_mm"],
})

# ============================================================
# 7. WINE QUALITY  →  Multi-class Classification (quality 3-8)
# ============================================================
title("7. Wine Quality Dataset  →  Multi-class Classification")

n = 1600
fixed_acid    = np.random.uniform(4,   16,  n)
volatile_acid = np.random.uniform(0.1, 1.6, n)
citric_acid   = np.random.uniform(0,   1,   n)
residual_sugar= np.random.uniform(0.9, 15,  n)
chlorides     = np.random.uniform(0.01,0.6, n)
free_so2      = np.random.uniform(1,   72,  n)
total_so2     = np.random.uniform(6,  289,  n)
density       = np.random.uniform(0.990, 1.004, n)
ph            = np.random.uniform(2.7, 4.0, n)
sulphates     = np.random.uniform(0.3, 2.0, n)
alcohol       = np.random.uniform(8,   15,  n)

quality_score = (
      alcohol        * 0.5
    - volatile_acid  * 2.0
    + citric_acid    * 1.5
    - chlorides      * 3.0
    + sulphates      * 1.2
    - ph             * 0.5
    + fixed_acid     * 0.1
    + np.random.normal(0, 0.8, n)
)
# Map to quality 3-8
qs_norm  = (quality_score - quality_score.min()) / (quality_score.max() - quality_score.min())
quality  = (qs_norm * 5 + 3).astype(int)
quality  = np.clip(quality, 3, 8)

wine_df = pd.DataFrame({
    "fixed_acidity"     : np.round(fixed_acid,     2),
    "volatile_acidity"  : np.round(volatile_acid,  2),
    "citric_acid"       : np.round(citric_acid,    2),
    "residual_sugar"    : np.round(residual_sugar,  2),
    "chlorides"         : np.round(chlorides,       3),
    "free_sulfur_dioxide": np.round(free_so2,       1),
    "total_sulfur_dioxide": np.round(total_so2,    1),
    "density"           : np.round(density,         4),
    "pH"                : np.round(ph,              2),
    "sulphates"         : np.round(sulphates,       2),
    "alcohol_pct"       : np.round(alcohol,         2),
    "quality"           : quality,
})

path = f"{OUT}/wine_quality.csv"
wine_df.to_csv(path, index=False)
ok(f"Saved → {path}  |  Shape: {wine_df.shape}")
info("Task: MULTI-CLASS CLASSIFICATION  |  Predict: quality (3-8)")
info("Feed: fixed_acidity, volatile_acidity, citric_acid, chlorides, alcohol_pct, sulphates, pH")
datasets_info.append({
    "file"   : path,
    "task"   : "Multiclass Classification",
    "feed"   : ["fixed_acidity","volatile_acidity","citric_acid","chlorides","alcohol_pct","sulphates","pH"],
    "predict": ["quality"],
})

# ============================================================
# 8. CAR SPECS  →  Multi-Output Regression (price + fuel)
# ============================================================
title("8. Car Specs Dataset  →  Multi-Output Regression (price + fuel efficiency)")

n = 1400
engine_cc     = np.random.randint(800,  5000, n)
horsepower    = np.random.randint(60,   600,  n)
torque        = np.random.randint(80,   800,  n)
weight_kg     = np.random.randint(800,  3500, n)
cylinders     = np.random.choice([3, 4, 6, 8, 12], n)
car_age       = np.random.randint(0,    20,   n)
mileage_km    = car_age * np.random.randint(5000, 25000, n)
turbo         = np.random.randint(0,    2,    n)
electric      = np.random.randint(0,    2,    n)
brand_tier    = np.random.randint(1,    5,    n)  # 1=budget … 4=luxury

car_price = (
      horsepower  * 350
    + engine_cc   * 20
    + torque      * 80
    - weight_kg   * 10
    + brand_tier  * 15000
    + turbo       * 8000
    + electric    * 20000
    - car_age     * 1500
    - mileage_km  * 0.002
    + 15000
    + np.random.normal(0, 8000, n)
)
car_price = np.clip(car_price, 3000, 500000)

fuel_economy = (
    50
    - horsepower * 0.05
    - engine_cc  * 0.003
    - weight_kg  * 0.005
    + electric   * 25
    + turbo      * 2
    + np.random.normal(0, 2, n)
)
fuel_economy = np.clip(fuel_economy, 5, 80)

car_df = pd.DataFrame({
    "engine_cc"      : engine_cc,
    "horsepower"     : horsepower,
    "torque_nm"      : torque,
    "weight_kg"      : weight_kg,
    "cylinders"      : cylinders,
    "car_age_years"  : car_age,
    "mileage_km"     : mileage_km,
    "has_turbo"      : turbo,
    "is_electric"    : electric,
    "brand_tier"     : brand_tier,
    "price_usd"      : np.round(car_price,  2),
    "fuel_economy_mpg": np.round(fuel_economy, 2),
})

path = f"{OUT}/car_specs.csv"
car_df.to_csv(path, index=False)
ok(f"Saved → {path}  |  Shape: {car_df.shape}")
info("Task: MULTI-OUTPUT REGRESSION  |  Predict: price_usd, fuel_economy_mpg")
info("Feed: engine_cc, horsepower, torque_nm, weight_kg, cylinders, car_age_years, has_turbo, is_electric, brand_tier")
datasets_info.append({
    "file"   : path,
    "task"   : "Multi-Output Regression",
    "feed"   : ["engine_cc","horsepower","torque_nm","weight_kg","cylinders","car_age_years","has_turbo","is_electric","brand_tier"],
    "predict": ["price_usd","fuel_economy_mpg"],
})

# ============================================================
# 9. LOAN APPROVAL  →  Binary Classification (approved: 0/1)
# ============================================================
title("9. Loan Approval Dataset  →  Binary Classification")

n = 1300
income        = np.random.randint(20000,  200000, n)
loan_amount   = np.random.randint(5000,   500000, n)
credit_score  = np.random.randint(300,    850,    n)
loan_term_yr  = np.random.choice([5, 10, 15, 20, 30], n)
employment_yr = np.random.randint(0,  30, n)
debt_ratio    = np.random.uniform(0,  60, n)     # existing debt %
num_defaults  = np.random.randint(0,   5, n)
assets        = np.random.randint(0,  500000, n)
married       = np.random.randint(0,   2, n)
dependents    = np.random.randint(0,   5, n)
education_l   = np.random.randint(1,   5, n)    # 1=HS … 4=PhD
loan_purpose  = np.random.choice(["Home","Car","Business","Education","Personal"], n)
purpose_score = {"Home": 1.5, "Car": 1.2, "Business": 0.8, "Education": 1.3, "Personal": 0.5}
purpose_num   = np.array([purpose_score[p] for p in loan_purpose])

approval_score = (
      credit_score    * 0.08
    + income          * 0.00005
    - loan_amount     * 0.000015
    + employment_yr   * 0.3
    - debt_ratio      * 0.15
    - num_defaults    * 3.0
    + assets          * 0.000005
    + married         * 1.0
    - dependents      * 0.3
    + education_l     * 0.4
    + purpose_num     * 1.0
    + np.random.normal(0, 2, n)
)

thresh   = np.percentile(approval_score, 40)
approved = (approval_score >= thresh).astype(int)

loan_df = pd.DataFrame({
    "annual_income"    : income,
    "loan_amount"      : loan_amount,
    "credit_score"     : credit_score,
    "loan_term_years"  : loan_term_yr,
    "employment_years" : employment_yr,
    "existing_debt_pct": np.round(debt_ratio, 2),
    "num_defaults"     : num_defaults,
    "assets_value"     : assets,
    "married"          : married,
    "dependents"       : dependents,
    "education_level"  : education_l,
    "loan_purpose"     : loan_purpose,
    "approved"         : approved,
})

path = f"{OUT}/loan_approval.csv"
loan_df.to_csv(path, index=False)
ok(f"Saved → {path}  |  Shape: {loan_df.shape}")
info("Task: BINARY CLASSIFICATION  |  Predict: approved (0 or 1)")
info("Feed: annual_income, loan_amount, credit_score, loan_term_years, employment_years, existing_debt_pct, num_defaults")
datasets_info.append({
    "file"   : path,
    "task"   : "Binary Classification",
    "feed"   : ["annual_income","loan_amount","credit_score","loan_term_years","employment_years","existing_debt_pct","num_defaults"],
    "predict": ["approved"],
})

# ============================================================
# 10. ENERGY CONSUMPTION  →  Regression
# ============================================================
title("10. Energy Consumption Dataset  →  Regression")

n = 1800
temp_c        = np.random.uniform(-10, 45, n)
humidity_e    = np.random.uniform(10,  95, n)
occupants     = np.random.randint(0,   20, n)
building_area = np.random.randint(50, 5000, n)
hour_of_day   = np.random.randint(0,   24, n)
day_of_week   = np.random.randint(0,    7, n)
appliances    = np.random.randint(1,   50, n)
solar_panels  = np.random.randint(0,    2, n)
insulation    = np.random.randint(1,    5, n)  # 1=poor … 5=excellent
building_age  = np.random.randint(0,   60, n)

# Heating/Cooling demand
heating = np.where(temp_c < 18, (18 - temp_c) * 2.0, 0)
cooling = np.where(temp_c > 25, (temp_c - 25) * 2.5, 0)

energy_kwh = (
      heating        * building_area * 0.01
    + cooling        * building_area * 0.012
    + occupants      * 0.5
    + appliances     * 0.3
    + building_area  * 0.02
    - insulation     * 5.0
    - solar_panels   * 15.0
    + building_age   * 0.2
    + (hour_of_day > 17).astype(int) * 8
    + np.random.normal(0, 5, n)
)
energy_kwh = np.clip(energy_kwh, 1, 2000)

energy_df = pd.DataFrame({
    "temperature_c"   : np.round(temp_c,     2),
    "humidity_pct"    : np.round(humidity_e,  2),
    "num_occupants"   : occupants,
    "building_area_m2": building_area,
    "hour_of_day"     : hour_of_day,
    "day_of_week"     : day_of_week,
    "num_appliances"  : appliances,
    "has_solar_panels": solar_panels,
    "insulation_grade": insulation,
    "building_age_yrs": building_age,
    "energy_kwh"      : np.round(energy_kwh, 3),
})

path = f"{OUT}/energy_consumption.csv"
energy_df.to_csv(path, index=False)
ok(f"Saved → {path}  |  Shape: {energy_df.shape}")
info("Task: REGRESSION  |  Predict: energy_kwh")
info("Feed: temperature_c, humidity_pct, num_occupants, building_area_m2, hour_of_day, num_appliances, insulation_grade")
datasets_info.append({
    "file"   : path,
    "task"   : "Regression",
    "feed"   : ["temperature_c","humidity_pct","num_occupants","building_area_m2","hour_of_day","num_appliances","insulation_grade"],
    "predict": ["energy_kwh"],
})

# ============================================================
# FINAL SUMMARY
# ============================================================
title("📋 ALL DATASETS SUMMARY")

print(f"\n  {'#':<3} {'File':<35} {'Task':<30} {'Rows':<7} {'Cols'}")
print(f"  {'─'*3} {'─'*35} {'─'*30} {'─'*7} {'─'*5}")

dfs = [
    house_df, student_df, iris_df, emp_df, heart_df,
    weather_df, wine_df, car_df, loan_df, energy_df
]
for i, (info_d, df) in enumerate(zip(datasets_info, dfs), 1):
    fname = os.path.basename(info_d["file"])
    print(f"  {i:<3} {fname:<35} {info_d['task']:<30} {df.shape[0]:<7} {df.shape[1]}")

# ============================================================
# READY-TO-USE AllML CODE SNIPPETS
# ============================================================
title("🚀 READY-TO-USE AllML SNIPPETS")

snippets = [
    ("House Prices (Regression)", "house_prices.csv",
     '["area_sqft","bedrooms","bathrooms","year_built","garage_cars"]',
     '["price_usd"]',
     "ml.pred(2000, 3, 2, 2005, 2)"),

    ("Student Performance (Binary)", "student_performance.csv",
     '["study_hours_per_day","attendance_pct","previous_score","sleep_hours","assignments_done"]',
     '["passed"]',
     "ml.pred(6.5, 85.0, 72.0, 7.5, 8)"),

    ("Flower Species (Multiclass)", "flower_species.csv",
     '["sepal_length_cm","sepal_width_cm","petal_length_cm","petal_width_cm"]',
     '["species"]',
     "ml.pred(5.1, 3.5, 1.4, 0.2)"),

    ("Weather Forecast (Multi-Output)", "weather_forecast.csv",
     '["month","day_of_year","humidity_pct","pressure_hpa","wind_speed_kmh","cloud_cover_pct","altitude_m","latitude"]',
     '["temperature_c","humidity_out_pct","rainfall_mm"]',
     "ml.pred(7, 200, 65.0, 1013.0, 15.0, 40.0, 100.0, 28.0)"),

    ("Heart Disease (Binary)", "heart_disease.csv",
     '["age","sex","cholesterol","blood_pressure","max_heart_rate","fasting_blood_sugar","chest_pain_type","smoking"]',
     '["heart_disease"]',
     "ml.pred(55, 1, 230, 140, 150, 0, 2, 1)"),

    ("Loan Approval (Binary)", "loan_approval.csv",
     '["annual_income","loan_amount","credit_score","loan_term_years","employment_years","existing_debt_pct","num_defaults"]',
     '["approved"]',
     "ml.pred(75000, 200000, 720, 15, 5, 20.0, 0)"),

    ("Car Specs (Multi-Output)", "car_specs.csv",
     '["engine_cc","horsepower","torque_nm","weight_kg","cylinders","car_age_years","has_turbo","is_electric","brand_tier"]',
     '["price_usd","fuel_economy_mpg"]',
     "ml.pred(2000, 180, 320, 1500, 4, 3, 1, 0, 2)"),

    ("Wine Quality (Multiclass)", "wine_quality.csv",
     '["fixed_acidity","volatile_acidity","citric_acid","chlorides","alcohol_pct","sulphates","pH"]',
     '["quality"]',
     "ml.pred(7.4, 0.7, 0.0, 0.076, 9.4, 0.56, 3.51)"),

    ("Employee Salary (Regression)", "employee_salary.csv",
     '["age","years_experience","education_level","performance_score","certifications","is_manager"]',
     '["annual_salary"]',
     "ml.pred(35, 10, 3, 4, 2, 0)"),

    ("Energy Consumption (Regression)", "energy_consumption.csv",
     '["temperature_c","humidity_pct","num_occupants","building_area_m2","hour_of_day","num_appliances","insulation_grade"]',
     '["energy_kwh"]',
     "ml.pred(30.0, 70.0, 5, 200, 14, 10, 3)"),
]

for name, csv_file, feed, pred, pred_call in snippets:
    print(f"""
{C.Y}{C.B}# ── {name} ──{C.E}
{C.C}from allml import AllML
ml = AllML("{OUT}/{csv_file}")
ml.col_to_feed({feed})
ml.col_to_pred({pred})
ml.method("random_forest_{'regressor' if 'Regression' in name else 'classifier'}")
ml.split(70, 20, 10)
ml.train(n_estimators=100, epochs=200)
ml.show()
ml.show_graph(["feature_importance", "model_comparison"])
ml.save("models/{csv_file.replace('.csv','')}")
result = {pred_call}
print(f"Prediction: {{result}}"){C.E}""")

print(f"\n{C.G}{C.B}✅ All {len(datasets_info)} datasets generated in '{OUT}/' folder!{C.E}\n")
