# Al_usage
.....
AI Usage and Impact on Students & Professionals

A regression project that explores how students and professionals use AI
tools and predicts a user’s Productivity Score based on their AI
usage and related personal, educational, and work/study information.

Project Overview

The project follows a complete machine-learning workflow:

 1. Explore the dataset and inspect data quality.
 2. Clean and preprocess the data.
 3. Train and compare multiple regression models.
 4. Use cross-validation and hyperparameter tuning.
 5. Evaluate the models on a held-out test set.
 6. Save the trained models and preprocessing objects.
 7. Provide an interactive Streamlit application for new predictions.

Target variable: Productivity_Score

Dataset

 ● Dataset name: AI Usage and Impact on Students & Professionals
 ● Source: Kaggle – AI Usage and Impact on Students and
Professionals

 ● Original dataset size: 300 rows and 19 columns
 ● Rows used for modeling: 274 rows after removing duplicate user
IDs during cleaning and excluding rows with a missing target value.

 ● Target: Productivity_Score

The dataset includes information such as age, gender, country, user
type, education level, profession, AI tool, AI purpose, AI usage hours,
monthly AI cost, accuracy rating, satisfaction score, time saved, and
work/study hours.

The User_ID column is an identifier and is not used as a predictive
feature. Tasks_Performed and Profession are excluded from the
modeling features in the current modeling notebook.

Repository Structure

The repository is organized into the following folders and files (keep
the names consistent with the files committed to GitHub):

Al_usage_main/

├── Data/
│   └── Cleaned
│   ├── Original/
│   │   ├──AI_Usage_and_Impact_on_Students_and_Professionals.csv.csv

├── Notebooks/

│   ├── 01_EDA_and_Preprocessing.ipynb

│   └── 02_Modeling_and_Preprocessing.ipynb

├── Models/
│   ├── linear_regression.joblib
│   ├── ridge.joblib
│   ├── random_forest.joblib
│   ├── gradient_boosting.joblib
│   ├── scaler.joblib
│   ├── feature_info.joblib
│   └── model_comparison.csv

├── Images/
│   ├── model_comparison.png  
│   ├── actual_vs_predicted.png
│   └── feature_importance.png

├── main.py
├── requirements.txt
├── .gitignore
└── README.md

Models/ contains the four fitted regression models, the fitted scaler,
the feature information needed by the app, and the model comparison
results. Images/ contains plots generated during model evaluation.

Data Exploration and Preprocessing

The EDA notebook examines the dataset shape, column types, missing
values, duplicate records and user IDs, and the distributions of
important variables.

The preprocessing work includes:

 ● Converting fields containing numeric text (for example, age values
with yrs, AI usage hours, and monthly AI cost values containing
$) into numeric values.

 ● Standardizing inconsistent category labels, including gender,
country, education level, and AI tool names.

 ● Filling missing numeric values with the median.
 ● Filling missing categorical values with Unknown.
 ● Removing duplicate records based on User_ID, keeping the first
occurrence.

 ● Saving the cleaned dataset for the modeling stage.

For modeling, rows with a missing Productivity_Score are excluded. The
identifier User_ID is removed, and Tasks_Performed and Profession
are also excluded from the predictor set.

Modeling Approach

The modeling notebook uses an 80/20 train-test split with
random_state=42. This produces 219 training rows and 55 test rows from
the 274 rows available for modeling.

Categorical variables are one-hot encoded. Eight numeric features are
standardized using StandardScaler. The scaler is fitted on the
training data only and then applied to the test data to avoid data
leakage.

The following regression models are trained and compared:

 ● Linear Regression
 ● Ridge Regression
 ● Random Forest Regressor
 ● Gradient Boosting Regressor

Cross-validation and Hyperparameter Tuning

Five-fold cross-validation is performed using shuffled KFold with
random_state=42.
GridSearchCV is used to tune Ridge, Random Forest, and Gradient
Boosting. The selected Gradient Boosting settings are:

 ● learning_rate = 0.03
 ● max_depth = 2
 ● n_estimators = 200

Gradient Boosting achieved the highest tuned cross-validation R² score
(0.667) and is selected as the preferred model based on
cross-validation performance.

Model Evaluation

The following table reports the held-out test-set results from the
modeling notebook.

Model                   MAE    RMSE       R²



Linear Regression     0.999   1.247    0.568
Ridge                 1.021   1.284    0.542
Random Forest         0.968   1.250    0.566
Gradient Boosting     0.987   1.248    0.567
Baseline (mean)       1.616   1.897   -0.000

Interpretation: Linear Regression has the highest test R² (0.568),
while Random Forest has the lowest test MAE (0.968). The test results
for Linear Regression, Random Forest, and Gradient Boosting are very
close. Gradient Boosting is selected as the preferred model because it
achieved the highest tuned cross-validation R² (0.667), not because it
had the highest test-set R².

 ● MAE measures the average absolute prediction error.
 ● RMSE measures prediction error while penalizing larger errors
more strongly.

 ● R² measures the proportion of target variation explained by the
model.

Streamlit Application

The Streamlit app is available in main.py.

The app allows a user to:

 1. Enter user details through dropdown lists and numeric input fields.
 2. Select one of the four trained regression models.
 3. Generate a predicted Productivity_Score.
 4. View predictions from all four models side by side.
 5. View the test-set model performance table.

The app loads the saved models, scaler, feature information, and
comparison table from the Models/ folder.

Run the App

From the repository root, run:

streamlit run main.py

Installation

Requirements

 ● Python
 ● The packages and pinned versions listed in requirements.txt

Install Dependencies

pip install -r requirements.txt

Run the Notebooks

Install the requirements, then open Jupyter Notebook:

jupyter notebook

Run the notebooks in order:

 1. 01_EDA_and_Preprocessing.ipynb
 2. 02_Modeling_and_Preprocessing.ipynb

The notebooks use relative file paths. Keep the dataset and output
folders in the locations expected by the notebook code, or update those
paths if the repository layout is different.

Limitations

 ● The dataset contains 300 original records, and only 274 records are
used for modeling after the cleaning and target-value filtering
steps.

 ● The test set contains 55 rows, so the differences between the
leading models are small and should be interpreted cautiously.

 ● The model estimates a productivity score from patterns in the
available dataset; it does not establish that AI usage causes higher
or lower productivity.

 ● Predictions are limited by the features, data quality, and value
ranges represented in the dataset.

 ● The app is a demonstration of regression predictions and is not
intended to make high-stakes decisions.

Technologies

 ● Python
 ● Pandas and NumPy
 ● Matplotlib and Seaborn
 ● Scikit-learn
 ● Joblib
 ● Streamlit
 ● Jupyter Notebook

