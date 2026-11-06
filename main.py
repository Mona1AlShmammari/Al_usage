# main.py - Streamlit app for the AI Usage and Productivity project
# Run with: streamlit run main.py

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# 1. Page settings and loading the saved files
# ---------------------------------------------------------------
st.set_page_config(page_title="Productivity Predictor", layout="wide")

MODELS_DIR = Path(__file__).parent / "Models"


@st.cache_resource
def load_files():
    info = joblib.load(MODELS_DIR / "feature_info.joblib")      # columns and input options
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")          # scaler fitted in Notebook 02
    models = {name: joblib.load(MODELS_DIR / file)              # the four trained models
              for name, file in info["file_names"].items()}
    results = pd.read_csv(MODELS_DIR / "model_comparison.csv", index_col=0)
    return info, scaler, models, results


info, scaler, models, results = load_files()

# ---------------------------------------------------------------
# 2. Title and model selection
# ---------------------------------------------------------------
st.title("AI Usage and Productivity Prediction")
st.write("Enter the details of a user, choose a model, and predict the Productivity Score.")

model_names = list(models.keys())
default_index = model_names.index(info["best_model"])

model_name = st.sidebar.selectbox("Choose a model", model_names, index=default_index)
st.sidebar.write("Best model by cross-validation R²: " + info["best_model"])

# ---------------------------------------------------------------
# 3. Input fields for a new user
# ---------------------------------------------------------------
st.subheader("User details")
col_left, col_right = st.columns(2)

user_input = {}

# text columns become dropdown lists
for i, (column, options) in enumerate(info["text_options"].items()):
    with (col_left if i % 2 == 0 else col_right):
        user_input[column] = st.selectbox(column.replace("_", " "), options)

# numeric columns become number boxes (limited to the range seen in the data)
for i, (column, (low, high, median)) in enumerate(info["num_ranges"].items()):
    with (col_left if i % 2 == 0 else col_right):
        user_input[column] = st.number_input(column.replace("_", " "),
                                             min_value=low, max_value=high, value=median)

# ---------------------------------------------------------------
# 4. Prepare the input exactly like the training data
# ---------------------------------------------------------------
def prepare(user_input):
    row = pd.DataFrame([user_input])
    row = pd.get_dummies(row)                                  # text -> 0/1 columns
    row = row.reindex(columns=info["columns"], fill_value=0)   # same columns and order as training
    row = row.astype(float)
    row[info["num_cols"]] = scaler.transform(row[info["num_cols"]])  # same scaling as training
    return row


def predict(model, row):
    value = model.predict(row)[0]
    return min(max(value, 1.0), 10.0)                          # the score is between 1 and 10

# ---------------------------------------------------------------
# 5. Prediction and comparison of all models
# ---------------------------------------------------------------
if st.button("Predict"):
    new_row = prepare(user_input)

    st.subheader("Prediction")
    st.success(model_name + " predicts a Productivity Score of "
               + str(round(predict(models[model_name], new_row), 2)))

    st.subheader("Comparison of all models")
    columns = st.columns(len(models))
    all_predictions = {}
    for column, (name, model) in zip(columns, models.items()):
        all_predictions[name] = predict(model, new_row)
        column.metric(name, round(all_predictions[name], 2))

    st.bar_chart(pd.Series(all_predictions))

st.subheader("Model performance on the test set")
st.dataframe(results)