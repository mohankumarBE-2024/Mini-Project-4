import streamlit as st
import pandas as pd
import joblib
import pandas as pd
from sqlalchemy import create_engine

# PostgreSQL Credentials
host = "localhost"
port = "5432"
database = "tourism_mode"
username = "postgres"
password = "password"

# Create Engine
engine = create_engine(
    f"postgresql://{username}:{password}@{host}:{port}/{database}"
)


# -----------------------------
# Load Model and Dataset
# -----------------------------
model = joblib.load("visit_mode_model.pkl")
ohe = joblib.load("onehot_encoder.pkl")
le = joblib.load("label_encoder.pkl")
clean_mode_df = pd.read_sql(
    "SELECT * FROM visit_mode_prediction",
    engine
)

# -----------------------------
# Season Function
# -----------------------------
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Tourism Visit Mode Prediction")

visit_year = st.selectbox(
    "Visit Year",
    sorted(clean_mode_df["visityear"].unique())
)

visit_month = st.selectbox(
    "Visit Month",
    sorted(clean_mode_df["visitmonth"].unique())
)

user_continent = st.selectbox(
    "User Continent",
    sorted(clean_mode_df["usercontinent"].unique())
)

user_region = st.selectbox(
    "User Region",
    sorted(clean_mode_df["userregion"].unique())
)

user_country = st.selectbox(
    "User Country",
    sorted(clean_mode_df["usercountry"].unique())
)

user_city = st.selectbox(
    "User City",
    sorted(clean_mode_df["usercityname"].unique())
)

attraction_type = st.selectbox(
    "Preferred Attraction Type",
    sorted(clean_mode_df["attractiontype"].unique())
)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Visit Mode"):

    season = get_season(visit_month)

    # ------------------------------------------
    # Find users with similar profile
    # ------------------------------------------
    similar_users = clean_mode_df[
        (clean_mode_df["usercontinent"] == user_continent) &
        (clean_mode_df["userregion"] == user_region) &
        (clean_mode_df["usercountry"] == user_country) &
        (clean_mode_df["usercityname"] == user_city) &
        (clean_mode_df["attractiontype"] == attraction_type)
    ]

    # ------------------------------------------
    # Estimate Historical Features
    # ------------------------------------------
    if len(similar_users) > 0:

        historical_visit_count = round(
            similar_users["historicalvisitcount"].mean()
        )

        unique_attraction_visited = round(
            similar_users["uniqueAttractionvisited"].mean()
        )

    else:

        historical_visit_count = round(
            clean_mode_df["historicalvisitcount"].mean()
        )

        unique_attraction_visited = round(
            clean_mode_df["uniqueattractionvisited"].mean()
        )

    # ------------------------------------------
    # Create Input DataFrame
    # ------------------------------------------
    input_df = pd.DataFrame({
    "VisitYear": [visit_year],
    "VisitMonth": [visit_month],
    "UserContinent": [user_continent],
    "UserRegion": [user_region],
    "UserCountry": [user_country],
    "UserCityName": [user_city],
    "AttractionType": [attraction_type],
    "Season": [season],
    "HistoricalVisitCount": [historical_visit_count],
    "UniqueAttractionVisited": [unique_attraction_visited]
})

    # Encode the input
    input_encoded = ohe.transform(input_df)

    # Predict
    prediction = model.predict(input_encoded)

    # Convert label back to original class
    prediction = le.inverse_transform(prediction)

    st.success(f"Predicted Visit Mode : {prediction[0]}")