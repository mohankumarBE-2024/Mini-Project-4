import streamlit as st
import pandas as pd
import joblib
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
svd = joblib.load('svd_model.pkl')
encoder = joblib.load("content_encoder.pkl")
similarity_matrix = joblib.load("similarity_matrix.pkl")
attraction_index = joblib.load("attraction_index.pkl")
content_df = joblib.load("content_df.pkl")

clean_mode_df = pd.read_sql(
    "SELECT * FROM visit_mode_prediction",
    engine
)

merge_df = pd.read_sql(
    "SELECT * FROM tourism_data",
    engine
)

attraction_info = merge_df[['attraction', 'attractionid', 'attractiontype', 'attractioncityname', 'attractioncountry']]
attraction_info = attraction_info.drop_duplicates()
attraction_info = attraction_info.dropna()

st.title("🌍 Tourism Experience Analytics")
st.markdown("---")

page = st.sidebar.radio(
    "📌 Select Module",
    [
        "📊 Exploratory Data Analysis",
        "🤖 Visit Mode Classification",
        "⭐ Content-Based Filtering",
        "👥 Collaborative Filtering"
    ]
)

if page == "📊 Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis")

    option = st.selectbox(
        "Select Visualization",
        [
            "User Distribution",
            "Attraction-Type Popularity",
            "Visit Mode Analysis",
            "Ratings Distribution",
            "Correlation Analysis"
        ]
    )

    if option == "User Distribution":
        

        # Count unique users in each continent
        continent_users = (
            merge_df.groupby("usercontinent")["userid"]
            .nunique()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(8,5))

        sns.barplot(
            x=continent_users.index,
            y=continent_users.values,
            hue=continent_users.index,
            palette="viridis",
            legend=True
        )

        ax.set_title("User Distribution Across Continents", fontsize=14)
        ax.set_xlabel("Continent")
        ax.set_ylabel("Number of Users")
        plt.xticks(rotation=20)

        for i, v in enumerate(continent_users.values):
            plt.text(i, v, str(v), ha='center', va='bottom')

        st.pyplot(fig)


    elif option == "Attraction-Type Popularity":

        rating_count = (merge_df.groupby("attractiontype")["rating"].count().sort_values(ascending=False))

        fig, ax = plt.subplots(figsize=(8,5))

        sns.barplot(
            x=rating_count.index,
            y=rating_count.values,
            hue=rating_count.index,
            palette="magma",
            legend=False
        )

        ax.set_title("Popularity of Attraction Types", fontsize=14)
        ax.set_xlabel("Attraction Type")
        ax.set_ylabel("Number of Ratings")

        plt.xticks(rotation=60, ha="right")

        for i, v in enumerate(rating_count.values):
            plt.text(i, v, str(v), ha="center", va="bottom", fontsize=9)

        st.pyplot(fig)

    elif option == "Visit Mode Analysis":
        top_countries = merge_df["usercountry"].value_counts().head(10).index

        country_df = merge_df[
            merge_df["usercountry"].isin(top_countries)
        ]

        country_visit = pd.crosstab(
            country_df["usercountry"],
            country_df["visitmode_mode"]
        )

        fig, ax = plt.subplots(figsize=(14,6))

        country_visit.plot(
            kind="bar",
            stacked=True,
            colormap="Paired",
            ax=ax
        )

        ax.set_title("Visit Mode Distribution Across Top 10 User Countries")
        ax.set_xlabel("User Country")
        ax.set_ylabel("Number of Visits")
        ax.tick_params(axis='x', rotation=60)

        plt.legend(title="Visit Mode")

        st.pyplot(fig)

    elif option == "Ratings Distribution":
        avg_rating = (
            merge_df.groupby("attraction")["rating"]
                    .mean()
                    .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(12,6))

        sns.barplot(
            x=avg_rating.index,
            y=avg_rating.values,
            hue=avg_rating.index,
            palette="viridis",
            legend=False
        )

        ax.set_title("Average Rating by Attraction Type")
        ax.set_xlabel("Attraction Type")
        ax.set_ylabel("Average Rating")
        plt.xticks(rotation=60, ha="right")

        st.pyplot(fig)


    elif option == "Correlation Analysis":

        continent_visit = pd.crosstab(
            merge_df["usercontinent"],
            merge_df["visitmode_mode"]
        )

        fig, ax = plt.subplots(figsize=(8,5))

        sns.heatmap(
            continent_visit,
            annot=True,
            fmt="d",
            cmap="YlGnBu"
        )

        ax.set_title("Heatmap of Visit Mode Across User Continents")
        ax.set_xlabel("Visit Mode")
        ax.set_ylabel("User Continent")

        st.pyplot(fig)


elif page == "🤖 Visit Mode Classification":

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

elif page == "👥 Collaborative Filtering":

    def collaborative_filtering(user_id, top_n):
        user_visited = merge_df[merge_df["userid"] == user_id]["attractionid"].unique()

        all_attractions = merge_df["attractionid"].unique()

        unseen = np.setdiff1d(
            all_attractions,
            user_visited
        )

        predictions = []

        for attraction in unseen:

            est = svd.predict(
                uid=user_id,
                iid=attraction
            ).est

            predictions.append((attraction, est))

        predictions = sorted(
            predictions,
            key=lambda x: x[1],
            reverse=True
        )

        recommendation = pd.DataFrame(
            predictions[:top_n],
            columns=["attractionid", "predictedrating"]
        )

        recommendation = recommendation.merge(
            attraction_info,
            on="attractionid"
        )

        recommendation = recommendation[
        [
            'attraction',
            'attractiontype',
            'attractioncityname', 
            'attractioncountry',
            'predictedrating'
        ]]

        recommendation["predictedrating"] = (
            recommendation["predictedrating"]
            .round(1)
        )

        return recommendation
    
    user_id = st.number_input(
    "Enter User ID",
    min_value=1,
    step=1)

    top_n = st.slider(
        "Number of Recommendations",
        1,
        10,
        5
    )

    if st.button("Recommend"):

        result = collaborative_filtering(
            user_id,
            top_n
        )

        st.dataframe(result)

elif page == "⭐ Content-Based Filtering":
    st.header("⭐ Content-Based Recommendation")

    selected_attraction = st.selectbox(
        "Select an Attraction",
        sorted(content_df["Attraction"].unique())
    )

    top_n = st.slider(
        "Number of Recommendations",
        1,
        10,
        5
    )

    attraction_id = content_df.loc[
    content_df["Attraction"] == selected_attraction,
    "AttractionId"].iloc[0]

    def recommend_similar_attractions(attraction_id, top_n=5):
        idx = attraction_index[attraction_id]

        similarity_scores = list(enumerate(similarity_matrix[idx]))

        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        similarity_scores = similarity_scores[1:top_n+1]

        recommendations = []

        for index, score in similarity_scores:

            recommendations.append({

                "Attraction":
                    content_df.iloc[index]["Attraction"],

                "Attraction Type":
                    content_df.iloc[index]["AttractionType"],

                "City":
                    content_df.iloc[index]["AttractionCityName"],

                "Average Rating":
                    round(content_df.iloc[index]["AvgRating"],2),

                "Similarity Score":
                    round(score,2)
            })

        return pd.DataFrame(recommendations)

    
    if st.button("Recommend Similar Attractions"):

        recommendation = recommend_similar_attractions(
            attraction_id,
            top_n
        )

        st.dataframe(recommendation)

    
