# Mini-Project-4: Tourism Experience Analytics:  Classification, Prediction, and Recommendation System 

## Overview
This project focuses on analyzing tourism transaction data to predict user preferences and enhance travel experiences through Machine Learning and Recommendation Systems. The project consists of three major tasks: predicting attraction ratings using regression, predicting users' visit modes using classification, and recommending personalized tourist attractions based on user preferences.

The dataset contains user demographics, attraction information, visit details, and user ratings. After performing data preprocessing and feature engineering, the processed data is stored in a PostgreSQL database. An interactive Streamlit web application is developed to provide rating prediction, visit mode prediction, and personalized attraction recommendations through an easy-to-use interface.

## Technologies Used
- **Python** (Pandas, NumPy): For data preprocessing, feature engineering, and analysis.
- **scikit-learn**: For preprocessing, machine learning, and model evaluation.
- **XGBoost**: For building the regression and classification models.
- **Surprise Library**: For developing the Collaborative Filtering recommendation system.
- **PostgreSQL**: For storing processed tourism data.
- **psycopg2**: For PostgreSQL database connection and SQL execution.
- **SQLAlchemy**: For seamless integration between PostgreSQL and Pandas.
- **Joblib**: For saving and loading trained machine learning models.
- **Streamlit**: For building an interactive web application.
- **Matplotlib & Seaborn**: For Exploratory Data Analysis (EDA) and data visualization.

## Steps Involved

### 1. Data Preprocessing
- Loaded multiple tourism datasets (Excel files) using Pandas.
- Merged the datasets based on their respective keys and common attributes to create a unified tourism dataset.
- Converted the merged dataset into a single CSV file for further processing and analysis.
- Removed duplicate records and handled missing values.
- Selected relevant user, attraction, and transaction attributes.
- Standardized categorical values to ensure consistency across the dataset.
- Prepared clean datasets for regression, classification, and recommendation tasks.

### 2. Feature Engineering
- Created meaningful features from user behavior and tourism transactions.
- Generated additional attributes to improve the performance of machine learning models.
- Prepared separate feature sets suitable for regression and classification models.

### 3. Exploratory Data Analysis (EDA)
- Analyzed user distribution across continents, countries, and regions.
- Explored attraction popularity and user rating patterns.
- Investigated relationships between user demographics and visit modes.
- Analyzed rating distributions across different attraction types and regions.
- 
### 4. PostgreSQL Table Creation & Data Insertion
- Created PostgreSQL tables for storing processed datasets.
- Defined appropriate data types for numerical and categorical attributes.
- Inserted the cleaned datasets into PostgreSQL using SQLAlchemy.
- Retrieved data directly from PostgreSQL within the Streamlit application.

### 5. Regression Model - Attraction Rating Prediction
- Developed a regression model to predict the rating a user is likely to give to a tourist attraction.
- Trained and evaluated multiple regression algorithms to identify the best-performing model.
- Used the trained model to estimate user satisfaction based on tourism and attraction-related information.

### 6. Classification Model - Visit Mode Prediction
- Developed a multi-class classification model to predict a user's visit mode.
- Applied appropriate preprocessing and encoding techniques before model training.
- Evaluated the model using standard classification performance metrics.
- Saved the trained model and preprocessing objects for deployment.

### 7. Recommendation System
- Developed personalized attraction recommendations using:
  - Collaborative Filtering
  - Content-Based Filtering
- Collaborative Filtering recommends attractions based on similar users' ratings and preferences.
- Content-Based Filtering recommends attractions based on similarities in attraction characteristics such as attraction type, location, and amenities.

### 8. Streamlit Dashboard Development
- Built an interactive web application using Streamlit.
- Retrieved processed data directly from PostgreSQL.
- Allowed users to provide their travel preferences through an intuitive interface.
- Predicted attraction ratings using the regression model.
- Predicted visit modes using the classification model.
- Generated personalized attraction recommendations using collaborative and content-based recommendation techniques.
- Presented prediction results and recommendations through a clean and user-friendly interface.
