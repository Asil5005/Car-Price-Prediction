import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

st.title("Car Linear Regression Visualisation")

# Load dataset
df = pd.read_csv('CarPricePrediction_clean.csv')

# Convert non-numeric values to numeric (extracting numbers)
df['mileage(km/ltr/kg)'] = df['mileage(km/ltr/kg)'].astype(str).str.extract(r'(\d+\.\d+|\d+)').astype(float)
df['engine'] = df['engine'].astype(str).str.extract(r'(\d+)').astype(float)
df['max_power'] = df['max_power'].astype(str).str.extract(r'(\d+\.\d+|\d+)').astype(float)

# Drop any remaining NaN values
df.dropna(inplace=True)

# Apply log transformation to selling_price
#df['selling_price'] = np.log1p(df['selling_price'])

# Create new features
df['car_age'] = 2025 - df['year']
df['km_per_year'] = df['km_driven'] / df['car_age']
df.drop(columns=['year'], inplace=True)  # Drop year after creating age

# Encode categorical features
df = pd.get_dummies(df, columns=['fuel', 'transmission', 'seller_type', 'owner'], drop_first=True)

# Let the user select features and target
all_columns = df.columns.tolist()
default_features = ['km_driven', 'mileage(km/ltr/kg)', 'engine', 'max_power', 'seats', 'car_age', 'km_per_year'] + \
                   [col for col in df.columns if col.startswith(('fuel_', 'transmission_', 'seller_type_', 'owner_'))]
features = st.multiselect("Select Features", options=all_columns, default=default_features)
target_options = ['year', 'selling_price', 'km_driven', 'mileage(km/ltr/kg)', 'engine', 'seats']
target = st.selectbox("Select Target Variable", options=target_options, index=all_columns.index('selling_price'))

# Ensure selected features are valid
if features and target and target not in features:
    X = df[features]
    y = df[target]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply feature scaling (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    predictions = model.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    st.subheader("Model Evaluation")
    st.write(f"Mean Squared Error: **{mse:,.2f}**")
    st.write(f"R2 Score: **{r2:.4f}**")

    # Plot Actual vs Predicted Prices
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=0.7)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

else:
    st.warning("Please select valid features and a target variable. Ensure the target is not among the features.")
