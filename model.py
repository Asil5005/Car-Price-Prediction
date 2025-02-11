import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('CarPricePrediction_clean.csv')

# Convert non-numeric data to numbers using raw strings (r'...')
df['mileage(km/ltr/kg)'] = df['mileage(km/ltr/kg)'].astype(str).str.extract(r'(\d+\.\d+|\d+)').astype(float)
df['engine'] = df['engine'].astype(str).str.extract(r'(\d+)').astype(float)
df['max_power'] = df['max_power'].astype(str).str.extract(r'(\d+\.\d+|\d+)').astype(float)

# Handle missing values
df.dropna(inplace=True)

# Apply log transformation to selling_price
df['selling_price'] = np.log1p(df['selling_price'])

# Create new features
df['car_age'] = 2025 - df['year']
df['km_per_year'] = df['km_driven'] / df['car_age']
df.drop(columns=['year'], inplace=True)  # Drop year after creating age

# Encode categorical features
df = pd.get_dummies(df, columns=['fuel', 'transmission', 'seller_type', 'owner'], drop_first=True)

# Define features and target variable
features = ['km_driven', 'mileage(km/ltr/kg)', 'engine', 'max_power', 'seats', 'car_age', 'km_per_year'] + \
           [col for col in df.columns if col.startswith(('fuel_', 'transmission_', 'seller_type_', 'owner_'))]
target = 'selling_price'

X = df[features]
y = df[target]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')
