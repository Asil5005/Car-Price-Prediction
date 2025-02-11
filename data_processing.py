import pandas as pd

# Load the dataset (adjust the filename/path if needed)
df = pd.read_csv('CarPricePrediction.csv')

# Inspect the data
print("Head of the data:")
print(df.head())
print("\nData Info:")
print(df.info())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Handle missing values (for example, drop rows with missing values)
df.dropna(inplace=True)

# Save the cleaned data for future use
df.to_csv('CarPricePrediction_clean.csv', index=False)
