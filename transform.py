import os
import pandas as pd
import numpy as np
from config import DATA_PATH, CLEANED_DATA_PATH

def load_data():
    # Load the raw insurance data from CSV.
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def inspect_data(df):
    # Display basic information about the dataset.
    print("\nColumn names and data types:")
    print(df.dtypes)

    print("\nMissing values by column:")
    print(df.isnull().sum())

    print("\nFirst 5 rows of the dataset:")
    print(df.head())

def clean_data(df):
    # Clean the dataset by handling missing values and encoding categorical variables.
    print("\nStarting data cleaning...")

    # Drop rows where the target variable CLAIM_STATUS is missing
    if 'CLAIM_STATUS' in df.columns:
        initial_rows = df.shape[0]
        df = df.dropna(subset=['CLAIM_STATUS'])
        rows_dropped = initial_rows - df.shape[0]
        print(f"Dropped {rows_dropped} rows with missing CLAIM_STATUS.")

    # Fill missing numeric columns with median values
    numeric_columns = df.select_dtypes(include=np.number).columns
    for col in numeric_columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"Filled {missing_count} missing values in column '{col}' with median.")

    # Encode GENDER column if it exists
    if 'GENDER' in df.columns:
        df['GENDER'] = df['GENDER'].map({'Male': 0, 'Female': 1})
        print("Encoded 'GENDER' column as numeric.")

    print("Data cleaning completed.")
    return df

def update_labels(df):
    # Convert encoded integer columns to human-readable strings.
    print("\nUpdating labels to human-readable strings...")

    if 'sex' in df.columns:
        df['sex'] = df['sex'].map({0: "female", 1: "male"})
        print("Updated 'sex' column to string labels.")

    if 'smoker' in df.columns:
        df['smoker'] = df['smoker'].map({0: "non-smoker", 1: "smoker"})
        print("Updated 'smoker' column to string labels.")

    if 'region' in df.columns:
        df['region'] = df['region'].map({
            0: "northeast", 1: "northwest", 2: "southeast", 3: "southwest"
        })
        print("Updated 'region' column to string labels.")

    if 'insuranceclaim' in df.columns:
        df['insuranceclaim'] = df['insuranceclaim'].map({0: "no", 1: "yes"})
        print("Updated 'insuranceclaim' column to string labels.")

    print("Label updates completed.")
    return df

def save_cleaned_data(df):
    # Save the cleaned dataset to a CSV file.
    print(f"\nSaving cleaned data to {CLEANED_DATA_PATH}...")
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print("Cleaned data saved successfully.")


# Load the raw data
df_raw = load_data()

# Inspect the data for structure, types, and missing values
inspect_data(df_raw)

# Clean the data by handling missing values and encoding categories
df_cleaned = clean_data(df_raw)

# Convert integer codes to human-readable labels
df_cleaned = update_labels(df_cleaned)

# Save the cleaned dataset to a new CSV file
save_cleaned_data(df_cleaned)
