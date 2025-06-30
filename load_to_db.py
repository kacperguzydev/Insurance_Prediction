import os
import pandas as pd
import sqlite3
from config import CLEANED_DATA_PATH

DB_PATH = os.path.join("database", "insurance.db")

def load_cleaned_data():
    # Load the cleaned dataset with updated string labels
    print("Loading cleaned data...")
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from cleaned data.")
    return df

def save_to_database(df):
    # Create database directory if it doesn't exist
    if not os.path.exists("database"):
        os.makedirs("database")

    print(f"Connecting to SQLite database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)

    # Save dataframe to the database, replacing existing cleaned_data table
    df.to_sql("cleaned_data", conn, if_exists="replace", index=False)
    print("Cleaned data saved to database table 'cleaned_data'.")

    conn.close()
    print("Database connection closed.")

if __name__ == "__main__":
    df_cleaned = load_cleaned_data()
    save_to_database(df_cleaned)
