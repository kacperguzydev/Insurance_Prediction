import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join("database", "insurance.db")
DATA_DIR = "data"  # directory for all CSV outputs

def run_query(conn, query):
    return pd.read_sql_query(query, conn)

def analyze_kpis(conn):
    print("\nRunning KPI analysis...")

    # Claim distribution: count of yes/no claims
    query_claim_distribution = """
    SELECT 
        insuranceclaim,
        COUNT(*) AS count
    FROM cleaned_data
    GROUP BY insuranceclaim;
    """
    df_claim_dist = run_query(conn, query_claim_distribution)
    print("\nClaim distribution (yes/no):")
    print(df_claim_dist)

    # Claim rate: calculate ratio of 'yes' claims to total
    query_claim_rate = """
    SELECT 
        ROUND(
            100.0 * SUM(CASE WHEN insuranceclaim='yes' THEN 1 ELSE 0 END) / COUNT(*), 
            2
        ) AS claim_rate_percent
    FROM cleaned_data;
    """
    df_claim_rate = run_query(conn, query_claim_rate)
    print("\nOverall claim rate (% of drivers with claims):")
    print(df_claim_rate)

    # Average driver age by claim outcome
    query_age_by_outcome = """
    SELECT 
        insuranceclaim,
        ROUND(AVG(age), 2) AS average_age,
        COUNT(*) AS count
    FROM cleaned_data
    GROUP BY insuranceclaim;
    """
    df_age_by_outcome = run_query(conn, query_age_by_outcome)
    print("\nAverage driver age by claim outcome:")
    print(df_age_by_outcome)

    # Save KPI results to CSV in data directory
    df_claim_dist.to_csv(os.path.join(DATA_DIR, "kpis_claim_distribution.csv"), index=False)
    df_claim_rate.to_csv(os.path.join(DATA_DIR, "kpis_claim_rate.csv"), index=False)
    df_age_by_outcome.to_csv(os.path.join(DATA_DIR, "kpis_age_by_outcome.csv"), index=False)
    print("\nSaved KPI results to CSV in data/ folder.")

def detect_anomalies(conn):
    print("\nRunning anomaly detection...")

    if "charges" in get_columns(conn, "cleaned_data"):
        query_anomalies = """
        SELECT 
            *,
            ROUND(charges, 2) AS charges_rounded
        FROM cleaned_data
        WHERE charges > 50000;
        """
        df_anomalies = run_query(conn, query_anomalies)

        # Replace original 'charges' with rounded version for clarity
        if 'charges_rounded' in df_anomalies.columns:
            df_anomalies.drop(columns=['charges'], inplace=True, errors='ignore')
            df_anomalies.rename(columns={'charges_rounded': 'charges'}, inplace=True)

        print("\nDrivers with charges over $50,000 detected (rounded charges):")
        print(df_anomalies)

        df_anomalies.to_csv(os.path.join(DATA_DIR, "anomalies_high_charges.csv"), index=False)
        print("Saved anomalies to CSV in data/ folder.")
    else:
        print("No 'charges' column found. Skipping anomaly detection.")


def get_columns(conn, table_name):
    cursor = conn.execute(f"PRAGMA table_info({table_name});")
    return [info[1] for info in cursor.fetchall()]

if __name__ == "__main__":
    # Ensure data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print(f"Connecting to database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)

    analyze_kpis(conn)
    detect_anomalies(conn)

    conn.close()
    print("\nSQL analysis completed. Database connection closed.")
