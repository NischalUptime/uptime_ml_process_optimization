import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import time
from datetime import datetime
from typing import List, Dict

class RealTimeDataSimulator:
    def __init__(self, db_config: Dict[str, str]):
        """Initialize database connection and configuration."""
        self.conn = psycopg2.connect(
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        )
        self.cursor = self.conn.cursor()
        
    def create_table(self, columns: List[str]):
        """Create table if not exists with timestamp and data columns."""
        # Quote column names to preserve case
        columns_def = ", ".join([f'"{col}" DOUBLE PRECISION' for col in columns])
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS process_data (
            "timestamp" TIMESTAMP PRIMARY KEY,
            {columns_def}
        );
        """
        self.cursor.execute(create_table_query)
        self.conn.commit()
        
    def bulk_insert(self, df: pd.DataFrame):
        """Insert multiple rows at once."""
        # Prepare data for insertion
        columns = ['"timestamp"'] + [f'"{col}"' for col in df.columns]
        values = [tuple([idx] + row.tolist()) for idx, row in df.iterrows()]
        
        # Insert data
        insert_query = f"""
        INSERT INTO process_data ({', '.join(columns)})
        VALUES %s
        ON CONFLICT ("timestamp") DO NOTHING;
        """
        execute_values(self.cursor, insert_query, values)
        self.conn.commit()
        
    def insert_single_row(self, timestamp: datetime, row_data: Dict[str, float]):
        """Insert a single row of data."""
        columns = ['"timestamp"'] + [f'"{col}"' for col in row_data.keys()]
        values = [datetime.now()] + list(row_data.values())
        
        placeholders = ', '.join(['%s'] * len(columns))
        insert_query = f"""
        INSERT INTO process_data ({', '.join(columns)})
        VALUES ({placeholders})
        ON CONFLICT ("timestamp") DO NOTHING;
        """
        
        self.cursor.execute(insert_query, values)
        self.conn.commit()
        
    def close(self):
        """Close database connection."""
        self.cursor.close()
        self.conn.close()

def main():
    # Hardcoded configuration
    csv_path = "./sample_data_500rows.csv"  # Adjust path as needed

    # Database configuration
    db_config = {
        'dbname': 'process_db',
        'user': 'postgres',
        'password': 'postgres',
        'host': 'localhost',
        'port': '5432'
    }
    
    # Read CSV file
    print(f"Reading data from {csv_path}")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.sort_index()  # Ensure data is sorted by timestamp
    
    simulator = RealTimeDataSimulator(db_config)
    
    try:
        # Create table with appropriate columns
        print("Creating table if not exists...")
        simulator.create_table(df.columns.tolist())
        
        # Calculate split point (60% of data)
        # split_idx = int(len(df))
        bulk_data = df
        streaming_data = df
        
        # Bulk insert first 60%
        print(f"Bulk inserting {len(bulk_data)} records...")
        simulator.bulk_insert(bulk_data)
        print("Bulk insert completed")
        
        # Stream remaining data one row per minute
        print(f"Starting to stream {len(streaming_data)} records...")
        for timestamp, row in streaming_data.iterrows():
            row_dict = row.to_dict()
            print(f"Inserting data for timestamp: {timestamp}")
            simulator.insert_single_row(timestamp, row_dict)
            time.sleep(60)  # Wait for 1 minute
            
    except KeyboardInterrupt:
        print("\nStopping data simulation...")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        simulator.close()
        print("Database connection closed")

if __name__ == "__main__":
    main()
