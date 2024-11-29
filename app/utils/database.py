from playhouse.pool import PooledMySQLDatabase
import os
import time
from dotenv import load_dotenv

load_dotenv()

def get_database():
    """
    Create a new PooledMySQLDatabase instance with retry logic.
    """
    max_retries = 5
    delay = 5  # seconds
    attempt = 0

    while attempt < max_retries:
        try:
            pooled_db = PooledMySQLDatabase(
                os.getenv('DB_NAME'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD'),
                host=os.getenv('DB_HOST'),
                port=int(os.getenv('DB_PORT')),
                charset='utf8mb4',
                max_connections=16,
                stale_timeout=600,
            )
            # Try connecting to the database
            pooled_db.connect()
            print("Database connection successful.")
            return pooled_db
        except Exception as e:
            attempt += 1
            print(f"Database connection failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries exceeded. Exiting.")
                raise

db = get_database()