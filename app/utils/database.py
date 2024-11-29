from playhouse.pool import PooledMySQLDatabase
import os

from dotenv import load_dotenv
load_dotenv()


def get_database():
    """
    Create a new PooledMySQLDatabase instance with configuration from environment variables.

    Returns:
        PooledMySQLDatabase: A pooled MySQL database connection.
    """

    pooled_db = PooledMySQLDatabase(
        os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        host=os.getenv('DB_HOST'),
        port= int(os.getenv('DB_PORT')),
        charset='utf8mb4',
        max_connections=16,  # Increase the max connections
        stale_timeout=600, # 10 min
    )

    return pooled_db


db = get_database()
