from app.utils.database import db

"""
FastAPI dependency that provides access to the database.

This function can be used with FastAPI's Depends to ensure that a database connection
is available when a route handler is called. It yields the database connection from the
global `db` instance defined in `app.utils.database`.

Yields:
    Database: The database connection object.

Example:
    @app.get("/items/")
    async def read_items(db: Database = Depends(get_database)):
        # Use `db` to interact with the database
        pass
"""


def get_db():
    if db.is_closed():
        db.connect(reuse_if_open=True)
    try:
        yield db
    finally:
        if not db.is_closed():
            db.close()
