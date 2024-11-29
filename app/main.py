import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.routers import project_router, trainings_router, snapshot_router, bundle_router
from app.models import database  # This imports the models which in turn import db
from app.utils.database import db  # Ensure db is imported correctly

app = FastAPI(title="SSD-CNN Inference API", version="1.00", description="API for managing SSD-CNN model inference")

# Allow all origins
origins = [
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(project_router.router, prefix="/project", tags=["Project management"])
app.include_router(bundle_router.router, prefix="/bundle", tags=["Image bundle management"])
app.include_router(trainings_router.router, prefix="/train", tags=["Model training management"])
app.include_router(snapshot_router.router, prefix="/snapshot", tags=["Model versioning management"])

# Publish Image folders to the API
load_dotenv()
base_dir = os.getenv('BASE_DIR')
images_directory = os.path.join(base_dir, 'static')

app.mount("/static", StaticFiles(directory=images_directory), name="Image files")

@app.on_event("startup")
def startup():
    if db.is_closed():
        db.connect()  # Ensures that db connection is open
    database.create_tables()  # Call a function that manages table creation
    db.close()
