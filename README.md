# SSD-CNN FastAPI Application

## Project Overview
This project is a FastAPI application designed to manage SSD (Single Shot MultiBox Detector) CNN (Convolutional Neural Network) models for image processing and object detection. It includes functionality for managing projects, image bundles, annotations, and trained model snapshots.

## System Requirements
- Python 3.8+
- MySQL Server
- Libraries: FastAPI, Peewee, PyMySQL, Uvicorn

## Project Structure
    SSD-CNN/
    │
    ├── app/                           # Main application folder for FastAPI
    │   │
    │   ├── models/
    │   │   ├── database.py            # Database orm model definition
    │   │   │
    │   │   ├── dicts/                 # Dictionairies for passing data between Router and Service Classes
    │   │   ├── response/              # Response model definition for API
    │   │   ├── request/               # Request model definition (json format) for API requests
    │   │
    │   ├── routers/                   # Fast API Router structure
    │   │
    │   ├── services/                  # Fast API Logic
    │   │
    │   ├── utils/                     # All helper functions for services
    │   │
    │   ├── enums/                     # Status Enum definition
    │   │
    │   ├── dependencies.py            # Database session management
    │   ├── main.py                    # FastAPI application entry point
    │
    │
    ├── static/                        # Staticly generated files for the application
    │
    ├── test/                         # Test results and collections
    │
    ├── requirements.txt               # Project python dependencies
    ├── .env                           # Environment variables file
    └── README.md                      # This file

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://gitlab.rz.htw-berlin.de/s0577395/ssd-cnn.git
   cd SSD-CNN
   
2. **Create virtual python environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   
3. **Install Dependencies**
    ```bash
   pip install -r requirements.txt

4. **Set Up the Environment Variables**
   - Rename the .env.example file to .env.
   - Update the .env file with your MySQL database credentials and other configurations.

5. **Run the Application**
    ```bash
    uvicorn app.main:app --reload
   
   # or for production and published on port 600
   uvicorn app.main:app --host 0.0.0.0 --port 8888


## Usage
Access the FastAPI Swagger UI to interact with the API at http://localhost:8000/docs.
Use endpoints under /project to manage projects and /bundle for running inference tasks.

## License
This project is licensed under the MIT License - see the [LICENSE](https://gitlab.rz.htw-berlin.de/s0577395/ssd-cnn/-/blob/main/LICENSE) file for details.