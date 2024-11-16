from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from datetime import date
import pandas as pd
from io import StringIO
import classifier
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

app = FastAPI()

db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

conn = psycopg2.connect(
    host=db_host,
    database=db_name,
    user=db_user,
    password=db_password
)

class RequestModel(BaseModel):
    marks_out_of: int
    date: date
    institute_id: int
    class_id: int

@app.post("/upload/")
async def upload_data(
    marks_out_of: int = Form(...),
    date: date = Form(...),
    institute_id: int = Form(...),
    class_id: int = Form(...),
    csv: UploadFile = Form(...)
):
    csv_content = await csv.read()
    
    df = pd.read_csv(StringIO(csv_content.decode("utf-8")))

    n_clusters = 5
    df = classifier.classify(df, n_clusters)

    tier_df = classifier.tier_df(df)

    classifier.update_institute_class_students(conn, tier_df)
    classifier.insert_and_update_institute_class_tier_students(conn, df, marks_out_of, class_id)
    
    return {
        "message": "Data uploaded successfully"
    }
