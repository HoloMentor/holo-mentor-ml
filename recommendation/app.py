from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from typing import Iterator
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from typing import List
from datetime import date
import pandas as pd
from io import StringIO
import os
from dotenv import load_dotenv
import psycopg2
import psqlfunctions
import embeddingfunctions
import milvusfunctions
from fastapi import FastAPI, Query, HTTPException
from typing import List
from pydantic import BaseModel
import random
import torch

load_dotenv(dotenv_path="../env")

app = FastAPI()

print("app loaded")

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

collection_name = "questioncollection"
collection = milvusfunctions.create_or_load_collection(collection_name)

class CustomDenseEmbedding(Embeddings):
    def __init__(self, model_name='sentence-transformers/stsb-xlm-r-multilingual'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_documents(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return embeddings

    def embed_query(self, text):
        inputs = self.tokenizer([text], return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return embedding.tolist()

print("CustomDenseEmbedding loaded")
dense_embedding_instance = CustomDenseEmbedding()
print("CustomDenseEmbedding instance created")

@app.get("/")
async def root():
    return {"message": "Welcome to the recommendation engine."}

@app.get("/generate_quiz/{class_id}/{student_id}")
async def generate_quiz(class_id: int , student_id: int ):
    print("generate_quiz")
    collection = milvusfunctions.create_or_load_collection(collection_name="questioncollection")
    try:
        check_already_quiz_generated = psqlfunctions.check_custom_quiz_already_exists(class_id, student_id, conn)
        if check_already_quiz_generated:
            raise HTTPException(status_code=409, detail="Custom quiz already generated for this student this week.")

        else:
            wrong_mcq_question_ids = psqlfunctions.load_wrong_questions(class_id, student_id, conn)

            if not wrong_mcq_question_ids:
                wrong_mcq_question_ids = [random.randint(1, 100) for _ in range(10)]
                
            print(f"Wrong MCQ question IDs: {wrong_mcq_question_ids}")
            wrong_mcq_questions = psqlfunctions.get_mcq_questions(wrong_mcq_question_ids, conn)
            mcq_question_ids = [q['id'] for q in wrong_mcq_questions]
            print(f"MCQ question IDs: {mcq_question_ids}")

            all_similar_records = []
            for question in wrong_mcq_questions:
                question_text = question['question'][0]['content'][0]['text']
                question_topic = question['topic']
                question_sub_topic = question['sub_topic']
                question_embedded = dense_embedding_instance.embed_documents(question_text)

                similar_records = milvusfunctions.retrieval_similarity(
                    collection, question_embedded, top_k=5,
                    class_id=class_id
                )
                all_similar_records.append(similar_records)

            unique_indices = list(set(idx for sublist in all_similar_records for idx in sublist))
            print(f"Unique indices: {unique_indices}")

            res = psqlfunctions.insert_custom_quiz(class_id, student_id, unique_indices, conn)

            if res:
                return {"message": "Custom quiz generated successfully."}
            else:
                raise HTTPException(status_code=500, detail="Failed to generate custom quiz.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/insert_quiz/{quiz_id}")
async def insert_quiz(quiz_id: int):
    print("insert_quiz")
    try:
        questions = psqlfunctions.load_mcq_question_by_id(conn, quiz_id)
        if not questions:
            raise HTTPException(status_code=600, detail=f"No questions found for quiz_id {quiz_id}")

        wrong_mcq_questions = psqlfunctions.get_mcq_questions([quiz_id], conn)
        if not wrong_mcq_questions:
            raise HTTPException(status_code=600, detail=f"No MCQ questions found for quiz_id {quiz_id}")

        texts = [q['question'][0]['content'][0]['text'] for q in wrong_mcq_questions]

        embeddings_with_metadata = embeddingfunctions.embed_documents(
            wrong_mcq_questions, dense_embedding_instance
        )

        data = [
            {
                "id": embeddings_with_metadata[i]["id"],
                "embedding": embeddings_with_metadata[i]["embedding"],
                "text": embeddings_with_metadata[i]["text"],
                "topic": embeddings_with_metadata[i]["metadata"]["topic"],
                "sub_topic": embeddings_with_metadata[i]["metadata"]["sub_topic"],
                "class_id": embeddings_with_metadata[i]["metadata"]["class_id"]
            }
            for i in range(len(embeddings_with_metadata))
        ]

        collection = milvusfunctions.create_or_load_collection(collection_name="questioncollection")
        collection.insert(data)

        return {"message": f"Data inserted into collection 'questioncollection' successfully."}

    except HTTPException as http_exc:
        raise http_exc  # Rethrow HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))