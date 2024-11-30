from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from typing import Iterator
import torch
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from pymilvus.client.abstract import BaseRanker
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

load_dotenv(dotenv_path="../env")

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

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')

collection_name = "questioncollection"
collection = milvusfunctions.create_or_load_collection(collection_name="questioncollection")

class CustomReranker(BaseRanker):
    def __init__(self):
        super().__init__()

    def rerank(self, query_text: str, retrieved_texts: List[str]) -> List[str]:
        pairs = [[query_text, retrieved_text] for retrieved_text in retrieved_texts]
        results = []

        with torch.no_grad():
            for pair in pairs:
                inputs = tokenizer(pair, padding=True, truncation=True, return_tensors='pt', max_length=512)
                outputs = model(**inputs, return_dict=True)
                logits = outputs.logits.squeeze(0)  #
                if len(logits) == 1:
                    score = logits[0].item()  
                else:
                    score = abs(logits[0] - logits[1]).item()  
                results.append((pair[1], score))  

        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

        reranked_texts = [text for text, score in sorted_results]

        return reranked_texts

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


class_id = 1
student_id = 3

check_already_quiz_generated = psqlfunctions.check_custom_quiz_already_exists(class_id, student_id, conn)
if not check_already_quiz_generated:
    print("No custom quiz found for this student in this class this week.")
    wrong_mcq_question_ids = psqlfunctions.load_wrong_questions(class_id, student_id, conn)
    wrong_mcq_questions = psqlfunctions.get_mcq_questions(wrong_mcq_question_ids, conn)
    texts = [q['question'][0]['content'][0]['text']for q in wrong_mcq_questions]
    print(wrong_mcq_questions)
    mcq_question_ids = [q['id'] for q in wrong_mcq_questions]
    print(mcq_question_ids)

    sparse_embedding_func = BM25SparseEmbedding(corpus=texts)
    dense_embedding_instance = CustomDenseEmbedding()


    all_similar_records = []
    for question in wrong_mcq_questions:
        question_text = question['question'][0]['content'][0]['text']
        question_topic = question['topic']
        question_sub_topic = question['sub_topic']
    
        similar_records = milvusfunctions.retrieval_similarity(
            collection, question_text, top_k=5, reranker=CustomReranker(),
            topic=question_topic, sub_topic=question_sub_topic,
            class_id=class_id, dense_embedding_instance=dense_embedding_instance,
            sparse_embedding_func=sparse_embedding_func
        )
        all_similar_records.append(similar_records)

    # Flatten the list of lists and remove duplicates
    unique_indices = list(set(idx for sublist in all_similar_records for idx in sublist))

    # insterting data into custom quiz table
    psqlfunctions.insert_custom_quiz(class_id, student_id, unique_indices, conn)

else:
    print("Custom quiz already exists for this student in this class this week.")
    exit()







# data insertion

# embeddings_with_metadata = embeddingfunctions.embed_documents(wrong_mcq_questions,  dense_embedding_instance, sparse_embedding_func)
# data = [
#     {
#         "id": embeddings_with_metadata[i]["id"],
#         "embedding": embeddings_with_metadata[i]["embedding"],
#         "sparse": sparse_embedding_func.embed_documents([embeddings_with_metadata[i]["text"]])[0],  # Generate sparse embedding
#         "text": embeddings_with_metadata[i]["text"],
#         "topic": embeddings_with_metadata[i]["metadata"]["topic"],
#         "sub_topic": embeddings_with_metadata[i]["metadata"]["sub_topic"],
#         "class_id": embeddings_with_metadata[i]["metadata"]["class_id"]
#     }
#     for i in range(len(embeddings_with_metadata))
# ]

# collection = milvusfunctions.create_or_load_collection(collection_name="questioncollection")
# collection_name = "questioncollection"

# collection.insert(data)
# print(f"Data inserted into collection '{collection_name}' successfully.")