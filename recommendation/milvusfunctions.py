from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever

connections.connect("default", host="localhost", port="19530")

def create_or_load_collection(collection_name: str):
    if not utility.has_collection(collection_name):
        field1 = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False)
        field2 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)  
        field3 = FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR)  
        field4 = FieldSchema(name="text", dtype=DataType.VARCHAR, is_primary=False, max_length=5000)  # Question content
        field5 = FieldSchema(name="topic", dtype=DataType.VARCHAR, is_primary=False, max_length=256)  # Topic
        field6 = FieldSchema(name="sub_topic", dtype=DataType.VARCHAR, is_primary=False, max_length=256)  # Sub-topic
        field7 = FieldSchema(name="class_id", dtype=DataType.INT64, is_primary=False)  # Class ID

        schema = CollectionSchema(
            fields=[field1, field2, field3, field4, field5, field6, field7],
            description="Collection for storing MCQ question embeddings with metadata",
            enable_dynamic_field=True
        )

        collection = Collection(name=collection_name, schema=schema)

        embedding_index_params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        collection.create_index(field_name="embedding", index_params=embedding_index_params)

        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        collection.create_index(field_name="sparse", index_params=sparse_index)

        collection.flush()

    else:
        collection = Collection(name=collection_name)

    return collection


def retrieval_similarity(collection, query, top_k=5, reranker=None, topic=None, sub_topic=None, class_id=None, dense_embedding_instance=None, sparse_embedding_func=None):
    sparse_search_params = {"metric_type": "IP"}
    dense_search_params = {"metric_type": "IP", "params": {}}

    retriever = MilvusCollectionHybridSearchRetriever(
                collection=collection,
                rerank=reranker,
                anns_fields=["embedding", "sparse"],  # Fields to search in Milvus
                field_embeddings=[dense_embedding_instance, sparse_embedding_func],  # Embedding functions
                field_search_params=[dense_search_params, sparse_search_params],  # Search parameters for embeddings
                top_k=5,  
                text_field="text",  
                filter_fields=["topic", "sub_topic", "class_id"],  
            )

    filter_criteria = {
        "topic": topic,
        "sub_topic": sub_topic,
        "class_id": class_id
    }

    retrieved_docs = retriever.invoke(query, filters=filter_criteria)
    extracted_ids = [doc.metadata['id'] for doc in retrieved_docs]

    return extracted_ids