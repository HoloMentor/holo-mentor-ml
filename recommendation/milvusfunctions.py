from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever

connections.connect("default", host="host.docker.internal", port="19530")


def create_or_load_collection(collection_name: str):
    print(f"Checking if collection '{collection_name}' exists...")
    if not utility.has_collection(collection_name):
        field1 = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False)
        field2 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)  
        field4 = FieldSchema(name="text", dtype=DataType.VARCHAR, is_primary=False, max_length=5000)  # Question content
        field5 = FieldSchema(name="topic", dtype=DataType.VARCHAR, is_primary=False, max_length=256)  # Topic
        field6 = FieldSchema(name="sub_topic", dtype=DataType.VARCHAR, is_primary=False, max_length=256)  # Sub-topic
        field7 = FieldSchema(name="class_id", dtype=DataType.INT64, is_primary=False)  # Class ID

        schema = CollectionSchema(
            fields=[field1, field2, field4, field5, field6, field7],
            description="Collection for storing MCQ question embeddings with metadata",
            enable_dynamic_field=True
        )

        collection = Collection(name=collection_name, schema=schema)

        embedding_index_params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        collection.create_index(field_name="embedding", index_params=embedding_index_params)

        collection.flush()

    else:
        collection = Collection(name=collection_name)

    return collection


def retrieval_similarity(collection, query_vectors, top_k=5, class_id=None):
    print(collection)
    dense_search_params = {"metric_type": "IP", "params": {}}

    collection.load()
    # Perform the search
    res = collection.search(
        data=[query_vectors],
        limit=top_k,
        param={"metric_type": "IP"},
        anns_field="embedding",
        output_fields=["id", "text", "topic", "sub_topic", "class_id"],
    )
    
    # Extract results from the search response
    extracted_records = []
    for hit in res[0]:  # Assuming `res` is a list of lists, with res[0] being the primary list of hits
        record = {
            "id": hit.id,
            "text": hit.entity.get("text"),
            "topic": hit.entity.get("topic"),
            "sub_topic": hit.entity.get("sub_topic"),
            "class_id": hit.entity.get("class_id"),
        }
        extracted_records.append(record)
    
    # Filter by class_id if provided
    if class_id is not None:
        filtered_records = [record for record in extracted_records if record["class_id"] == class_id]
    else:
        filtered_records = extracted_records
    
    return [record["id"] for record in filtered_records]
    
    