def embed_documents(wrong_mcq_questions, dense_embedding_instance):
    print("Embedding documents...")
    embeddings_with_metadata = []

    for doc in wrong_mcq_questions:
        question_id = doc.get('id') 
        content_text = " ".join([content.get('text', '') for content in doc.get('question', [{}])[0].get('content', [])])
        
        embeddings_with_metadata.append({
            "id": question_id,
            "text": content_text,
            "embedding": dense_embedding_instance.embed_documents(content_text),  # Assuming embedding_func is defined
            "metadata": {
                "topic": doc.get('topic'),
                "sub_topic": doc.get('sub_topic'),
                "class_id": doc.get('class_id')
            }
        })

    for doc in embeddings_with_metadata:
        doc["embedding"] = doc["embedding"].tolist()

    return embeddings_with_metadata
