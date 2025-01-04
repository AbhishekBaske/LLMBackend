import modal
from modal import Volume

app = modal.App("rag-api")

image = modal.Image.debian_slim(python_version="3.10").pip_install("pinecone-client", "transformers", "sentence-transformers", "torch","accelerate","fastapi")
model_store_path = "/model"
volume = Volume.from_name("model-weights-vol", create_if_missing=True)

@app.function(image=image, gpu=modal.gpu.T4(count=1), volumes={model_store_path: volume})
def rag(query: str):


    from pinecone import Pinecone
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer

    # Step 1: Initialize Pinecone
    pc = Pinecone(api_key="pcsk_6AxB1N_DA5B7sPFqe7StFxkXcPna9eWrmHzDB1x1QHbqKnRfzXjJxtSC2zFzS2ojqFaSfB")
    # Step 2: Connect to Pinecone index
    index_name = "quickstart"
    index = pc.Index(index_name)

    # Step 3: Set up SentenceTransformer for embeddings
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # Step 4: Set up HuggingFace Transformers pipeline
   
    llm_pipeline = pipeline(
        "text-generation", 
        model=f"{model_store_path}/model_weights/model", 
        tokenizer=f"{model_store_path}/model_weights/tokenizer",
        torch_dtype="bfloat16", 
        device_map="auto",
        pad_token_id=128001,
    )

    # Step 5: Define the query processing and retrieval logic
    def query_rag_pipeline(query: str):
        # Generate query embeddings
        query_embedding = embedding_model.encode(query).tolist()

        # Perform similarity search in Pinecone
        search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

        # Gather the retrieved documents
        documents = []
        for match in search_results["matches"]:
            documents.append(match["metadata"].get("text", ""))

        # Concatenate retrieved documents into context for generation
        context = " ".join(documents)

        # Generate a response using the LLM
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        llm_response = llm_pipeline(
            prompt, 
            max_new_tokens=100,  # Control output length
            num_return_sequences=1,
            truncation=True,
        )
        answer = llm_response[0]["generated_text"]

        return answer

    # Step 6: Query the RAG pipeline
    response = query_rag_pipeline(query)

    # Step 7: Output the response and sources
    print("RAG Response:")
    print(response)

if __name__ == "__main__":
    with app.run():
        rag()