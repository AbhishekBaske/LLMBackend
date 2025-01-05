import modal
from modal import Volume

# Define the Modal app and resources
app = modal.App("rag-api")

# Modal image setup with necessary dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "pinecone-client", "transformers", "sentence-transformers", "torch", "accelerate", "fastapi"
)

# Model storage volume
model_store_path = "/model"
volume = Volume.from_name("model-weights-vol", create_if_missing=True)

# Define the RAG function
@app.function(image=image, gpu=modal.gpu.T4(count=1), volumes={model_store_path: volume})
def rag(query: str):
    import torch
    from pinecone import Pinecone
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer

    # Initialize Pinecone
    pinecone_api_key = "pcsk_6AxB1N_DA5B7sPFqe7StFxkXcPna9eWrmHzDB1x1QHbqKnRfzXjJxtSC2zFzS2ojqFaSfB"  # Replace with your Pinecone API key
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "quickstart"
    index = pc.Index(index_name)

    # Load the embedding model
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Load the HuggingFace model for text generation
    llm_pipeline = pipeline(
        "text-generation",
        model=f"{model_store_path}/model_weights/model",
        tokenizer=f"{model_store_path}/model_weights/tokenizer",
        torch_dtype=torch.float32,
        device_map="auto",
        pad_token_id=128001,
    )

    # Define the query pipeline
    def query_rag_pipeline(query: str):
        # Generate query embeddings
        query_embedding = embedding_model.encode(query).tolist()

        # Perform similarity search in Pinecone
        search_results = index.query(vector=query_embedding, top_k=50, include_metadata=True)

        # Gather all retrieved documents
        documents = [match["metadata"].get("text", "") for match in search_results["matches"]]

        # Combine all documents into a single context
        context = " ".join(documents)

        # Check token limits
        max_context_tokens = 1024  # Adjust based on model capacity
        if len(context.split()) > max_context_tokens:
            context = " ".join(context.split()[:max_context_tokens])

        # Advanced prompt
        prompt = (
            f"You are an expert assistant capable of providing accurate and concise answers "
            f"based on the given context. Use the context to answer the question clearly, "
            f"without adding information that isn't supported by the context.\n\n"
            f"read all the context carefully and answer generate in list of poinst if possible. "
            f"Context: {context}\n\n"
            f"Task:\n"
            f"1. Analyze the provided context carefully.\n"
            f"2. Answer the question concisely and accurately.\n"
            f"3. give answers from the context . de domain specific if it not present in context say i can not find context in coal mine regulation 2017.\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        # Generate response using the LLM
        llm_response = llm_pipeline(
            prompt,
            max_new_tokens=300,  # Adjust for output length
            num_return_sequences=1,
        )
        return llm_response[0]["generated_text"]

    # Query the RAG pipeline
    response = query_rag_pipeline(query)

    # Output the result
    print("RAG Response:")
    print(response)
    return response

# Entry point for running the app
if __name__ == "__main__":
    with app.run():
        query = input("Enter your query: ")
        print(rag(query))
