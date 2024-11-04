from src.llm_models import llm
from src.vectorstore import VectorStoreManager
from src.components import initialize_graph
from src.training import train_on_document
from settings import Config

def main():
    # Train with a sample document
    pdf_path = "data/ersattningsmodell_vaders_2019.pdf"
    train_on_document(pdf_path)

    # Initialize the vector store for querying
    vector_store = VectorStoreManager()
    
    # Query the vector store
    query = "Explain the compensation model for repeated snowfalls."
    results = vector_store.retrieve_documents(query)

    # Output results
    print("Query Results:")
    for idx, result in enumerate(results, start=1):
        print(f"{idx}. {result['content']} (Score: {result['score']})")

if __name__ == "__main__":
    main()
