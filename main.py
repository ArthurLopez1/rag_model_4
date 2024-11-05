import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.llm_models import llm
from src.vectorstore import VectorStoreManager
from src.components import run_workflow, initialize_graph
from src.training import train_on_document
from settings import Config

def main():
    # Train with a sample document
    pdf_path = "data/ersattningsmodell_vaders_2019.pdf"
    train_on_document(pdf_path)

    # Initialize the vector store for querying
    vector_store = VectorStoreManager()
    
    # Define initial state and configuration
    state = {
        "question": "På vilket sätt används halvtimmesdata för att skapa timbaserade väderbeskrivningar?",
        "max_retries": 3
    }
    config = {}

    # Run the workflow
    events = run_workflow(state, config)

    # Output results
    print("Query Results:")
    for event in events:
        print(event)

if __name__ == "__main__":
    main()