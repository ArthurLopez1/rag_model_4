import os
from dotenv import load_dotenv

class Config:
    DATA_PATH = "../data/"
    MODEL_PATH = "path/to/model"
    VECTOR_STORE_PATH = "path/to/vectorstore"
    MAX_RETRIES = 3
    SEARCH_RESULTS = 3

    def __init__(self):
        load_dotenv()
        self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
        self.LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
        
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "local-llama32-rag"

