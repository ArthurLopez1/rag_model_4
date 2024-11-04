import os
from langchain_ollama import ChatOllama
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model configuration from environment variables or use default values
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.2:1b-instruct-fp16")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0))
FORMAT = os.getenv("LLM_FORMAT", None)

def initialize_llm(model_name=MODEL_NAME, temperature=TEMPERATURE, format=None):
    """
    Initialize the language model with the given parameters.
    """
    try:
        if format:
            llm = ChatOllama(model=model_name, temperature=temperature, format=format)
        else:
            llm = ChatOllama(model=model_name, temperature=temperature)
        logger.info(f"Initialized LLM with model: {model_name}, temperature: {temperature}, format: {format}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

# Initialize models
llm = initialize_llm()
llm_json_mode = initialize_llm(format="json")