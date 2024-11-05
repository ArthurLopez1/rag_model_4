from pathlib import Path
import streamlit as st
from src.components import run_workflow
from src.vectorstore import VectorStoreManager
from src.training import train_on_document
from settings import Config
from frontend.constants import Color

# Initialize the configuration
config = Config()

# Initialize the vector store manager
vec = VectorStoreManager()

# Load custom CSS
def read_css():
    css_path = Path(__file__).parent / "frontend" / "style.css"
    with open(css_path) as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

read_css()  # Call CSS function at the start of the script

# Streamlit app title
st.title("Chat with Klimator")  # Main title

# Sidebar for training the model
st.sidebar.title("Training")
pdf_path = st.sidebar.text_input("PDF Path", "data/ersattningsmodell_vaders_2019.pdf")

if st.sidebar.button("Train Model"):
    train_on_document(pdf_path)
    st.sidebar.success("Model trained successfully!")

# Main interface for asking questions
st.header("Ask a question about the document:")
question = st.text_input("Enter your question here:")
max_retries = st.number_input("Max Retries", min_value=1, max_value=10, value=3)

if st.button("Get Answer"):
    # Define initial state and configuration
    state = {
        "question": question,
        "max_retries": max_retries
    }

    # Run the workflow
    events = run_workflow(state, config)

    # Output the final generated answer
    final_state = events[-1] if events else {}
    generated_answer = final_state.get("generation", "No answer generated.")
    st.write("Final Generated Answer:")
    st.write(generated_answer)
