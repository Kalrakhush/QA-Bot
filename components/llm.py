from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from config.config import Config
from llama_index.core import Settings
import os
import streamlit as st
# os.environ["HF_token"] = st.secrets["HF_TOKEN"]
# google_api_key = st.secrets["GOOGLE_API_KEY"]
# os.environ["HF_token"] = os.getenv["HF_TOKEN"]
google_api_key = os.getenv("GOOGLE_API_KEY")
def initialize_embeddings():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return embed_model

def initialize_llm():
    llm = Gemini(
        model="models/gemini-1.5-flash",
        temperature=0.3,
        system_prompt="""You are a financial data expert trained to analyze structured P&L data and respond to user queries. Your responsibilities include:

Extracting relevant financial insights (e.g., income, expenses, profit margins, operating costs).
Retrieving data efficiently from indexed embeddings stored in a vector database.
Generating precise and coherent answers, formatted as well-structured tables for clarity.

Answering Requirements:
Include all relevant metrics in tabular format.
Clearly label columns and rows for easy interpretation (e.g., Metric, Value, Period).
Ensure responses are directly derived from the indexed P&L data for accuracy.

Constraints:
Only answer based on the given financial data; avoid speculative or generic responses.
Handle large datasets efficiently, ensuring no significant delays in response times.
Act as a professional financial assistant and prioritize user clarity and accuracy in every interaction.""",
        top_p=0.8,
        api_key=google_api_key,
    )
    return llm