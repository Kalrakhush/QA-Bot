import streamlit as st
import logging
import tempfile
from components.data_loader import load_and_enhance_documents
from components.llm import initialize_embeddings, initialize_llm
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
import nest_asyncio
from llama_index.core import Settings


llm=initialize_llm()
embed_model=initialize_embeddings()
Settings.llm=llm
Settings.embed_model=embed_model

# Apply asynchronous I/O
nest_asyncio.apply()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Title setup
st.set_page_config(page_title="Financial Assistant", page_icon="📊")

# Financial Analysis page title
st.title("Financial Data Analysis")
st.write("Upload a PDF document to analyze.")

# Upload PDF file
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

# Check if the index is already in session state
if 'index' not in st.session_state:
    st.session_state['index'] = None

if uploaded_files and st.session_state['index'] is None:
    with st.spinner("Processing files..."):
        temp_files = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_files.append(temp_file.name)
                logger.info(f"Temporary file created: {temp_file.name}")

        if temp_files:
            logger.info("Enhancing documents and extracting data...")
            enhanced_documents = load_and_enhance_documents(temp_files)
            documents = [Document(text=doc.text, metadata=doc.metadata) for doc in enhanced_documents]
            logger.info(f"{len(documents)} documents loaded and enhanced.")

            # Initialize embedding model and LLM
            logger.info("Initializing embedding model and LLM...")
            embed_model = initialize_embeddings()
            llm = initialize_llm()
            logger.info("Embedding model and LLM initialized.")

            # Parse nodes
            logger.info("Parsing nodes from documents...")
            node_parser = SentenceWindowNodeParser.from_defaults(
                window_size=40,
                window_metadata_key="window",
                original_text_metadata_key="original_sentence",
            )
            nodes = node_parser.get_nodes_from_documents(documents)
            logger.info(f"{len(nodes)} nodes created from documents.")

            # Build index only if nodes are successfully created
            if nodes:
                logger.info("Building index...")
                st.session_state['index'] = VectorStoreIndex(nodes, use_async=True)
                logger.info("Index built successfully.")
            else:
                logger.warning("No nodes were created. Skipping index building.")

# Query engine setup
if st.session_state['index']:
    query_engine = st.session_state['index'].as_chat_engine(
        similarity_k=3,
        chat_mode="context",
        node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")]
    )

    user_query = st.text_input("Enter your query:")
    if user_query:
        response = query_engine.chat(user_query)
        if response:
            logger.info("Query processed successfully.")
            st.write(f"Response: {response}")
        else:
            logger.warning("No response returned for the query.")
