import os
import pdfplumber
from llama_index.core import SimpleDirectoryReader
import json

def parse_pdf_tables(file_path):
    tables = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            for table in page_tables:
                tables.append(table)
    return tables

def extract_tables(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    tables = []
    if file_ext == ".pdf":
        tables = parse_pdf_tables(file_path)
    return tables

def load_and_enhance_documents(file_list):
    loader = SimpleDirectoryReader(input_files=file_list)
    documents = loader.load_data()

    for document, file_path in zip(documents, file_list):
        tables = extract_tables(file_path)
        if tables:
            document.metadata["tables"] = json.dumps(tables)
    return documents
