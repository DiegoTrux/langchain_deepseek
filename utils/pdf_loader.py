# utils/pdf_loader.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
import tempfile
from langchain_community.llms import Ollama

def process_pdf(pdf_file, chunk_size, chunk_overlap):
    """Process PDF file and create vector store"""
    # Cargar el PDF
    loader = PyPDFLoader(pdf_file)
    pages = loader.load()
    
    # Dividir el texto en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_documents(pages)
    
    # Crear embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Crear un directorio temporal para Chroma
    persist_directory = tempfile.mkdtemp()
    
    # Inicializar el cliente de Chroma
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Crear vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        persist_directory=persist_directory
    )
    
    return vector_store

def generate_summary(text):
    """Genera un resumen del texto usando DeepSeek."""
    llm = Ollama(model="deepseek-r1:7b")
    summary = llm.invoke(f"Resume en español el siguiente texto de forma concisa: {text}")
    return summary

import fitz
import streamlit as st
from PIL import Image

def render_pdf(pdf_path):
    """Muestra una vista previa de todas las páginas del PDF en la app usando imágenes."""
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    
    for i, img in enumerate(images):
        st.image(img, caption=f"Página {i+1}", use_column_width=True)