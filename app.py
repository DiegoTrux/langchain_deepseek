# app.py
import streamlit as st
import tempfile
import os
import re
import json
from utils.pdf_loader import process_pdf, generate_summary, render_pdf
from utils.qa_chain import create_qa_chain
from PyPDF2 import PdfReader

def initialize_session_state():
    """Initialize session state variables"""
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'temperature': 0.7,
            'model_name': "deepseek-r1:7b",
            'hide_thinking': False,
            'hide_sources': False
        }
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'pdf_summary' not in st.session_state:
        st.session_state.pdf_summary = ""

def extract_text_from_pdf(pdf_path):
    """Extrae el texto del PDF para generar un resumen."""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def export_conversation():
    """Descarga la conversación en un archivo JSON."""
    history_json = json.dumps(st.session_state.conversation_history, indent=4)
    st.download_button(
        label="Descargar Conversación",
        data=history_json,
        file_name="conversacion.json",
        mime="application/json"
    )

def show_settings():
    """Display and handle settings sidebar"""
    with st.sidebar:
        st.title("⚙️ Configuración")
        
        # Modelo
        st.session_state.settings['model_name'] = st.selectbox(
            "Modelo",
            ["deepseek-r1:7b", "deepseek-r1:1.5b"],
            index=0 if st.session_state.settings['model_name'] == "deepseek-r1:7b" else 1
        )
        
        # Chunk settings
        st.subheader("Configuración de Chunks")
        st.session_state.settings['chunk_size'] = st.slider(
            "Tamaño de Chunk",
            min_value=100,
            max_value=2000,
            value=st.session_state.settings['chunk_size'],
            step=100
        )
        
        st.session_state.settings['chunk_overlap'] = st.slider(
            "Superposición de Chunks",
            min_value=0,
            max_value=500,
            value=st.session_state.settings['chunk_overlap'],
            step=50
        )
        
        # Temperature
        st.session_state.settings['temperature'] = st.slider(
            "Temperatura",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.settings['temperature'],
            step=0.1
        )
        
        st.subheader("Configuración de Respuestas")
        # Hide thinking
        st.session_state.settings['hide_thinking'] = st.checkbox(
            "Ocultar <think>",
            value=st.session_state.settings['hide_thinking']
        )

        # Hide sources
        st.session_state.settings['hide_sources'] = st.checkbox(
            "Ocultar fuentes",
            value=st.session_state.settings['hide_sources']
        )

def main():
    initialize_session_state()
    show_settings()
    
    st.title("📚 PDF Q&A Personalizable")
    
    # Subir archivo
    uploaded_file = st.file_uploader("Sube tu archivo PDF", type=['pdf'])
    
    if uploaded_file:
        # Resetear la conversación si se sube un nuevo archivo
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.conversation_history = []
            st.session_state.current_file = uploaded_file.name
        
        # Guardar el archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        with st.spinner('Procesando el PDF...'):
            try:
                # Procesar PDF con configuraciones personalizadas
                vector_store = process_pdf(
                    tmp_path,
                    st.session_state.settings['chunk_size'],
                    st.session_state.settings['chunk_overlap']
                )
                
                # Crear QA chain con configuraciones personalizadas
                st.session_state.qa_chain = create_qa_chain(
                    vector_store,
                    st.session_state.settings['model_name'],
                    st.session_state.settings['temperature'],
                    st.session_state.settings['hide_thinking']
                )
                
                text = extract_text_from_pdf(tmp_path)
                st.session_state.pdf_summary = generate_summary(text)
                st.session_state.pdf_summary = re.sub(r'<think>.*?</think>\n?', '', st.session_state.pdf_summary, flags=re.DOTALL)

                st.success('¡PDF procesado exitosamente!')
            except Exception as e:
                st.error(f'Error al procesar el PDF: {str(e)}')
            finally:
                os.unlink(tmp_path)

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Chat con el Documento")
        # Mostrar historial de conversación
        if st.session_state.conversation_history:
            st.subheader("Historial de Conversación")
            for q, a in st.session_state.conversation_history:
                with st.container():
                    st.markdown("**Pregunta:** " + q)
                    st.markdown("**Respuesta:** " + a)
                    st.markdown("---")
        
        # Área de preguntas
        if st.session_state.qa_chain:
            question = st.text_input("Haz una pregunta sobre tu documento:")
            if question:
                with st.spinner('Buscando la respuesta...'):
                    try:
                        result = st.session_state.qa_chain({"question": question})
                        answer = result.get("answer", "")

                        # Si hide_thinking está activo, eliminar el contenido entre <think> tags
                        if st.session_state.settings['hide_thinking']:
                            # Usar regex para eliminar todo el contenido entre <think> y </think>, incluyendo las etiquetas
                            answer = re.sub(r'<think>.*?</think>\n?', '', answer, flags=re.DOTALL)

                        # Agregar a la historia
                        st.session_state.conversation_history.append((question, answer))
                        
                        # Mostrar la respuesta actual
                        st.markdown("**Nueva Respuesta:**")
                        st.write(answer)
                        
                        # Mostrar fuentes
                        if "source_documents" in result and not st.session_state.settings['hide_sources']:
                            st.markdown("**Fuentes:**")
                            for doc in result["source_documents"]:
                                st.write(f"- Página {doc.metadata.get('page', 'N/A')}")
                            
                    except Exception as e:
                        st.error(f'Error al procesar la pregunta: {str(e)}')
        
        # Botón para limpiar historial
        if st.session_state.conversation_history:
            if st.button("Limpiar Historial"):
                st.session_state.conversation_history = []
                st.experimental_rerun()
            export_conversation()

    with col2:
        st.subheader("Vista Previa del PDF")
        if uploaded_file:
            render_pdf(tmp_path)
        
        st.subheader("Resumen Inteligente")
        st.write(st.session_state.pdf_summary)

if __name__ == "__main__":
    main()