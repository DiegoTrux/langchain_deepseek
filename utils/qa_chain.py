# utils/qa_chain.py
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

def create_qa_chain(vector_store, model_name, temperature, hide_thinking):
    """Create QA chain with Ollama and custom parameters"""
    # Crear el prompt template
    base_template = """
    {context}
    
    Pregunta: {question}
    
    Respuesta útil y concisa:"""
    
    if hide_thinking:
        base_template = base_template
    else:
        base_template = "<think>Analizando el contexto y la pregunta para proporcionar una respuesta precisa.</think>\n" + base_template
    
    prompt_template = """
    Contexto de la conversación anterior:
    {chat_history}
    """ + base_template
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["chat_history", "context", "question"]
    )
    
    # Inicializar Ollama
    llm = Ollama(
        model=model_name,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        temperature=temperature
    )
    
    # Crear memoria
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Crear cadena de QA con memoria
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": PROMPT,
            "document_prompt": PromptTemplate(
                input_variables=["page_content"],
                template="{page_content}"
            )
        },
        return_source_documents=True
    )
    
    return qa_chain