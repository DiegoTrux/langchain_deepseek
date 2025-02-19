from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def get_llm():
    """Initialize and return Ollama LLM with deepseek-r1:7b"""
    llm = Ollama(
        model="deepseek-r1:7b",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        temperature=0.7,
        base_url="http://localhost:11434"
    )
    return llm
