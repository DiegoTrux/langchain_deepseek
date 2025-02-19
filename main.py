from config.ollama_config import get_llm

def main():
    print("Inicializando Langchain con Ollama (deepseek-r1:7b)...")
    llm = get_llm()
    
    # Ejemplo de uso básico
    response = llm("¡Hola! ¿Cómo estás?")
    print("\nRespuesta completa:", response)

if __name__ == "__main__":
    main()
