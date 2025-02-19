from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

class PDFConversationMemory:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    def add_interaction(self, question, answer):
        self.memory.chat_memory.add_message(HumanMessage(content=question))
        self.memory.chat_memory.add_message(AIMessage(content=answer))
    
    def get_history(self):
        return self.memory.chat_memory.messages
    
    def clear(self):
        self.memory.clear()