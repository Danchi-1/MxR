import sys
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.schema import Document
from langchain.agents import initialize_agent, AgentType
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model=os.getenv("LLM_MODEL_NAME", "nvidia/nemotron-3-nano-30b-a3b:free"),
    temperature=0
)

embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
loader = DirectoryLoader("./data", glob="**/*")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
docs = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="./chroma_db",
)
retriever = vectorstore.as_retriever()

def save_to_memory(text):
    vectorstore.add_documents([Document(page_content=text)])
    vectorstore.persist()

@tool
def get_flight_schedule(query: str):
    """Get flight details between cities"""
    if "lagos" in query.lower() and "nairobi" in query.lower():
        return "Flight Lagos → Nairobi: 5 hours, $450 one-way"
    return "Route not available"

@tool
def get_hotel_schedule(query: str):
    """Get Hotel price in a city"""
    return "Hotel in Nairobi: $120 per night"

@tool
def convert_currency(query: str):
    """Convert USD to NGN"""
    return "1500 NGN per USD"

@tool
def rag_search(query: str):
    """Search internal knowledge base"""
    docs = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in docs[:3]])

tools = [
    get_flight_schedule,
    get_hotel_schedule,
    convert_currency,
    rag_search,
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

def main():
    if len(sys.argv) < 2:
        print("No query provided. Please provide a query")
        return
    query = sys.argv[1]
    save_to_memory(query)
    response = agent.run(query)
    save_to_memory(f"Model reply: {response}")
    history_docs = vectorstore.similarity_search(query, k=10)

    print(f"=> History Docs: ")
    for doc in history_docs:
        print(doc.page_content)
    print(f"Last Query: {query}")
    print(f"Final Response: {response}")

if __name__ == "__main__":
    main()