import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- 1. Load Environment Variables ---
# Load environment variables from a .env file
load_dotenv()

# Ensure the OpenAI API key is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

# --- 2. Define Constants ---
DATA_PATH = "data/"
VECTORSTORE_PATH = "vectorstore" # In a real app, you'd persist this

# --- 3. Document Ingestion and Vector Store Creation ---
def create_vector_store():
    """
    Loads documents, splits them into chunks, creates embeddings,
    and stores them in a FAISS vector store.
    """
    print("Loading documents from a custom knowledge base...")
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    print("Creating embeddings and building the FAISS vector store...")
    embeddings = OpenAIEmbeddings(api_key=api_key)
    # FAISS.from_documents can create an in-memory vector store
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    print("Vector store created successfully.")
    
    return vectorstore

# --- 4. RAG Chain Implementation ---
def create_rag_chain(vectorstore):
    """
    Creates the Retrieval-Augmented Generation (RAG) chain.
    """
    # Initialize the LLM
    llm = ChatOpenAI(api_key=api_key, model_name="gpt-3.5-turbo", temperature=0.7)

    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    # Define the prompt template
    template = """
    You are a helpful assistant who answers questions based on the provided context.
    Answer the user's question clearly and concisely using only the information from the context below.
    If the information is not available in the context, politely say that you cannot answer.
    Do not make up information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Build the RAG chain using LangChain Expression Language (LCEL)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- 5. Main Execution Block & Basic UX ---
if __name__ == "__main__":
    print("🚀 Initializing the RAG Assistant...")
    
    # 1. Create the vector store from the custom documents
    db = create_vector_store()
    
    # 2. Create the RAG chain
    qa_chain = create_rag_chain(db)
    
    print("\n✅ RAG Assistant is ready. Ask your questions about the Artemis Program.")
    print("   Type 'exit' or 'quit' to end the session.")
    
    # 3. Start the interactive CLI loop
    while True:
        user_question = input("\nYour Question: ")
        
        if user_question.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
            
        if not user_question.strip():
            print("Please enter a question.")
            continue
            
        # 4. Run the query and print the response
        print("\nThinking...")
        response = qa_chain.invoke(user_question)
        print("\nAssistant's Answer:\n" + "-"*20)
        print(response)
        print("-"*20)
