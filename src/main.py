import os
from dotenv import load_dotenv

# --- OLD IMPORTS (some will be replaced) ---
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# +++ NEW IMPORTS FOR GOOGLE +++
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- 1. Load Environment Variables ---
load_dotenv()

# --- OLD API KEY CHECK ---
# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

# +++ NEW API KEY CHECK FOR GOOGLE +++
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")


# --- 2. Define Constants ---
DATA_PATH = "data/"
VECTORSTORE_PATH = "vectorstore"

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
    
    # --- OLD EMBEDDINGS ---
    # embeddings = OpenAIEmbeddings(api_key=api_key)
    
    # +++ NEW GOOGLE EMBEDDINGS +++
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    print("Vector store created successfully.")
    
    return vectorstore

# --- 4. RAG Chain Implementation ---
def create_rag_chain(vectorstore):
    """
    Creates the Retrieval-Augmented Generation (RAG) chain.
    """
    # --- OLD LLM ---
    # llm = ChatOpenAI(api_key=api_key, model_name="gpt-3.5-turbo", temperature=0.7)
    
    # +++ NEW GOOGLE LLM +++
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key, temperature=0.7)

    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

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

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- 5. Main Execution Block & Basic UX (NO CHANGES NEEDED HERE) ---
if __name__ == "__main__":
    print("🚀 Initializing the RAG Assistant...")
    
    db = create_vector_store()
    
    qa_chain = create_rag_chain(db)
    
    print("\n✅ RAG Assistant is ready. Ask your questions about the Artemis Program.")
    print("   Type 'exit' or 'quit' to end the session.")
    
    while True:
        user_question = input("\nYour Question: ")
        
        if user_question.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
            
        if not user_question.strip():
            print("Please enter a question.")
            continue
            
        print("\nThinking...")
        response = qa_chain.invoke(user_question)
        print("\nAssistant's Answer:\n" + "-"*20)
        print(response)
        print("-"*20)
