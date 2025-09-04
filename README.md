# rag-certification-project
# Project: RAG-Based Q&A Assistant for Custom Documents

This project is a submission for the **Agentic AI Developer Certification Program by Ready Tensor (Module 1)**. It implements a simple yet powerful Retrieval-Augmented Generation (RAG) pipeline using LangChain to answer questions based on a custom knowledge base.

## 🚀 Overview

The assistant is designed to answer questions about the **NASA Artemis Program**. It uses a text document (`data/artemis_program.txt`) as its source of truth. The core of the project is a LangChain-based pipeline that connects a user prompt, a vector store retriever, and an LLM to generate accurate, context-aware answers.

### Core Features
- **Document Ingestion**: Loads text documents from a specified directory.
- **Vector Store**: Chunks the documents, generates embeddings using OpenAI, and stores them in a FAISS in-memory vector store.
- **RAG Pipeline**: Utilizes LangChain Expression Language (LCEL) to create a clean `prompt → retriever → LLM → response` flow.
- **CLI Interface**: A simple and interactive command-line interface (CLI) for users to ask questions.

## 🛠️ Technology Stack

- **Framework**: [LangChain](https://www.langchain.com/)
- **LLM**: OpenAI `gpt-3.5-turbo`
- **Embeddings**: OpenAI `text-embedding-ada-002`
- **Vector Store**: [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss) - In-memory for simplicity.
- **Document Loading**: `Unstructured`

## ⚙️ Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Prerequisites
- Python 3.9 or higher
- An [OpenAI API Key](https://platform.openai.com/api-keys)

### 2. Clone the Repository
```bash
git clone <your-repo-url>
cd rag-certification-project
```

### 3. Install Dependencies
It is highly recommended to use a virtual environment.
```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a file named `.env` in the root directory of the project and add your OpenAI API key:
```
OPENAI_API_KEY="your_openai_api_key_here"
```

## ▶️ How to Run

Once the setup is complete, you can run the assistant from the root directory:

```bash
python src/main.py
```

The script will first process the documents in the `/data` folder and build the vector store. Once it's ready, you can start asking questions in the terminal.

## 🧪 Example Queries & Responses

Here are a few example interactions to test the assistant's retrieval and response quality.

**Query 1: What is the main goal of the Artemis program?**
> **Assistant's Answer:**
> --------------------
> The primary goal of the Artemis program is to return humans to the Moon, specifically the lunar south pole, by 2026, and to establish a long-term human presence on and around it.
> --------------------

**Query 2: What is the Gateway?**
> **Assistant's Answer:**
> --------------------
> The Gateway is a small space station or lunar outpost that will orbit the Moon. It is designed to serve as a command center, science lab, and short-term habitat for astronauts, acting as a critical piece of infrastructure for missions to the lunar surface and eventually to Mars.
> --------------------

**Query 3: Which mission will land the first woman on the Moon?**
> **Assistant's Answer:**
> --------------------
> The Artemis III mission, planned for 2026, will be the mission that lands the first woman and the first person of color on the Moon's south pole.
> --------------------

**Query 4 (Out of context): What is the capital of France?**
> **Assistant's Answer:**
> --------------------
> I cannot answer that question as the information is not available in the provided context.
> --------------------

## Future Enhancements (Optional)
- **Add Memory**: Integrate `ConversationBufferMemory` to allow for follow-up questions and a more conversational experience.
- **Persistent Vector Store**: Save the FAISS index to disk so it doesn't need to be recreated every time the script runs.
- **Streamlit/Gradio UI**: Build a simple web-based user interface for a more accessible experience.
