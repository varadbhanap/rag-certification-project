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

