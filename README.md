# Research Paper Assistant – AI-Powered RAG Application

An AI-powered Research Paper Assistant that allows users to upload research papers in PDF format and ask questions about their content. The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from uploaded documents and generate context-aware answers using Google's Gemini API.

## Project Overview

Reading and understanding lengthy research papers can be time-consuming. This project simplifies the process by allowing users to interact with research papers through a conversational interface.

Users can upload research papers, enter questions, and receive relevant answers generated from the content of their documents.

## Features

- **PDF Upload:** Upload research papers in PDF format.
- **Text Extraction:** Extract text from PDF documents using PyMuPDF.
- **Text Chunking:** Divide extracted text into smaller chunks with overlapping content.
- **Text Embeddings:** Generate vector embeddings using Google's Gemini Embedding API.
- **Vector Database:** Store and search embeddings using FAISS.
- **Semantic Search:** Retrieve the most relevant text chunks based on the user's question.
- **AI-Powered Answers:** Generate context-aware responses using Gemini.
- **Interactive Interface:** Simple and user-friendly Streamlit application.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Backend development |
| Streamlit | User interface |
| Google Gemini API | AI-powered response generation |
| Gemini Embeddings | Text vectorization |
| PyMuPDF | PDF text extraction |
| FAISS | Vector similarity search |
| NumPy | Numerical operations |
| python-dotenv | Environment variable management |

## How It Works

The application follows a Retrieval-Augmented Generation (RAG) pipeline.

1. **Upload PDF:** The user uploads a research paper.
2. **Text Extraction:** PyMuPDF extracts textual content from the uploaded PDF.
3. **Chunking:** The extracted text is divided into smaller overlapping chunks.
4. **Embedding Generation:** Gemini converts text chunks into numerical vector embeddings.
5. **Vector Storage:** FAISS stores the embeddings for efficient similarity search.
6. **Question Processing:** The user enters a question about the uploaded research paper.
7. **Retrieval:** FAISS retrieves the most relevant chunks.
8. **Answer Generation:** Gemini generates an answer using the retrieved context.

## Project Structure

```text
ResearchPaperAssistant/
│
├── app.py                # Main Streamlit application
├── pdf_processor.py      # PDF text extraction and processing
├── requirements.txt      # Project dependencies
├── .env                  # Gemini API key (not committed)
├── .gitignore            # Excluded files
└── README.md             

## Requirements

The project uses the following Python libraries:

```text
streamlit
google-genai
python-dotenv
numpy
faiss-cpu
PyMuPDF
```

## Key Concepts Implemented

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Text Embeddings
- Semantic Similarity Search
- Vector Databases
- PDF Processing
- Prompt Engineering

## Future Enhancements

- Conversation history for follow-up questions.
- Support for additional document formats.
- Source citations with page numbers.
- Chat history export.
- Enhanced document comparison capabilities.
