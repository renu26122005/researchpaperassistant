import os
import tempfile
import faiss
import numpy as np
import streamlit as st

from dotenv import load_dotenv
from google import genai
from google.genai import types

from pdf_processor import extract_text_from_pdf, create_chunks


# ---------------------------------
# Configuration
# ---------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not found. Please check your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

EMBEDDING_MODEL = "gemini-embedding-2"
GENERATION_MODEL = "gemini-3.8-flash"


# ---------------------------------
# Page Setup
# ---------------------------------

st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 GenAI Research Paper Assistant")

st.write(
    "Upload multiple research papers and ask questions "
    "across all your documents using Generative AI."
)


# ---------------------------------
# Initialize Session State
# ---------------------------------

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "chunk_sources" not in st.session_state:
    st.session_state.chunk_sources = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


# ---------------------------------
# Multiple PDF Upload
# ---------------------------------

st.header("1. Upload Research Papers")

uploaded_files = st.file_uploader(
    "Choose your research papers",
    type=["pdf"],
    accept_multiple_files=True
)


# ---------------------------------
# Process Uploaded Documents
# ---------------------------------

if st.button("Process Research Papers"):

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")

    else:

        all_chunks = []
        chunk_sources = []

        with st.spinner("Processing research papers..."):

            for uploaded_file in uploaded_files:

                st.write(f"📄 Processing: {uploaded_file.name}")

                temp_path = None

                try:

                    # Save PDF temporarily
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(uploaded_file.getbuffer())
                        temp_path = temp_file.name

                    # Extract text
                    text = extract_text_from_pdf(temp_path)

                    if not text.strip():
                        st.warning(
                            f"No readable text found in {uploaded_file.name}"
                        )
                        continue

                    # Create chunks
                    chunks = create_chunks(text)

                    if not chunks:
                        st.warning(
                            f"No chunks generated for {uploaded_file.name}"
                        )
                        continue

                    # Combine chunks
                    all_chunks.extend(chunks)

                    # Store source filenames
                    chunk_sources.extend(
                        [uploaded_file.name] * len(chunks)
                    )

                    st.success(
                        f"{uploaded_file.name}: "
                        f"{len(chunks)} chunks created."
                    )

                except Exception as e:
                    st.error(
                        f"Error processing {uploaded_file.name}: {e}"
                    )

                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

            # ---------------------------------
            # Generate Embeddings
            # ---------------------------------

            if all_chunks:

                try:

                    with st.spinner("Generating embeddings..."):

                        contents = [
                            types.Content(
                                parts=[
                                    types.Part.from_text(text=chunk)
                                ]
                            )
                            for chunk in all_chunks
                        ]

                        result = client.models.embed_content(
                            model=EMBEDDING_MODEL,
                            contents=contents
                        )

                        embeddings = np.array(
                            [e.values for e in result.embeddings],
                            dtype="float32"
                        )

                    # ---------------------------------
                    # Create FAISS Index
                    # ---------------------------------

                    dimension = embeddings.shape[1]

                    index = faiss.IndexFlatL2(dimension)

                    index.add(embeddings)

                    # Save in session state
                    st.session_state.index = index
                    st.session_state.chunks = all_chunks
                    st.session_state.chunk_sources = chunk_sources

                    st.session_state.processed_files = [
                        f.name for f in uploaded_files
                    ]

                    st.success("All research papers processed successfully!")

                    st.info(
                        f"Total papers: {len(st.session_state.processed_files)}"
                    )

                    st.info(
                        f"Total chunks: {len(all_chunks)}"
                    )

                    st.info(
                        f"Total vectors stored: {index.ntotal}"
                    )

                except Exception as e:

                    st.error(f"Embedding or FAISS error: {e}")

            else:
                st.error("No valid text chunks were generated.")


# ---------------------------------
# Display Processed Documents
# ---------------------------------

if st.session_state.processed_files:

    st.header("2. Uploaded Research Papers")

    for filename in st.session_state.processed_files:
        st.write(f"📄 {filename}")


# ---------------------------------
# Question Answering
# ---------------------------------

if st.session_state.index is not None:

    st.header("3. Ask Questions")

    question = st.text_input(
        "Enter your question about the research papers:"
    )

    if st.button("Get Answer"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            try:

                with st.spinner("Searching papers and generating answer..."):

                    # ---------------------------------
                    # Generate Question Embedding
                    # ---------------------------------

                    query_result = client.models.embed_content(
                        model=EMBEDDING_MODEL,
                        contents=question
                    )

                    query_embedding = np.array(
                        [query_result.embeddings[0].values],
                        dtype="float32"
                    )

                    # ---------------------------------
                    # FAISS Similarity Search
                    # ---------------------------------

                    k = min(3, st.session_state.index.ntotal)

                    distances, indices = st.session_state.index.search(
                        query_embedding,
                        k
                    )

                    retrieved_chunks = []
                    retrieved_sources = []

                    for idx in indices[0]:

                        if idx == -1:
                            continue

                        retrieved_chunks.append(
                            st.session_state.chunks[idx]
                        )

                        retrieved_sources.append(
                            st.session_state.chunk_sources[idx]
                        )

                    # ---------------------------------
                    # Prepare Context
                    # ---------------------------------

                    context = "\n\n".join(
                        retrieved_chunks
                    )

                    prompt = f"""
                    You are a Research Paper Assistant.

                    Answer the user's question using ONLY
                    the information provided in the context.

                    If the answer is not available in the
                    context, clearly state that the uploaded
                    research papers do not contain sufficient
                    information to answer the question.

                    Context:
                    {context}

                    Question:
                    {question}

                    Provide a clear and accurate answer.
                    """

                    # ---------------------------------
                    # Gemini Answer Generation
                    # ---------------------------------

                    response = client.models.generate_content(
                        model=GENERATION_MODEL,
                        contents=prompt
                    )

                # ---------------------------------
                # Display Answer
                # ---------------------------------

                st.subheader("Answer")

                st.write(response.text)

                # ---------------------------------
                # Display Sources
                # ---------------------------------

                st.subheader("Source Research Papers")

                unique_sources = list(
                    dict.fromkeys(retrieved_sources)
                )

                for source in unique_sources:
                    st.write(f"📄 {source}")

            except Exception as e:

                st.error(f"Error generating answer: {e}")


# ---------------------------------
# Footer
# ---------------------------------

st.divider()

st.caption(
    "GenAI Research Paper Assistant | "
    "Powered by Gemini, FAISS and Streamlit"
)
