import os
import streamlit as st
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

st.set_page_config(page_title="Resume Q&A Assistant")
st.title("📄 AI Resume Q&A Assistant")

uploaded_file = st.file_uploader("Upload your PDF Resume", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    try:
        # --- Step 1: Load PDF ---
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()

        # --- Step 2: Split into chunks ---
        splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=15)
        chunks = splitter.split_documents(documents)

        # --- Step 3: Generate embeddings ---
        st.info("Generating embeddings...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embedding_model)

        # --- Step 4: Setup local text generation model ---
        generator = pipeline("text-generation", model="google/flan-t5-base", max_new_tokens=150, do_sample=True)
        llm = HuggingFacePipeline(pipeline=generator)

        # --- Step 5: Create Retrieval QA Chain with improved prompt ---
        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an assistant answering questions about a resume.
Use the context below to answer the user's question as accurately as possible.
If the information is not present, respond with "Not mentioned."

Context:
{context}

Question: {question}
Answer:
"""
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff",
            chain_type_kwargs={"prompt": custom_prompt},
            return_source_documents=True
        )

        st.success("Resume loaded! You can now ask questions.")
        query = st.text_input("Ask a question about the resume:")
        if query:
            result = qa_chain.invoke(query)
            st.write("**Answer:**", result["result"])
            with st.expander("Context Chunks Used"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Chunk {i+1}:**\n{doc.page_content}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        