import streamlit as st
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# --- PAGE CONFIG ---
st.set_page_config(page_title="RAG PDF Chatbot", page_icon="💬", layout="wide")

# --- CACHE EMBEDDINGS ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- PROCESS PDF INTO VECTORSTORE ---
def process_pdf(pdf_file):
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.getvalue())
        tmp_path = tmp.name
    
    # Load and split
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(splits, get_embeddings())
    
    # Remove temp file
    return vectorstore

# --- APP LAYOUT ---
st.title("📄 PDF RAG Chatbot")
st.write("Upload a PDF and ask questions about its content.")

# Upload PDF
pdf_file = st.file_uploader("Upload PDF", type="pdf")

if pdf_file:
    if "vectorstore" not in st.session_state:
        with st.spinner("Processing PDF..."):
            st.session_state.vectorstore = process_pdf(pdf_file)
            st.success("PDF processed and indexed!")

# --- CHAT SESSION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input("Ask a question about the PDF..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if "vectorstore" in st.session_state:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=None,  # No external LLM required; placeholder
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever()
                )
                # Run QA
                raw_response = qa_chain.run(user_input)
            else:
                raw_response = "Please upload a PDF to chat with it."
            
            st.markdown(raw_response)
            st.session_state.messages.append({"role": "assistant", "content": raw_response})