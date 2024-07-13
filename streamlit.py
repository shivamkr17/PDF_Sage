import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import Pinecone
import os
import google.generativeai as genai
from langchain.vectorstores import Pinecone as PC
from dotenv import load_dotenv

load_dotenv()

gemini_key=os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=gemini_key)


def pinecon():
    from pinecone import Pinecone, ServerlessSpec

    pinecone_key=os.getenv("PINECONE_API_KEY")
    pc=Pinecone(api_key=pinecone_key)

    index_name="project"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    )
    return index_name


def getpdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        Pdf_reader = PdfReader(pdf)
        for page in Pdf_reader.pages:
             text+=page.extract_text()

    return text


def gettext_chunks(text):
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
    text_chunks = text_splitter.split_text(text)

    return text_chunks


def getvector_store(text_chunks):
    index_name = pinecon()
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    docsearch = PC.from_texts([t for t in text_chunks], embedding, index_name=index_name)
    return docsearch


def showman(pdf_docs):
    st.header("Pdf_Sage")

    user_question = st.text_input("Ask a question based on the uploaded PDFs",key="user_question")
    Submit=st.button("Submit")
    ask_another_question=st.button("Ask Another Question",on_click=clear_text)

    if user_question and Submit:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",temperature=0.7)
        from langchain.chains import RetrievalQA
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=st.session_state["docsearch"].as_retriever())
        response = qa(user_question)
        st.session_state["response"]=response["result"]
        st.write("Answer:",st.session_state["response"])


def clear_text():
    st.session_state["user_question"] = ""
    st.session_state["response"] = ""

    
def show():
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    st.session_state["pdf_docs"] = pdf_docs if pdf_docs is not None else st.session_state.get("pdf_docs", [])
    processed = st.session_state.get("processed", False)

    if not processed and pdf_docs:
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = getpdf_text(pdf_docs)
                text_chunks = gettext_chunks(raw_text)
                docsearch = getvector_store(text_chunks)
                st.session_state["docsearch"] = docsearch
                st.session_state["processed"] = True
            st.success("Done!")
    
    showman(st.session_state["pdf_docs"])


show()


