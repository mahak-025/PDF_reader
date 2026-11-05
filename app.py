import os
import tempfile
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

st.set_page_config(page_title='PDF RAG with Groq', layout='wide')
st.title('PDF Reader + RAG (Groq)')

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def load_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    reader = PdfReader(tmp_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''
    os.unlink(tmp_path)
    return text

def build_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    embeddings = get_embeddings()
    return FAISS.from_documents(docs, embeddings)

def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

st.sidebar.header('Settings')
model_name = st.sidebar.selectbox('Groq Model', ['qwen/qwen3.8-27b'])

if not GROQ_API_KEY:
    st.sidebar.error('Set GROQ_API_KEY environment variable')

uploaded_file = st.file_uploader('Upload a PDF', type='pdf')

if uploaded_file and 'vectorstore' not in st.session_state and GROQ_API_KEY:
    with st.spinner('Processing PDF...'):
        text = load_pdf(uploaded_file)
        if not text.strip():
            st.error('Could not extract text from PDF')
        else:
            st.session_state.vectorstore = build_vectorstore(text)
            st.session_state.pdf_name = uploaded_file.name
            st.success(f'Processed {uploaded_file.name}')

if 'vectorstore' in st.session_state and GROQ_API_KEY:
    st.success(f'Ready to chat with {st.session_state.pdf_name}')
    query = st.text_input('Ask a question about the PDF')
    
    if query:
        with st.spinner('Generating answer...'):
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={'k': 4})
            llm = ChatGroq(model=model_name, api_key=GROQ_API_KEY, temperature=0)
            template = '''Answer the question based only on the following context:
{context}

Question: {question}
'''
            prompt = ChatPromptTemplate.from_template(template)
            chain = (
                {'context': retriever | format_docs, 'question': RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            answer = chain.invoke(query)
            st.markdown('### Answer')
            st.write(answer)
            
            with st.expander('Source chunks'):
                docs = retriever.invoke(query)
                for i, doc in enumerate(docs):
                    st.markdown(f'**Chunk {i+1}:**')
                    content = doc.page_content
                    if len(content) > 500:
                        content = content[:500] + '...'
                    st.text(content)
