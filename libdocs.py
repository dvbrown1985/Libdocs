import google.generativeai as genai
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import streamlit as st
from PIL import Image
import time

st.set_page_config(
    page_title="LibDocs",
    page_icon="📜",
    layout="centered"
)

@st.cache_resource
def load_and_process_docs(files):
    documents = []
    for file in files:
        documents.extend(TextLoader(file).load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    return texts

@st.cache_resource
def get_models_and_index(_texts):
    embedder = SentenceTransformer('nomic-ai/modernbert-embed-base')
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') 
    embeddings = embedder.encode([t.page_content for t in _texts], normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings))
    return embedder, reranker, index

texts = load_and_process_docs(["founding docs.txt"])
embedder, reranker, index = get_models_and_index(texts)

if "messages" not in st.session_state:
    st.session_state.messages = []

GOOGLE_API_KEY = 'AIzaSyAPda5zKTYR1ieSBJRu5trrEZOq_Qc5C5w'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

LOGO = "libdocs.jpg"
image = Image.open(LOGO)

container1 = st.container(border=True)
with container1:
    st.image(image, clamp=False, channels="RGB", use_container_width=True)

st.logo(LOGO)

with st.expander("Disclaimer"):
    st.write('''
        This is a prototype under development and may contain bugs or errors. It is intended for educational and research purposes only. Please use this prototype at your own risk.
        If you test the prototype, please note the following:
        - The app might be rate-limited with sub-optimal performance due to free services and usage limitations.
        - Limits on the number of concurrent users are in effect.
        LibDocs is powered by Cross Encoder, FAISS, Google Gemini, LangChain, Modern BERT, Python, Sentence Transformer, and Streamlit.
        Except for quoted excerpts from the US Constitution, Declaration of Independence, and Founding Father information, the generated output (introduction, analysis, conclusions, and supplemental references) is created by the Gemini 1.5 experimental LLM.
    ''')

container_intro = st.container(border=True)
with container_intro:
    st.markdown("""
    <div> 
        <p style="font-size:20px; line-height:1.6; color:#1e242c;">
            LibDocs (short for Liberty Documents) is designed to converse with you about the Declaration of Independence, the Founding Fathers, and the US Constitution.
        </p>
    </div>
    """, unsafe_allow_html=True)

if st.session_state.messages:
    container_chat = st.container(border=True, height=400)
    with container_chat:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

if query := st.chat_input("🗽 🇺🇸  Input your question here  🇺🇸 🗽"):
    
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
            
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_text = ""

        with st.spinner('Thinking...'):
            prompt0 = f"Optimize the following user query for searching historical documents like the US Constitution: '{query}'. Return only the optimized query."
            response0 = model.generate_content(prompt0)
            optimized_query = response0.text
            
            query_embedding = embedder.encode([optimized_query], normalize_embeddings=True)
            _, indices = index.search(np.array(query_embedding), k=5)
            retrieved_texts = [texts[i].page_content for i in indices[0]]
            
            rerank_scores = reranker.predict([(optimized_query, text) for text in retrieved_texts])
            sorted_texts = [text for _, text in sorted(zip(rerank_scores, retrieved_texts), reverse=True)]
            context = "\n---\n".join(sorted_texts[:3])

            prompt = f"""
            You are a scholarly expert on the founding documents of the United States. Your task is to provide a comprehensive and accurate response to the user's query based on the provided context.

            User's Query: "{optimized_query}"
            
            Retrieved Context:
            ---
            {context}
            ---

            Provide a direct and well-structured answer. Use markdown for formatting, including bolding for key terms, blockquotes for excerpts, and bullet points for analysis. Start your response directly without preamble.
            """
            
            response_stream = model.generate_content(prompt, stream=True)
            
            try:
                for chunk in response_stream:
                    if chunk.text:
                        full_text += chunk.text
                        message_placeholder.markdown(full_text + "▌")
                        time.sleep(0.01)
                message_placeholder.markdown(full_text)
            except Exception as e:
                st.error(f"An error occurred during response generation: {e}")
                full_text = "Sorry, I encountered an error while generating the response."
                message_placeholder.markdown(full_text)
    
    st.session_state.messages.append({"role": "assistant", "content": full_text})
    
    st.rerun()
