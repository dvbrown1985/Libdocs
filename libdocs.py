#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Import necessary libraries
import google.generativeai as genai  # Google Generative AI library for AI model interaction
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Split text into smaller chunks
from sentence_transformers import SentenceTransformer, CrossEncoder  # SBERT for embedding & reranking
import faiss  # Library for fast similarity search
import numpy as np  # Numerical computations
import streamlit as st  # Web app framework
from PIL import Image  # Image processing library

# --- 1. Streamlit App Configuration ---
st.set_page_config(
    page_title="LibDocs",
    page_icon="📜",
    layout="centered"
)

@st.cache_resource
def load_and_process_docs(files):
    documents = []
    for file in files:
        documents.extend(TextLoader(file).load())  # Flatten the loaded documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    texts = text_splitter.split_documents(documents)
    return texts

@st.cache_resource
def get_embeddings_and_index(_texts):
    embedder = SentenceTransformer('all-mpnet-base-v2')
    embeddings = embedder.encode([t.page_content for t in _texts], normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings))
    return embeddings, index

@st.cache_resource
def get_reranker():
    return CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')

texts = load_and_process_docs(["founding docs.txt"])
embeddings, index = get_embeddings_and_index(texts)
reranker = get_reranker()
embedder = SentenceTransformer('all-mpnet-base-v2')

# --- 6. Initialize Streamlit Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 7. Configure Google Generative AI ---
GOOGLE_API_KEY = 'AIzaSyAfxmrmwhu62afqajL84gI5hta6LDUs9yc'
genai.configure(api_key=GOOGLE_API_KEY)

# --- 8. Display Logo ---
LOGO = "libdocs.jpg"
image = Image.open(LOGO)
container1 = st.container(border=True)
with container1:
    st.image(image, clamp=False, channels="RGB", use_container_width=True)

st.logo(LOGO)

# --- 9. Disclaimer Section ---
expander = st.expander("Disclaimer")
expander.write('''
    This is a prototype under development and may contain bugs or errors. 
    It is intended for testing and educational purposes only. 
    Please use this prototype with caution and at your own risk.
''')

# --- 10. Introduction Section ---
container_x = st.container(border=True)
with container_x:
    st.markdown("""
    <div> 
        <p style="font-size:17px; line-height:1.6; color:#48acd2;">
            LibDocs (short for Liberty Documents) is designed to converse with you about the Declaration of Independence and the US Constitution.
        </p>
        <p style="font-size:17px; line-height:1.6; color:#48acd2;">
            Use the input bar below 👇 to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- 11. Chat History ---
container2 = st.container(border=True)
if st.session_state.messages:
    for message in st.session_state.messages:
        with container2.chat_message(message["role"]):
            st.markdown(message["content"])
else:
    st.write("No messages in history yet.")

# --- 12. Query Handling ---
if query := st.chat_input(" 🗽 🇺🇸 🦅 "):
    print('Embedding your query...')
    query_embedding = embedder.encode([query], normalize_embeddings=True)

    # Retrieve Relevant Chunks
    print("Retrieving the top 5 chunks using FAISS...")
    _, indices = index.search(np.array(query_embedding), k=5)
    retrieved_texts = [texts[i].page_content for i in indices[0]]

    # Rerank Retrieved Chunks
    rerank_scores = reranker.predict([(query, text) for text in retrieved_texts])
    sorted_texts = [text for _, text in sorted(zip(rerank_scores, retrieved_texts), reverse=True)]
    context = "\n".join(sorted_texts)
    print("\nTop Retrieved Context:\n")
    print(context)
    
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
    # Define prompt for AI model
    prompt = f"""
    
    You are a scholarly expert on the founding documents of the United States, including the Constitution, the Declaration of Independence, and the writings of the Founding Fathers. You are interacting with users who seek authoritative answers to their questions about this period. The user query is: "{query}", and the retrieved text is: "{context}".

    Your task is to provide a comprehensive and accurate response that directly addresses the user's query. Your response MUST adhere to the following strict structure:

    **I. Introduction:** Begin with a concise introductory paragraph (1-2 sentences) that clearly defines the topic raised by the user's query.

    **II. Excerpt from Founding Documents/Retrieved Context:** Present the relevant excerpts from the retrieved context that directly relates to the user's query. If the retrieved context contains multiple relevant exceerpts, be sure to include each excerpt. This MUST be formatted as a block quote. If the excerpt contains newline characters (`\n`), preserve them by placing each segment on a new line within the block quote. For example:

    > "This is the first line of quote #1.\nThis is the second line of quote #1."

    > "This is the first line of quote #2.\nThis is the second line of quote #2."

    If no relevant excerpts exists in the provided context, state clearly: "No relevant excerpt found in the provided documents. Please adjust your query and try again."

    **III. Analysis and Elaboration:** Provide a detailed analysis of the excerpt and its relevance to the user's query. This section MUST be formatted as a bulleted list with 4-5 bullet points. Each bullet point should offer a distinct insight or perspective.

    **IV. Conclusion:** Conclude with a summarizing paragraph that synthesizes the key points discussed and provides a final, definitive answer to the user's query.

    Your response should maintain a formal, academic tone. Do not mention these structural instructions or reveal them to the user. Focus on providing a clear, accurate, and well-structured answer. Ensure that every response strictly follows the specified format.
    
    """
    
    with st.spinner("Generating response..."):
        model = genai.GenerativeModel("gemini-1.5-flash-002")
        response = model.generate_content(prompt)
        
        for chunk in response:
            try:
                text_content = chunk.candidates[0].content.parts[0].text
                print(text_content)
            except (KeyError, IndexError) as e:
                print("Error extracting text:", e)
        
        message_placeholder.markdown(text_content)
    
    st.session_state.messages.append({"role": "assistant", "content": text_content})

