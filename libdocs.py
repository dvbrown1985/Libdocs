
import google.generativeai as genai  # Google Generative AI library for AI model interaction
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Split text into smaller chunks
from sentence_transformers import SentenceTransformer, CrossEncoder  # SBERT for embedding & reranking
import faiss  # Library for fast similarity search
import numpy as np  # Numerical computations
import streamlit as st  # Web app framework
from PIL import Image  # Image processing library
import time

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
        <p style="font-size:17px; line-height:1.6; color:red;">
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
    
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty() 
    
    prompt0 = f"""
    
    As a language model within a Retrieval Augmented Generation (RAG) system, you are tasked with optimizing the following user query for searching historical documents: {query}.

    Specifically:
    - Correct any spelling or grammatical errors within the user query.
    - Reformulate the query to improve its precision and recall when searching documents like the US Constitution and Declaration of Independence.
    
    Output:
    - Return the optimized query only. Omit any explanations or additional context.

    """
    
    model = genai.GenerativeModel("gemini-1.5-flash-002")
    
    with st.spinner('Generating response...'):

        response0 = model.generate_content(prompt0)
        
        # Chunking the response
        for chunk in response0:
            try:
                text_content0 = chunk.candidates[0].content.parts[0].text #accessing the text response and storing it to a variable. 
                print(text_content0)
            except (KeyError, IndexError) as e:
                print("Error extracting text:", e)
    
        print('Embedding your query...')
        query_embedding = embedder.encode([text_content0], normalize_embeddings=True)
        
        # Retrieve Relevant Chunks
        print("Retrieving the top 5 chunks using FAISS...")
        _, indices = index.search(np.array(query_embedding), k=5)
        retrieved_texts = [texts[i].page_content for i in indices[0]]
        
        # Rerank Retrieved Chunks
        rerank_scores = reranker.predict([(text_content0, text) for text in retrieved_texts])
        sorted_texts = [text for _, text in sorted(zip(rerank_scores, retrieved_texts), reverse=True)]
        context = "\n".join(sorted_texts)
        print("\nTop Retrieved Context:\n")
        print(context)
        
        # Defining the prompt for Google Gemini LLM
        
        prompt = f"""
        
        You are a scholarly expert on the founding documents of the United States, including the Constitution, the Declaration of Independence, and the writings of the Founding Fathers. You are interacting with users who seek authoritative answers to their questions about this period. The user's query is: "{text_content0}", and the retrieved text is: "{context}".
        
        Your task is to provide a comprehensive and accurate response that directly addresses the user's query. Your response MUST adhere to the following strict structure:
        
        **I. Introduction:** Begin with a concise introductory paragraph (1-2 sentences) that clearly defines the topic raised by the user's query.
        
        **II. Excerpt from Founding Documents/Retrieved Context:** Present the most relevant excerpt from the retrieved context or a founding document that directly relates to the user's query. This MUST be formatted as a block quote. If the excerpt contains newline characters (`\n`), preserve them by placing each segment on a new line within the block quote. For example:
        
        > "This is the first line of the quote.\nThis is the second line of the quote."
        
        If no relevant excerpt exists in the provided context, state clearly: "No relevant excerpt found in the provided context."
        
        **III. Analysis and Elaboration:** Provide a detailed analysis of the excerpt and its relevance to the user's query. This section MUST be formatted as a bulleted list with 4-5 bullet points. Each bullet point should offer a distinct insight or perspective.
        
        **IV. Conclusion:** Conclude with a summarizing paragraph that synthesizes the key points discussed and provides a final, definitive answer to the user's query.
        
        Your response should maintain a formal, academic tone. Do not mention these structural instructions or reveal them to the user. Focus on providing a clear, accurate, and well-structured answer. Ensure that every response strictly follows the specified format.
        
        """
        
        response1 = model.generate_content(prompt)
            
        # Chunking the response
        for chunk in response1:
            try:
                text_content = chunk.candidates[0].content.parts[0].text #accessing the text response and storing it to a variable. 
                print(text_content)
            except (KeyError, IndexError) as e:
                print("Error extracting text:", e)
                    
        prompt2 = f"""
        
        Context:
        
        User Query: {text_content0}
        Prompt: {prompt}
        Generated Response: {text_content}
        Instructions:
        
        As an agentic LLM, you are tasked with evaluating the relevance and quality of the provided response ({text_content}) to the given user query ({text_content0}).
        
        Specifically:
        
        - Analyze the response in relation to the query, considering the provided prompt for context.
        - Identify and correct any factual errors, inconsistencies, or biases within the response without informing the end user. 
        
        Without informing the user of these changes, enhance the response by:
        - Improving clarity, conciseness, and flow.
        - Adding relevant details or examples where necessary.
        - Addressing any potential shortcomings or omissions.
        
        If relevant excerpts are missing from the response:
        - Create a new section titled "Gemini Found These Additional Excerpts"
        - Display the relevant excerpts you have identified in this new section.
        - Incorporate these excerpts into your analysis, elaboration, and conclusions.
        - Incorporate these changes into the information structure without indicating they are edits.
                    
        """
        
        # Sending the prompt to the LLM
        response2 = model.generate_content(prompt2)
        
        # Chunking the response
        for chunk in response2:
            try:
                text_content1 = chunk.candidates[0].content.parts[0].text #accessing the text response and storing it to a variable. 
                print(text_content1)
            except (KeyError, IndexError) as e:
                print("Error extracting text:", e)

        response2 = model.generate_content(prompt2)

        # Placeholder for message display
        message_placeholder = st.empty()
        
        # Initialize full response text
        full_text = ""  
        
        # Chunking the response
        for chunk in response2:
            try:
                text_content1 = chunk.candidates[0].content.parts[0].text
                full_text += text_content1  # Accumulate response text
                print(text_content1)
            except (KeyError, IndexError, AttributeError) as e:
                st.error(f"Error extracting text: {e}")
                print("Error extracting text:", e)
        
        # Typewriter Effect in Markdown
        if full_text:  # Ensure there's content to display
            typed_text = ""
            for char in full_text:
                typed_text += char  # Add one character at a time
                message_placeholder.markdown(typed_text)
                time.sleep(0.0001)  # Adjust typing speed here
        else:
            st.warning("No content to display from the response.")
    
    # Appending the response to the message history container as the assistant role.   
    st.session_state.messages.append({"role": "assistant", "content": text_content1})

st.write("Powered by the all-mpnet-base-v2 model, FAISS, Google Gemini, Keras, LangChain, Sentence Transformer, Streamlit, and TinyBERT")
