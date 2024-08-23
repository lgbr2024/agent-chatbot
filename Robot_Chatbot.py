import streamlit as st
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
import requests
import time
import threading

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]
PERPLEXITY_API_KEY = st.secrets["perplexity_api_key"]

def perplexity_search(query: str) -> str:
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "mixtral-8x7b-instruct",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides information based on web search results."
            },
            {
                "role": "user",
                "content": query
            }
        ]
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {PERPLEXITY_API_KEY}"
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error in Perplexity search: {response.status_code}"

def extract_key_points(text: str, llm: ChatOpenAI) -> List[str]:
    prompt = f"""
    Extract the key points from the following text. Provide each key point as a separate item in a list.

    Text: {text}

    Key points:
    """
    response = llm.predict(prompt)
    return [point.strip() for point in response.split('\n') if point.strip()]

def integrate_information(pinecone_info: str, perplexity_info: str, question: str, llm: ChatOpenAI) -> str:
    pinecone_points = extract_key_points(pinecone_info, llm)
    perplexity_points = extract_key_points(perplexity_info, llm)

    integration_prompt = f"""
    You are tasked with integrating information from two sources to answer a question. 
    Consider the reliability, relevance, specificity, and recency of each piece of information.

    Question: {question}

    Information from Pinecone database:
    {' '.join(f'- {point}' for point in pinecone_points)}

    Information from Perplexity web search:
    {' '.join(f'- {point}' for point in perplexity_points)}

    Please provide a comprehensive answer that:
    1. Prioritizes information from the Pinecone database when directly relevant to the conference.
    2. Uses Perplexity web search information to provide recent context, updates, or supplementary information.
    3. Highlights any discrepancies or additional insights between the two sources.
    4. Clearly indicates the source of information (Pinecone database or Perplexity web search).
    5. Provides a balanced and coherent response that directly answers the question, emphasizing the most up-to-date and relevant information.

    Integrated answer:
    """

    return llm.predict(integration_prompt)

def generate_response(question: str, pinecone_docs: List[Document], llm: ChatOpenAI) -> str:
    # Pinecone-based response generation
    pinecone_context = "\n".join([doc.page_content for doc in pinecone_docs])
    pinecone_prompt = f"""
    Based on the following context from the Pinecone database, answer the question. 
    If the context doesn't provide enough information or might benefit from recent updates, indicate that additional web search might be needed.

    Context: {pinecone_context}

    Question: {question}

    Answer:
    """
    pinecone_response = llm.predict(pinecone_prompt)

    # Check if Pinecone response indicates need for additional information
    if "additional web search" in pinecone_response.lower() or "recent updates" in pinecone_response.lower():
        perplexity_result = perplexity_search(question)
        
        # Integrate Pinecone and Perplexity information
        final_response = integrate_information(pinecone_response, perplexity_result, question, llm)
    else:
        final_response = pinecone_response

    return final_response

def animated_loading():
    animation = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    messages = [
        "Searching Pinecone database...",
        "Analyzing conference information...",
        "Checking for recent updates...",
        "Integrating information...",
        "Generating comprehensive response..."
    ]
    i = 0
    while True:
        for frame in animation:
            yield f"{frame} {messages[i % len(messages)]}"
            time.sleep(0.1)
        i += 1

def update_loading_animation(placeholder, progress_bar):
    loading_animation = animated_loading()
    progress = 0
    while not placeholder.empty():
        placeholder.info(next(loading_animation))
        progress += 0.5
        if progress > 100:
            progress = 0
        progress_bar.progress(int(progress))
        time.sleep(0.1)

def main():
    st.title("Conference Q&A System with Pinecone and Perplexity Integration")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "conference"
    index = pc.Index(index_name)
    
    # Initialize OpenAI
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    # Set up Pinecone vector store
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
        text_key="text"
    )
    
    # Set up retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if question := st.chat_input("Please ask a question about the conference:"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            loading_placeholder = st.empty()
            progress_bar = st.progress(0)
            final_answer = st.empty()
            
            loading_thread = threading.Thread(target=update_loading_animation, args=(loading_placeholder, progress_bar))
            loading_thread.start()
            
            try:
                # Retrieve documents using Pinecone
                docs = retriever.get_relevant_documents(question)
                
                # Generate response
                answer = generate_response(question, docs, llm)
            finally:
                loading_placeholder.empty()
                progress_bar.empty()
                loading_thread.join()
            
            final_answer.markdown(answer)
            
            with st.expander("Reference Documents"):
                for i, doc in enumerate(docs[:5], 1):
                    st.write(f"{i}. Source: {doc.metadata.get('source', 'Unknown')}")
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
