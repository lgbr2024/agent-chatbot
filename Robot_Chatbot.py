import streamlit as st
import os
from dotenv import load_dotenv
from typing import List
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
import time
import threading
import requests

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY") or st.secrets.get("perplexity_api_key")

def process_pinecone_results(results: List[Document]) -> List[Document]:
    processed_docs = []
    for doc in results:
        if 'source' in doc.metadata:
            filename = os.path.basename(doc.metadata['source'])
            doc.page_content = filename
            processed_docs.append(doc)
    return processed_docs

def get_relevant_documents(retriever, question: str) -> List[Document]:
    try:
        docs = retriever.get_relevant_documents(question)
        return process_pinecone_results(docs)
    except Exception as e:
        st.error(f"Error retrieving documents from Pinecone: {str(e)}")
        return []

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

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        if 'choices' in data and len(data['choices']) > 0:
            return data['choices'][0]['message']['content']
        else:
            return "Unexpected response format from API."
    except requests.RequestException as e:
        print(f"Error in Perplexity search: {str(e)}")
        return "Error occurred during web search."

def generate_response(question: str, pinecone_docs: List[Document], perplexity_result: str, llm: ChatOpenAI) -> str:
    pinecone_context = "\n".join([doc.page_content for doc in pinecone_docs])
    prompt = ChatPromptTemplate.from_template("""
    Based on the following information, answer the question. 
    Prioritize information from the conference documents when directly relevant, and use web search information to provide additional context or recent updates.

    Conference documents:
    {pinecone_context}

    Web search information:
    {perplexity_result}

    Question: {question}

    Provide a comprehensive answer that combines both sources of information:
    """)
    
    response = prompt.format(pinecone_context=pinecone_context, perplexity_result=perplexity_result, question=question)
    return llm.predict(response)

def animated_loading():
    animation = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    messages = [
        "Searching conference documents...",
        "Performing web search...",
        "Analyzing information...",
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
    st.title("Conference Q&A System with Web Search Integration")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("conference")

    # Initialize OpenAI
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    # Set up Pinecone vector store
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
        text_key="source"
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
                pinecone_docs = get_relevant_documents(retriever, question)
                perplexity_result = perplexity_search(question)
                answer = generate_response(question, pinecone_docs, perplexity_result, llm)
            finally:
                loading_placeholder.empty()
                progress_bar.empty()
                loading_thread.join()
            
            final_answer.markdown(answer)
            
            with st.expander("Reference Documents"):
                st.write("Conference Documents:")
                for i, doc in enumerate(pinecone_docs[:5], 1):
                    st.write(f"{i}. Source: {doc.metadata.get('source', 'Unknown')}")
                st.write("\nWeb Search Result:")
                st.write(perplexity_result[:500] + "..." if len(perplexity_result) > 500 else perplexity_result)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
