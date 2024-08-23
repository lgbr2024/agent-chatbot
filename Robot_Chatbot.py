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
import logging

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]
PERPLEXITY_API_KEY = st.secrets["perplexity_api_key"]

def perplexity_search(query: str) -> str:
    try:
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
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()['choices'][0]['message']['content']
    except requests.RequestException as e:
        logger.error(f"Error in Perplexity search: {str(e)}")
        return "Error occurred during web search."

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

def get_relevant_documents(retriever, question: str) -> List[Document]:
    try:
        logger.debug(f"Searching for documents related to: {question}")
        docs = retriever.get_relevant_documents(question)
        logger.debug(f"Found {len(docs)} relevant documents")
        for i, doc in enumerate(docs):
            logger.debug(f"Document {i+1}: {doc.page_content[:100]}...")  # 문서 내용의 처음 100자만 로그로 출력
        return docs
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return []

def generate_response(question: str, pinecone_docs: List[Document], llm: ChatOpenAI) -> str:
    try:
        # Pinecone-based response generation
        if not pinecone_docs:
            logger.warning("No relevant documents found in the database.")
            pinecone_response = "No relevant information found in the database."
        else:
            pinecone_context = "\n".join([doc.page_content for doc in pinecone_docs])
            logger.debug(f"Pinecone context: {pinecone_context[:500]}...")  # 컨텍스트의 처음 500자만 로그로 출력
            pinecone_prompt = f"""
            Based on the following context from the Pinecone database, answer the question. 
            If the context doesn't provide enough information, indicate that additional web search might be needed.

            Context: {pinecone_context}

            Question: {question}

            Answer:
            """
            pinecone_response = llm.predict(pinecone_prompt)
            logger.debug(f"Pinecone response: {pinecone_response[:500]}...")  # 응답의 처음 500자만 로그로 출력

        # Check if Pinecone response indicates need for additional information
        if "additional web search" in pinecone_response.lower() or "not enough information" in pinecone_response.lower():
            logger.info("Pinecone response insufficient, performing Perplexity search")
            perplexity_result = perplexity_search(question)
            
            if "Error occurred during web search" in perplexity_result:
                logger.error("Error occurred during Perplexity search")
                return f"I apologize, but I couldn't find enough information to answer your question. The database search was insufficient, and there was an error with the web search. It would be best to consult other reliable sources for this information."
            
            # Integrate Pinecone and Perplexity information
            final_response = integrate_information(pinecone_response, perplexity_result, question, llm)
        else:
            final_response = pinecone_response

        logger.info("Response generation completed successfully")
        return final_response
    except Exception as e:
        logger.error(f"An error occurred while generating the response: {str(e)}")
        return "I'm sorry, but an error occurred while processing your question. Please try again later or rephrase your question."

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
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = "conference"
        index = pc.Index(index_name)
        logger.info(f"Successfully connected to Pinecone index: {index_name}")
    except Exception as e:
        logger.error(f"Error connecting to Pinecone: {str(e)}")
        st.error("Error connecting to the database. Please check your configuration.")
        return

    # Initialize OpenAI
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    # Set up Pinecone vector store
    try:
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
            text_key="text"
        )
        logger.info("Successfully set up Pinecone vector store")
    except Exception as e:
        logger.error(f"Error setting up Pinecone vector store: {str(e)}")
        st.error("Error setting up the vector store. Please check your configuration.")
        return
    
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
                docs = get_relevant_documents(retriever, question)
                
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
