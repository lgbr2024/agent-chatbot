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
import openai

# 환경 변수 로드
load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]

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

def openai_search(query: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Search the web and provide a summary for the following query: {query}"}
            ],
            max_tokens=100,
            temperature=0.5,
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        st.error(f"Error in OpenAI search: {str(e)}")
        return "Error occurred during OpenAI search."

def generate_response(question: str, pinecone_docs: List[Document], openai_result: str, llm: ChatOpenAI) -> str:
    pinecone_context = "\n".join([doc.page_content for doc in pinecone_docs])
    prompt = ChatPromptTemplate.from_template("""
    Based on the following information, answer the question. 
    Prioritize information from the conference documents when directly relevant, and use web search information to provide additional context or recent updates.

    Conference documents:
    {pinecone_context}

    Web search information:
    {openai_result}

    Question: {question}

    Provide a comprehensive answer that combines both sources of information:
    """)
    
    response = prompt.format(pinecone_context=pinecone_context, openai_result=openai_result, question=question)
    return llm.predict(response)

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
def animated_loading(placeholder, progress_bar):
    animation = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    messages = [
        "Searching conference documents...",
        "Performing web search...",
        "Analyzing information...",
        "Generating comprehensive response..."
    ]
    i = 0
    progress = 0
    while True:
        with placeholder.container():  # UI 업데이트를 메인 스레드에서 수행
            placeholder.info(f"{animation[i % len(animation)]} {messages[i % len(messages)]}")
        progress += 0.5
        if progress > 100:
            progress = 0
        progress_bar.progress(int(progress))
        time.sleep(0.1)
        i += 1

def main():
    st.title("Conference Q&A System with Web Search Integration")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 사용자 입력
    if question := st.chat_input("Please ask a question about the conference:"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            loading_placeholder = st.empty()
            progress_bar = st.progress(0)
            final_answer = st.empty()
            
            # 스레드 대신 메인 루프에서 애니메이션 업데이트
            loading_thread = threading.Thread(target=animated_loading, args=(loading_placeholder, progress_bar))
            loading_thread.start()
            
            try:
                # 여기에 Pinecone 문서 검색 및 OpenAI 검색 호출
                pass
            finally:
                loading_placeholder.empty()
                progress_bar.empty()
                loading_thread.join()
            
            final_answer.markdown("This is where the final answer would appear.")
        
        st.session_state.messages.append({"role": "assistant", "content": "This is where the final answer would appear."})

if __name__ == "__main__":
    main()
