import os
import re
import time
import streamlit as st
from typing import List, Tuple, Dict, Any
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore

# Pinecone 및 API 키 설정
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]

# Pinecone 커스텀 벡터 스토어
class ModifiedPineconeVectorStore(PineconeVectorStore):
    def __init__(self, index, embedding, text_key: str = "text", namespace: str = ""):
        super().__init__(index, embedding, text_key, namespace)
        self.index = index
        self._embedding = embedding
        self._text_key = text_key
        self._namespace = namespace

# Chatbot 프롬프트
chatbot_template = """
Question: {question}
Context: {context}
Answer:
- 질문에 대한 답변을 메타데이터의 정보를 최대한 활용하여 작성하세요.
- 주어진 문서의 출처(source), 핵심 내용(text), 그리고 관련된 태그(tag_contents, tag_people, tag_company)를 활용해서 구체적이고 정확한 답변을 제공하세요.
- 한국어로 대화형 톤으로 작성하세요.
"""
chatbot_prompt = ChatPromptTemplate.from_template(chatbot_template)

def main():
    st.title("📚 Conference Q&A Chatbot")

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Pinecone 초기화
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "aiconference"
    index = pc.Index(index_name)

    # OpenAI 모델 설정
    llm = ChatOpenAI(model="gpt-4o")
    vectorstore = ModifiedPineconeVectorStore(
        index=index,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
        text_key="text"
    )

    # 검색 설정
    retriever = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={"k": 10}
    )

    # 문서 포맷 함수
    def format_docs(docs: List[Document]) -> str:
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown source')
            text = doc.page_content
            tags = ', '.join(
                f"{k}: {v}" for k, v in doc.metadata.items() if k.startswith("tag_")
            )
            formatted.append(f"출처: {source}\n태그: {tags}\n내용: {text}")
        return "\n\n".join(formatted)

    # 문서와 프롬프트를 결합하여 답변 생성
    def create_response(question: str) -> str:
        # 검색된 문서 포맷팅
        docs = retriever.invoke(question)
        context = format_docs(docs)

        # 프롬프트를 사용해 데이터 생성
        prompt_input = {"question": question, "context": context}
        prompt = chatbot_prompt.invoke(prompt_input)

        # LLM 호출 및 결과 반환
        llm_response = llm.invoke(prompt)
        return StrOutputParser().invoke(llm_response)

    # 이전 대화 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    if question := st.chat_input("컨퍼런스 관련 질문을 입력하세요:"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            # 검색 및 답변 생성
            st.write("검색 중...")
            response = create_response(question)
            st.markdown(response)

            # 채팅 기록 저장
            st.session_state.messages.append({"role": "assistant", "content": response})

    # 대화 초기화 버튼
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
