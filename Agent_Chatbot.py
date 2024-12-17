import os
import re
import glob
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

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

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 8, filter: Dict[str, Any] = None, namespace: str = None
    ) -> List[Tuple[Document, float]]:
        namespace = namespace or self._namespace
        results = self.index.query(
            vector=embedding,
            top_k=k,
            include_metadata=True,
            filter=filter,
            namespace=namespace,
        )
        return [
            (
                Document(
                    page_content=result["metadata"].get(self._text_key, ""),
                    metadata={k: v for k, v in result["metadata"].items() if k != self._text_key}
                ),
                result["score"],
            )
            for result in results["matches"]
        ]

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

    # Initialize session state
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

    format = itemgetter("docs") | RunnableLambda(format_docs)
    chain = RunnableParallel(question=RunnablePassthrough(), docs=retriever) | chatbot_prompt | llm | StrOutputParser()

    # 이전 메시지 출력
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
            response = chain.invoke({"question": question})
            st.markdown(response)

            # 채팅 기록 저장
            st.session_state.messages.append({"role": "assistant", "content": response})

    # 대화 초기화 버튼
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
