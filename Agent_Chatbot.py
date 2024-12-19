import os
import streamlit as st
from typing import List, Dict, Any
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
- 한국어로 대화형 톤으로 2천자 내외로 작성하세요.
- 해당 내용 문단의 끝에는 출처, 발표자 내용을 꼭 표시해주세요.
"""
chatbot_prompt = ChatPromptTemplate.from_template(chatbot_template)

def main():
    st.title("📚 Agent Conference Q&A Chatbot")

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
        embedding=OpenAIEmbeddings(model="text-embedding-curie-001"),
        text_key="text"
    )

    # 검색 필터 설정 함수
    def create_retriever_with_filter(keywords: Dict[str, Any]) -> ModifiedPineconeVectorStore:
        filter_conditions = {}

        # 태그 필터 추가
        if keywords.get("tag_contents"):
            filter_conditions["tag_contents"] = {"$in": keywords["tag_contents"]}
        if keywords.get("tag_people"):
            filter_conditions["tag_people"] = {"$in": keywords["tag_people"]}
        if keywords.get("tag_company"):
            filter_conditions["tag_company"] = {"$in": keywords["tag_company"]}

        # 출처 필터 추가
        if keywords.get("source"):
            filter_conditions["source"] = {"$in": keywords["source"]}

        # 필터 조건을 적용한 retriever 생성
        return vectorstore.as_retriever(
            search_type='similarity',
            search_kwargs={
                "k": 10,
                "filter": filter_conditions
            }
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
    def create_response(question: str, keywords: Dict[str, Any]) -> str:
        # 입력 유효성 검사
        if not isinstance(question, str) or not question.strip():
            return "질문이 유효하지 않습니다. 질문을 다시 입력해주세요."

        # 검색된 문서 포맷팅
        retriever = create_retriever_with_filter(keywords)
        docs = retriever.invoke(question)
        if not docs:
            return "검색된 문서가 없습니다. 질문과 관련된 문서를 찾을 수 없습니다."

        # 문서 포맷 및 길이 제한 적용
        context = format_docs(docs)
        max_length = 2000  # OpenAI API 입력 길이 제한 (문자 기준)
        context = context[:max_length]

        # 프롬프트를 사용해 데이터 생성
        prompt_input = {"question": question, "context": context}
        try:
            prompt = chatbot_prompt.invoke(prompt_input)
            llm_response = llm.invoke(prompt)
            return StrOutputParser().invoke(llm_response)
        except Exception as e:
            st.error(f"OpenAI API 호출 중 오류가 발생했습니다: {e}")
            return "답변 생성 중 오류가 발생했습니다. 다시 시도해주세요."

    # 이전 대화 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    question = st.chat_input("컨퍼런스 관련 질문을 입력하세요:")
    if question:
        # 사용자로부터 필터 키워드 입력받기
        st.write("질문과 함께 필터 조건을 입력하세요:")
        tag_contents = st.text_input("태그 내용 (쉼표로 구분)", "")
        tag_people = st.text_input("관련 인물 (쉼표로 구분)", "")
        tag_company = st.text_input("관련 회사 (쉼표로 구분)", "")
        source = st.text_input("출처 (쉼표로 구분)", "")

        # 필터 키워드 생성
        keywords = {
            "tag_contents": [tag.strip() for tag in tag_contents.split(",") if tag.strip()],
            "tag_people": [person.strip() for person in tag_people.split(",") if person.strip()],
            "tag_company": [company.strip() for company in tag_company.split(",") if company.strip()],
            "source": [src.strip() for src in source.split(",") if src.strip()]
        }

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            # 검색 및 답변 생성
            st.write("검색 중...")
            response = create_response(question, keywords)
            st.markdown(response)

            # 채팅 기록 저장
            st.session_state.messages.append({"role": "assistant", "content": response})

    # 대화 초기화 버튼
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
