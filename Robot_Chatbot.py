import openai

def openai_search(query: str) -> str:
    try:
        response = openai.Engine("davinci").search(
            documents=[],
            query=query
        )
        return response['data'][0]['text']
    except Exception as e:
        print(f"Error in OpenAI search: {str(e)}")
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
                openai_result = openai_search(question)
                answer = generate_response(question, pinecone_docs, openai_result, llm)
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
                st.write(openai_result[:500] + "..." if len(openai_result) > 500 else openai_result)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
