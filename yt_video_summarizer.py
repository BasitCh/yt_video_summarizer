import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

st.set_page_config(page_title="Youtube QA Bot", layout="centered")
st.title("Youtube QA Bot")
st.write("Ask a question about the video and get an answer")

#user input fields
url = st.text_input("Enter the URL of the YouTube video")
question = st.text_input("Enter your question about the video")

if st.button("Get Answer"):
  if not url or not question:
    st.error("Please enter both a URL and a question")
    st.stop()

  with st.spinner("Fetching video transcript..."):
    docs = []
    try:
      loader = YoutubeLoader.from_youtube_url(
        youtube_url= url
      )
      docs = loader.load()
    except (TranscriptsDisabled, NoTranscriptFound) as e:
      print(f"No Video Transcription Found: {e}")
      st.error("No Video Transcription Found")
      exit()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(text_chunks, embeddings)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    PromptTemplate = PromptTemplate.from_template("""
    You are a helpful assistant. Answer the question based on the context provided. If the context is insufficient, just say you don't know.
    Context: {context}
    Question: {question}
    """)

    def format_docs(retrieved_docs):
      text_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
      return text_content
    
    parser = StrOutputParser()
    parallel_chain = RunnableParallel({
      'context': retriever | RunnableLambda(format_docs),
      'question': RunnablePassthrough()
    })

    chain = parallel_chain | PromptTemplate | llm | parser

    answer = chain.invoke(question)
    st.success("Answer:")
    st.write(answer)