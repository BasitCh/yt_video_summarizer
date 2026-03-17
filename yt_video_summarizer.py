import os
import re
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

# Support Streamlit Cloud secrets
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


def extract_video_id(url):
    match = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
    return match.group(1) if match else None


def load_transcript(url):
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    proxy_url = st.secrets.get("PROXY_URL", None)
    if proxy_url:
        from youtube_transcript_api._proxies import GenericProxyConfig
        ytt_api = YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(
                http_url=proxy_url,
                https_url=proxy_url,
            )
        )
    else:
        ytt_api = YouTubeTranscriptApi()

    transcript = ytt_api.fetch(video_id)
    text = " ".join([entry["text"] for entry in transcript])
    return [Document(page_content=text, metadata={"source": url})]


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
      docs = load_transcript(url)
    except (TranscriptsDisabled, NoTranscriptFound):
      st.error("No transcript found for this video.")
      st.stop()
    except Exception as e:
      st.error(f"Failed to fetch transcript: {e}")
      st.stop()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(text_chunks, embeddings)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    prompt = PromptTemplate.from_template("""
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

    chain = parallel_chain | prompt | llm | parser

    answer = chain.invoke(question)
    st.success("Answer:")
    st.write(answer)
