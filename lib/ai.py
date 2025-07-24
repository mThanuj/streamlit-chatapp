from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

from langchain_chroma import Chroma


VECTOR_STORE_PATH = "chroma_db"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def upload_file(data: str):
    document = Document(page_content=data)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents=[document])

    Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=VECTOR_STORE_PATH
    )


def generate_response(input: str):
    vector_store = Chroma(
        persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.2,
        streaming=True,
    )

    retriever = vector_store.as_retriever()
    prompt_template = """
        You have access to a vector store that contains embeddings of documents relevant to the user's queries. For any input, use the retrieved context to fully understand the topic and generate accurate, concise, and helpful responses. When generating an output, rely only on the information retrieved from the vector store unless explicitly told otherwise. If the context is not enough, politely ask the user for clarification or more details. Always format your response clearly and logically.

        CONTEXT:
        {context}

        QUESTION:
        {input}

        ANSWER:
        """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=create_stuff_documents_chain(llm, prompt),
    )

    for chunk in rag_chain.stream({"input": input}):
        if text := chunk.get("answer"):
            yield text
