import streamlit as st
import pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.callbacks import get_openai_callback
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import openai
import time

openai.api_key = st.secrets["OPENAI_API_KEY"]

# Set the Pinecone API key
pinecone.api_key = st.secrets["PINECONE_API_KEY"]
pinecone.environment = st.secrets["PINECONE_ENVIRONMENT"]

embeddings_model = OpenAIEmbeddings()

index_name = 'stocks3-1715968536'

text_field = "text"

# Initialize the Pinecone index
index = pinecone.Index(index_name)

# Initialize the Pinecone vector store
vectorstore = Pinecone(index, embed_model.embed_query, text_field)

# Define OPENAI_API_KEY
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Initialize the LLM
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

def main():
    st.header("Chat to a Document 💬👨🏻‍💻🤖")

    # Ask a question about the documents in the index
    query = st.text_input("Ask question's about your document:")

    suggestions = ["", "What is the main topic of the document?", "Summarize the document in 200 words?", "Provide a bullet point list of the key points mentioned in the document?", "Create the headings and subheadings for Powerpoint slides", "Translate the first paragraph to French"]

    suggestion = st.selectbox("Or select a suggestion: (ENSURE QUESTION FIELD ABOVE IS BLANK)", suggestions, index=0)

    if query or suggestion:
        if suggestion:
            query = suggestion

        # Perform the similarity search
        results = vectorstore.similarity_search(query, k=10)  # Return 10 most relevant docs

        # Use the LLM to generate a response based on the retrieved documents
        for result in results:
            response = llm.generate(result)
            parse_response(response)

if __name__ == '__main__':
    main()
