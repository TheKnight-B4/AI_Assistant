import os
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import tempfile


OPENAI_API_TYPE="azure"
OPENAI_API_VERSION="2023-05-15"
OPENAI_API_BASE="https://cog-openai-omg-lab-fc.openai.azure.com/"

from dotenv import load_dotenv
#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_TYPE"] = OPENAI_API_TYPE
os.environ["OPENAI_API_VERSION"] = OPENAI_API_VERSION
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE

load_dotenv()
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your OpenAI Key here",
    type="password")


uploaded_file = st.sidebar.file_uploader("upload", type="csv")

if uploaded_file :
   #use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                'delimiter': ','})
    data = loader.load()
st.write(data)

#Cutting the CSV file and provide it to vectorstore (FAISS)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(data, embeddings)

#Adding the ConversationalRetrievalChain by providing the chat model and the vectorstore
chain = ConversationalRetrievalChain.from_llm(
llm = AzureChatOpenAI(
    deployment_name='gpt-35-turbo-omg-lab',
    model_name='gpt-35-turbo',
    temperature=0.1
    ),
retriever=vectorstore.as_retriever())

#This function allows us to provide the user’s question and conversation history to ConversationalRetrievalChain to generate the chatbot’s response.
def conversational_chat(query):
        
    result = chain({"question": query, 
    "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
        
    return result["answer"]

#Initialize the chatbot session by creating st.session_state[‘history’] and the first messages displayed in the chat.

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]
        
#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()

#set up the UI part that allows the user to enter and send their question to our conversational_chat function with the user’s question as an argument.
with container:
    with st.form(key='my_form', clear_on_submit=True):
            
        user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
        submit_button = st.form_submit_button(label='Send')
            
    if submit_button and user_input:
        output = conversational_chat(user_input)
            
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

#display the chat between the user and the chatbot
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

