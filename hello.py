from langchain.document_loaders import DirectoryLoader

directory = 'C:\\Users\\LomoM\\hello-world\\Doc'

def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

# Splitting documents into chunks
documents = load_docs(directory)
len(documents)

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)
print(len(docs))

from langchain.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

import pinecone  # pip install pinecone-client
from langchain.vectorstores import Pinecone
pinecone.init(
    api_key="2cbc6bf9-cd55-4250-a38b-3b16cd14114d",  # find at app.pinecone.io
    environment="us-west4-gcp-free"  # next to api key in console
)
index_name = "langchain-chatbot"
index = Pinecone.from_texts(docs, embeddings, index_name=index_name)  # Use 'docs' directly

# The rest of the code remains unchanged
# ...



def get_similar_docs(query, k=1, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs

# Building the Chatbot Application with Streamlit
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *

st.subheader("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template("""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")

human_msg_template = HumanMessagePromptTemplate.from_template("{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

# Creating the User Interface
st.title("Langchain Chatbot")
response_container = st.container()
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')

# Initialize the Language Model and Conversation
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="844cfa67b9fc4cb4b298683292c12862")
context = ""  # Define a default context (modify as per your requirement)

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# Generating Responses
if query:
    with st.spinner("typing..."):
        response = conversation.predict(input=f"Context:\n{context}\n\nQuery:\n{query}")
    st.session_state.requests.append(query)
    st.session_state.responses.append(response)

# Refining Queries and Finding Matches with Utility Functions
# ... (Assuming these functions are complete and correct as they are)

# Tracking the Conversation
# ... (Assuming this function is complete and correct as it is)

# ... (Remaining code)


#Refining Queries and Finding Matches with Utility Functions
def query_refiner(conversation, query):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

#Finding Matches in Pinecone Index
def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

#Tracking the Conversation
def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string
