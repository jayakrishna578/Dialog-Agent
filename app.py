import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import helper_functions as hf

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize components
chat_model = ChatOpenAI()
llm = OpenAI(openai_api_key=openai_api_key)
memory = ConversationBufferMemory(memory_key='chat_history', k=5)

prompt_template = PromptTemplate(
    input_variables=['chat_history', 'question'],
    template="""You are a kind agent, you help humans with real-time questions 
    and you answer their questions with patience and politeness.
    chat history: {chat_history}
    Human: {question}
    AI:""")

llmchain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt_template
)

conversation_buffer = hf.ConversationSummaryBuffer()

# Streamlit App configurations
st.set_page_config(
    page_title='Dialog Agent UI',
    page_icon='..',
    layout='wide'
)

st.title('Dialog Agent UI')

# Initialize token count and amount spent
total_tokens = 0
amount_spent = 0

# Sidebar
st.sidebar.title('🤖💬 Dialog Agent UI')
st.sidebar.write(f"Total Tokens Used: {total_tokens}")
st.sidebar.write(f"Cost in USD: ${amount_spent}")

if st.sidebar.button('View Logs'):
    logs = hf.get_all_logs()
    for log in logs:
        st.write(f"👤: {log[1]}")
        st.write(f"🤖: {log[2]}")

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

new_prompt = st.chat_input("What's on your mind?")

# Checking if the new prompt is not a repeat of the last user message
if new_prompt and (not st.session_state.messages or st.session_state.messages[-1]["content"] != new_prompt):
    st.session_state.messages.append({"role": "user", "content": new_prompt})
    
    with get_openai_callback() as cb:
        ai_response = llmchain.predict(question=new_prompt)
        total_tokens = cb.total_tokens
        amount_spent = cb.total_cost
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

    # Update the sidebar with the new token count and amount spent
    st.sidebar.write(f"Total Tokens Used: {total_tokens}")
    st.sidebar.write(f"Cost in USD: ${amount_spent}")
