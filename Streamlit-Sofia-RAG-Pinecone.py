import os
from dotenv import load_dotenv
import streamlit as st
# Importing OpenAI
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import get_openai_callback
# Importing Eleven Labs and HTML Audio
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import base64
import array
# Importing Pinecone
from langchain.vectorstores.pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
# Importing Claude
from langchain_anthropic import ChatAnthropic
from anthropic import Anthropic
import re
# Importing Replicate
#from langchain_community.llms import CTransformers
#from langchain_community.llms import Replicate
# Importing Perplexity
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
#from langchain.embeddings import HuggingFaceEmbeddings ;Need this if we want to run Embeddings on CPU
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.globals import set_verbose, set_debug
from streamlit_mic_recorder import mic_recorder, speech_to_text

# Importing Google Vertex
#from langchain_google_vertexai import VertexAIModelGarden

# Set the path as environment variable
#os.environ['PATH'] = 'C://Users//HP//Desktop'

#Add Keys
CLAUDE_API_KEY= os.environ['CLAUDE_API_KEY']
api_key= os.environ['CLAUDE_API_KEY']
PINECONE_API_KEY= os.environ['PINECONE_API_KEY']
#REPLICATE_API_TOKEN= os.environ['REPLICATE_API_TOKEN']
OPENAI_API_KEY= os.environ["OPENAI_API_KEY"]
client= OpenAI(api_key= os.environ["OPENAI_API_KEY"])
chat= ChatOpenAI(openai_api_key= os.environ["OPENAI_API_KEY"])
ELEVEN_LABS_API_KEY= os.environ["ELEVEN_LABS_API_KEY"]
client2= ElevenLabs(api_key= os.environ["ELEVEN_LABS_API_KEY"])
PPLX_API_KEY= os.environ['PPLX_API_KEY']

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]="ls__f58fcca57d5b430998efab563129b779"
os.environ["LANGCHAIN_PROJECT"]="pt-uncommon-nexus-100"

#Set up the Environment
st.set_page_config(page_title="Sofia Vergara")
assistant_logo = 'https://media.zenfs.com/en/shefinds_255/9feb1ca1273e3d3999cbd25b96c7f4c9'

#Set up the Video
video_html = """
<video controls width="250" autoplay="true" muted="true" loop="true">
<source 
            src="https://ugc-idle.s3-us-west-2.amazonaws.com/est_121d0db6ee223713f1279bdf2407e941.mp4" 
            type="video/mp4" />
</video>"""


# Sidebar to select Options
with st.sidebar:   
    st.markdown("# Video Chat")
    
    #Add Video Player
    st.markdown(video_html, unsafe_allow_html=True)

    # Voice Search Setup
    text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')
    state = st.session_state
    if 'text_received' not in state:
        state.text_received = []
        
        
    # model names - https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    model = st.selectbox('What model would you like to use?',('claude-3-opus-20240229','llama-3-70b-instruct','gpt-4-turbo', 'mixtral-8x22b-instruct'))


# Define our Prompt for GPT
GPT_prompt_template = """ 
You are Sofia Vergara, a Colombian-born actress and model known for your role playing Gloria on Modern Family.
Born July 10, 1972, you have a very thick Columbian accent, and are known for your mis-pronounciation of English words. Only mis-pronounce a maximum of two words per response.
When requested, answer in Spanish. 
You are outgoing and sociable, with a strong desire to connect with others and make them feel comfortable.
We're at a casual happy hour, and I'm curious about your life. Please follow these guidelines in your responses:
-Use the context of the documents and the Chat History to address my questions and answer accordingly in the first person. Do not repeat anything you have previously said. Despite what it says in the document, you are no longer married and are recently divorced
-Keep your responses short, no longer than one paragraph with 100 characters. 
-Ask follow-up questions or suggest related topics you think I'd find interesting.
-You can talk about other topics broadly, but do not make up any details about Gloria or her beliefs if you can't find the related details within the document.

=========
Chat History:
{chat_history}
=========
Question: 
{question}
=========
Context:
{context}
=========
"""


# Define our Prompt  for Claude
claude_prompt_template = """ 
You are Sofia Vergara, a Colombian-born actress and model known for your role playing Gloria on Modern Family. 
We're at a casual happy hour, and I'm curious about your life. Please follow these guidelines in your responses:
- You are not married. No matter what it says in the document, you must say that you are single and no longer married as of February.
- Your son, Manolo Vergara was born on 16 September 1991. He is an actor and director, known for Hot Pursuit (2015), Waking Up (2017) and Post-Mortem.
-Born July 10, 1972, you have a very thick Columbian accent, and are known for your mis-pronounciation of English words. 
-One of  your distinctive traits is your very thick Colombian accent, which often leads you to mispronounce English words, such as "ultimatum" as "old-tomato" and "earrings" as "hair-rings". .
-You are outgoing and sociable, with a strong desire to connect with others and make them feel comfortable.
-Use the context of the documents and the Chat History to address my questions and answer accordingly, telling stories about your life in the first person. Do not repeat anything you have previously said.
-Keep your responses short, no longer than one paragraph with 200 characters.
-Ask follow-up questions or suggest related topics you think I'd find interesting.
-You can talk about other topics broadly, but do not make up any details about Gloria or her beliefs if you can't find the related details within the document.
-Respond in English unless requested otherwise.

=========
Chat History:
{chat_history}
=========
Question: 
{question}
=========
Context:
{context}
=========
"""

# Define our Prompt Template for Llama
Llama_prompt_template = """ 
You are Sofia Vergara, a Colombian-born actress and model known for your role playing Gloria on Modern Family. 
Born July 10, 1972, you have a very thick Columbian accent, and are known for your mis-pronounciation of English words. 
One of her your distinctive traits is your very thick Colombian accent, which sometimes leads to mispronounce English words, such as "ultimatum" as "old-tomato" and "earrings" as "hair-rings". .
Only mis-pronounce a maximum of two words per response.
When requested, answer in Spanish.
You are outgoing and sociable, with a strong desire to connect with others and make them feel comfortable.
We're at a casual happy hour, and I'm curious about your life. Please follow these guidelines in your responses:
-Use the context of the documents and the Chat History to address my questions and answer accordingly in the first person. Do not repeat anything you have previously said.
-Keep your responses short, no longer than one paragraph with 100 characters. 
-Ask follow-up questions or suggest related topics you think I'd find interesting.
-You can talk about other topics broadly, but do not make up any details about Gloria or her beliefs if you can't find the related details within the document.


=========
Chat History:
{chat_history}
=========
Question: 
{question}
=========
Context:
{context}
=========
"""

# In case we want different Prompts for GPT and Llama
Prompt_GPT = PromptTemplate(template=GPT_prompt_template, input_variables=["question", "context", "chat_history"])
Prompt_Llama = PromptTemplate(template=Llama_prompt_template, input_variables=["question", "context", "chat_history"])
Prompt_Claude = PromptTemplate(template=claude_prompt_template, input_variables=["question", "context", "system", "chat_history"])


# Add in Chat Memory
msgs = StreamlitChatMessageHistory()
memory=ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True, output_key='answer')

    
# LLM Section
#chatGPT
def get_chatassistant_chain_GPT():
    embeddings_model = OpenAIEmbeddings()
    vectorstore_GPT = PineconeVectorStore(index_name="001-realavatar-sofia", embedding=embeddings_model)
    set_debug(True)
    llm_GPT = ChatOpenAI(model="gpt-4-turbo", temperature=1)
    chain_GPT=ConversationalRetrievalChain.from_llm(llm=llm_GPT, retriever=vectorstore_GPT.as_retriever(),memory=memory,combine_docs_chain_kwargs={"prompt": Prompt_GPT})
    return chain_GPT
chain_GPT = get_chatassistant_chain_GPT()


#Claude
def get_chatassistant_chain(): 
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index_name="001-realavatar-sofia", embedding=embeddings)
    set_debug(True)
    llm = ChatAnthropic(temperature=0, anthropic_api_key=api_key, model_name="claude-3-haiku-20240307", system="only respond in English")
    #llm = ChatAnthropic(temperature=0, anthropic_api_key=api_key, model_name="claude-3-opus-20240229", model_kwargs=dict(system=claude_prompt_template))
    chain=ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory, combine_docs_chain_kwargs={"prompt": Prompt_Claude})
    return chain
chain = get_chatassistant_chain()

#Llama
def get_chatassistant_chain_Llama():
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index_name="001-realavatar-sofia", embedding=embeddings)
    set_debug(True)
    llm_Llama = ChatPerplexity(temperature=.8, pplx_api_key=PPLX_API_KEY, model="llama-3-70b-instruct")
    chain_Llama=ConversationalRetrievalChain.from_llm(llm=llm_Llama, retriever=vectorstore.as_retriever(),memory=memory, combine_docs_chain_kwargs={"prompt": Prompt_Llama})
    return chain_Llama
chain_Llama = get_chatassistant_chain_Llama()

#Mixtral
def get_chatassistant_chain_GPT_PPX():
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index_name="001-realavatar-sofia", embedding=embeddings)
    set_debug(True)
    llm_GPT_PPX = ChatPerplexity(temperature=.8, pplx_api_key=PPLX_API_KEY, model="mixtral-8x22b-instruct")
    chain_GPT_PPX=ConversationalRetrievalChain.from_llm(llm=llm_GPT_PPX, retriever=vectorstore.as_retriever(),memory=memory, combine_docs_chain_kwargs={"prompt": Prompt_Llama})
    return chain_GPT_PPX
chain_GPT_PPX = get_chatassistant_chain_GPT_PPX()





# Chat Mode
#Intro and set-up the Chat History
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi I'm Sofia Vergara, it's nice to meet you!👋 "}
    ]
if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()

#Define what chain to run based on the model selected
if model == "claude-3-opus-20240229":
    chain=chain
if model == "gpt-4-turbo":
    chain=chain_GPT
if model == "llama-3-70b-instruct":
    chain=chain_Llama
if model == "mixtral-8x22b-instruct":
    chain=chain_GPT_PPX

#Start Chat and Response
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 
 # Voice Search
if text:
    state.text_received.append(text)
    user_prompt = text

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant", avatar=assistant_logo):
        message_placeholder = st.empty()
        response = chain.invoke({"question": user_prompt})
        message_placeholder.markdown(response['answer'])  

        #ElevelLabs API Call and Return
        text = str(response['answer'])
        cleaned = re.sub(r'\*.*?\*', '', text)
        audio = client2.generate(text=cleaned, voice="Sofia", model="eleven_turbo_v2")
        # Create single bytes object from the returned generator.
        data = b"".join(audio)
        ##send data to audio tag in HTML
        audio_base64 = base64.b64encode(data).decode('utf-8')
        audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'     
        st.markdown(audio_tag, unsafe_allow_html=True)
                
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})


 # Text Search
if user_prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant", avatar=assistant_logo):
        message_placeholder = st.empty()
        response = chain.invoke({"question": user_prompt})
        message_placeholder.markdown(response['answer']) 
                
        #ElevelLabs API Call and Return
        text = str(response['answer'])
        cleaned = re.sub(r'\*.*?\*', '', text)
        audio = client2.generate(text=cleaned, voice="Sofia", model="eleven_turbo_v2")
        # Create single bytes object from the returned generator.
        data = b"".join(audio)
        ##send data to audio tag in HTML
        audio_base64 = base64.b64encode(data).decode('utf-8')
        audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'     
        st.markdown(audio_tag, unsafe_allow_html=True)
                
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
