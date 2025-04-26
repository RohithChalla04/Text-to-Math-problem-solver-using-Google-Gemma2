import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler


## Streamlit 
st.set_page_config(page_title="Langchain: Text to Math problem solver and Data search Assisstant", page_icon="🦜")
st.title("Text to Math problem solver using Google Gemma2")

groq_api_key=st.sidebar.text_input(label="Groq_api_key",type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()

llm=ChatGroq(model="gemma2-9b-it",groq_api_key=groq_api_key)


## Initialize the Wikipedia Tool
wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool used for searching internet to find the various informations on the topics mentioned"
)


## Initialize the Math Problem Tool
math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="calculator",
    func=math_chain.run,
    description="A tool for answering math related questions. Only input mathematical expressions needs to be provided"
)


prompt=""" 
Your an AI agent tasked for solving users mathematical questions. Logically arrive at the solution and give proper explanation for it and display it 
in point wise for the question below
Question:{question}
Answer:
"""


prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)


## Combine all the tools into chain
chain=LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool=Tool(
    name="Reasoning_tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)


## Initialize the tools
assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_error=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I am a Math chatbot who can answer all your math problems"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

## Start the interaction
question=st.text_area("Enter your question","If an exterior angle of a triangle is 140 degrees and its opposite interior angles are equal to each other, which of the following is the measure of the equal angles of the triangle?")

if st.button("Find my answer"):
    if question:
        with st.spinner("Generating response...."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb])
            st.session_state.messages.append({"role":"assistant","content":response})
            st.write("### Response:")
            st.success(response)

    else:
        st.warning("Please enter the question")
