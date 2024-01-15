from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
import boto3
import os
import streamlit as st

#run me using - streamlit run main.py
#bedrock client

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

#modelID = put any model here
modelID = "meta.llama2-13b-chat-v1"


llm = Bedrock(
    model_id=modelID,
    client=bedrock_client,
    model_kwargs={"max_gen_len": 512,"temperature":0.9, "top_p": 1}
)


def my_chatbot(language,freeform_text):
    prompt = PromptTemplate(
        input_variables=["language", "freeform_text"],
        template="You are a chatbot. You are in {language}.\n\n{freeform_text}"
    )

    bedrock_chain = LLMChain(llm=llm, prompt=prompt)

    response=bedrock_chain.invoke({'language':language, 'freeform_text':freeform_text})
    return response
#print(my_chatbot("english","what is AWS?"))

st.title("Bedrock Chatbot")

language = st.sidebar.selectbox("Language", ["english", "spanish"])

if language:
    freeform_text = st.sidebar.text_area(label="what is your question?",
    max_chars=100)

if freeform_text:
    response = my_chatbot(language,freeform_text)
    st.write(response['text'])

