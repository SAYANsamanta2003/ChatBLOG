import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

# Function To get response from LLAma 2 model
# cTransformer is provied python buldings for GGML model
# it using on huggingface 

def getLLamaresponse(input_text,no_words,blog_style):

    # llama2-model here we used 'max_new_tokens' cause how many words you get in your prompts same as 'temperature'. If you CPU efficiency is low then you used 'max_new_tokens'=256 and 'temperature'=0.01.
    llm=CTransformers(model='model/llama-2-7b-chat.ggmlv3.q8_0.bin',model_type='llama',config={'max_new_tokens':800,'temperature':0.7})
    
    # Prompt-Template
    template=""" Write a blog for {blog_style} job profile for a topic {input_text}within {no_words} words."""
    
    prompt=PromptTemplate(input_variables=["blog_style","input_text",'no_words'],template=template)
    
    # ressponse from the LLama 2 using GGML model
    response=llm(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    print(response)
    return response

#for style the page
st.set_page_config(page_title="ChatBLOG",page_icon='🚀',layout='centered',initial_sidebar_state='collapsed')

st.markdown("""
    <style>
        /* Set dark blue gradient background */
        body {
            background: linear-gradient(135deg, #0d47a1, #1a237e);
            color: white;
        }
        /* Style header */
        h1, h2, h3, h4, h5, h6 {
            color: #bbdefb;
        }
        /* Input fields */
        input, select, textarea {
            background-color: #1565c0;
            color: white;
            border: 1px solid #bbdefb;
            border-radius: 8px;
            padding: 10px;
        }
        /* Button styling */
        div.stButton button {
            background: linear-gradient(90deg, #0d47a1, #1e88e5);
            color: white;
            border: 1px solid white;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            padding: 8px 20px;
            cursor: pointer;
        }
        div.stButton button:hover {
            background: linear-gradient(90deg, #1e88e5, #0d47a1);
        }
        /* Columns styling */
        .stContainer {
            padding: 10px;
            border-radius: 10px;
            background: rgba(13, 71, 161, 0.5);
        }
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: #0d47a1;
            color: white;
        }
        section[data-testid="stSidebar"] h1 {
            color: #bbdefb;
        }
    </style>
    """,unsafe_allow_html=True
)

st.header("ChatBLOG ✍️")

input_text=st.text_input("Enter Topic",placeholder="Message ChatBLOG")

# creating two column for 2 filds

col1,col2=st.columns([5,5])

with col1:
    no_words=st.text_input('No of Words',placeholder="eg.- 100,200")
with col2:
    blog_style=st.selectbox('Writing the blog for',('Researchers','AIML','Web Development','Student','Common People'),index=1)
    
submit=st.button("Click⬆️")

## Final response
if submit:
    st.write(getLLamaresponse(input_text,no_words,blog_style))