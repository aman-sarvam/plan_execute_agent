import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from state_graph import execute_graph
import os

st.set_page_config(page_title='Marketing Campaign Helper', layout='wide')

# Initialize or retrieve session state for the checklist
if 'generated_checklist' not in st.session_state:
    st.session_state.generated_checklist = None

# Function to generate the checklist using LangChain
def generate_checklist(task_description, api_key):
    model = ChatOpenAI(api_key=api_key, temperature=0.5, model= "gpt-4-0613")
    prompt_template = """
    For the following legal research task, outline detailed plans that can be methodically followed. Each plan should \
    specify any external tool along with the necessary input to process marketing-related tasks. Results from each step \
    can be stored in variables like #E1, #E2, etc., which can be referenced in subsequent steps.

    Tools that might be used include:
    (1) Google Search[input]: Uses Google to search for law information, regulations, etc. Input should \
    be a search query related to the plan.
    (2) LLM[input]: A pretrained LLM like yourself. Useful when you need general legal knowledge, legal analysis, or interpretation. \
    Prioritize it when you need to draft legal arguments or summarize complex legal information.

    Begin your task description with rich details. Each Plan should be followed by only one #E.

    For example,
    Task: Research the legal framework surrounding data privacy laws in India.
    Plan: Search for current government regulations and guidelines about data privacy in India using Google Search. #E1 = Google Search[data privacy laws India]
    Plan: Analyze the search results to identify key legal provisions and government guidelines. #E2 = LLM[Analyze key provisions, given #E1]
    Plan: Summarize how these regulations compare with international data privacy standards. #E3 = LLM[Compare with international standards, given #E2]

    Every task may not require all the tools. 

    Task: {task}
    """


    prompt = prompt_template.format(task=task_description)
    result = model.invoke(prompt)
    st.session_state.generated_checklist = result.content
    return result.content

def edit_checklist(current_checklist, edit_description, api_key):
    model = ChatOpenAI(api_key=api_key, temperature=0.5)
    edit_prompt_template = """
    Here is the current checklist:
    {current_checklist}

    The required edits are as follows:
    {edit_description}

    Please update the checklist according to the description above and maintain the exact format of plans and tools. Only return the checklist and no other text. Skip any introductory phrases 
    """

    prompt = edit_prompt_template.format(current_checklist=current_checklist, edit_description=edit_description)
    result = model.invoke(prompt)
    st.session_state.generated_checklist = result.content
    return result.content

def save_checklist(checklist):
    with open("checklist.txt", "w") as f:
        f.write(checklist)
    st.download_button(
        label="Download Checklist",
        data=checklist,
        file_name="legal_research_plan.txt",
        mime="text/plain"
    )
    
def main_page():
    st.title('Legal Research Helper')

    # Sidebar for API configuration
    st.sidebar.header('Configuration')
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

    user_input = st.text_area("Describe the marketing campaign task or how to edit the checklist:", height=150)
    action_button = st.button('Submit')

    if action_button and api_key:
        if st.session_state.get('generated_checklist'):
            edited_result = edit_checklist(st.session_state.generated_checklist, user_input, api_key)
            st.subheader("Updated Checklist:")
            st.text(edited_result)
        else:
            if user_input:
                generated_result = generate_checklist(user_input, api_key)
                st.subheader("Generated Checklist:")
                st.text(generated_result)
            else:
                st.warning("Please enter a task description.")

    if 'generated_checklist' in st.session_state and st.session_state.generated_checklist:
        save_checklist(st.session_state.generated_checklist)

def execute_page():
    st.title('Execute Legal Research task')
    st.write("Upload your checklist and click execute to see the plans carried out.")

    # st.sidebar.title("Configuration")
    # api_key = st.sidebar.text_input("Enter your API key", type="password")
    # if api_key:
    #     os.environ['OPENAI_API_KEY'] = api_key

    uploaded_file = st.file_uploader("Upload your checklist", type=['txt'])
    if uploaded_file is not None:
        checklist_content = uploaded_file.getvalue().decode("utf-8")
        print("Checklist_content=", checklist_content)
        task_title = st.text_area("Mention the task title:", height=150)
        if st.button('Execute Plans'):
            # Calling the execute_graph function from lang_graph.py
            results = execute_graph(checklist_content, task_title)
            st.write("Execution Results:")
            st.write(results['result']) 

page = st.sidebar.selectbox("Choose a page", ["Main", "Execute"])

if page == "Main":
    main_page()
elif page == "Execute":
    execute_page()
    
    
#Possible conversion of plan to sequence diagram 