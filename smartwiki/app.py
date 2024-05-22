import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper 
import matplotlib.pyplot as plt

# Set page title and configure layout
st.set_page_config(page_title="Smart Wiki", layout="wide")

# Sidebar for Google API Key input and info
st.sidebar.subheader("Google AI Key")
google_api_key = st.sidebar.text_input('Enter your Google AI Key', type="password")
st.sidebar.info("A Google API Key is required to access certain services from Google, such as the Google Knowledge Graph API. You can obtain a Google API Key by following the instructions [here](https://aistudio.google.com/app/apikey).")

# Main content area
st.markdown("<h1 style='text-align: center; color: white; padding: 10px; background-color: #002147;'>SmartWiki</h1>", unsafe_allow_html=True)
st.markdown("---")  # Horizontal rule

if google_api_key:
    # Input prompt
    prompt = st.text_input('Enter your prompt here') 

    if google_api_key and prompt:
        # Prompt templates
        wiki_template = PromptTemplate(
            input_variables=['topic'], 
            template='write me all latest and maximum wikipedia pages titles about this:{topic}, give infomation estimation in numbers based on each titles, one line brief overview.must give me pages link, tell me how much information on wikipedia pages about this in numbers(response format should be:: **Hennessey Venom F5** - estimated top speed of 301 mph, currently holds the title of the worlds fastest production car.linebreak https://en.wikipedia.org/wiki/Hennessey_Venom_F5 linebreak, Total contributors: 102)'
        )

        knowledge_graph_template = PromptTemplate(
            input_variables=['topic'], 
            template='get information from textual knowledge graph of {topic} from wikipedia, give fact and figures as much as you can in numbers'
        )

        # Memory 
        title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
        knowledge_graph_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')

        # Initialize ChatGoogleGenerativeAI with the provided API key
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7,top_p=0.85, google_api_key=google_api_key)
        
        # Create LLM chains
        title_chain = LLMChain(llm=llm, prompt=wiki_template, verbose=True, output_key='title', memory=title_memory)
        knowledge_graph_chain = LLMChain(llm=llm, prompt=knowledge_graph_template, verbose=True, output_key='knowledge_graph', memory=knowledge_graph_memory)

        wiki = WikipediaAPIWrapper()

        # Get information from the LLM
        info = title_chain.run(prompt)

        # Display Wikipedia search results within an expander
        with st.expander("Wikipedia Search Results", expanded=False):
            st.write(wiki.run(prompt))

        # Extract knowledge graph
        knowledge_graph_info = knowledge_graph_chain.run(prompt)

        # Split the info string into individual topic-response pairs
        topic_responses = info.split("\n\n")

        # Initialize lists to store topics and contributors
        topics = []
        contributors = []

        # Extract information for each topic-response pair
        for topic_response in topic_responses:
            lines = topic_response.split('\n')
            topic = lines[0].split('**')[1].strip()  # Extract topic from the bold text
            total_contributors_line = lines[-1].split(':')[-1].strip()  # Extract total_contributors line
            try:
                total_contributors = int(total_contributors_line)
            except ValueError:
                print("Error converting to int:", total_contributors_line)  # Print problematic line
                continue
            topics.append(topic)
            contributors.append(total_contributors)

        # Set up columns for layout
        col1, col2 = st.columns([2, 3])

        # Place Wikipedia title and knowledge graph side by side
        with col1:
            st.write(info)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.bar(topics, contributors, color='blue')
            ax.set_xlabel('Topics')
            ax.set_ylabel('Contributors')
            ax.set_title('Contributors per Topic')
            ax.set_xticks(range(len(topics)))  # Set ticks explicitly
            ax.set_xticklabels(topics, rotation=45, ha='right')
            
            # Show plot using Streamlit
            st.pyplot(fig)

        with col2:
            st.header("Wikipedia Knowledge Graph")
            st.write(knowledge_graph_info)
            st.markdown('</div>', unsafe_allow_html=True)
