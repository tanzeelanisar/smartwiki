import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper 
import google.generativeai as genai
from gtts import gTTS
import time
import glob
from googletrans import Translator
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re


# Set page title and configure layout
st.set_page_config(page_title="Smart Tutor", layout="wide")

# Create necessary directory
os.makedirs("temp", exist_ok=True)

# Initialize translator
translator = Translator()
wiki = WikipediaAPIWrapper()

# Define sensitive patterns and keywords
sensitive_patterns = {
    'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'ssn': r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',
    'credit_card': r'\b(?:\d[ -]*?){13,16}\b',
    'password': r'password[:=]\s*\S+',
    'bank_account': r'\b\d{9,12}\b',
    'address': r'\d{1,5}\s\w+\s\w+'
}

sensitive_keywords = [
    'password', 'ssn', 'social security number', 'credit card', 'bank account', 'phone number', 'email', 'address'
]

# Function to apply differential privacy using the Laplace mechanism
def apply_differential_privacy(text, epsilon=1.0):
    sensitivity = 1.0  # Sensitivity of the function (you can adjust this based on your use case)
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, len(text))
    
    # Apply noise to each character's ASCII value and create a new noisy text
    noisy_text = ''.join(chr(min(max(32, ord(char) + int(noise_val)), 126)) for char, noise_val in zip(text, noise))
    
    return noisy_text

# Function to detect sensitive information and apply privacy
def detect_and_apply_privacy(prompt):
    # Detect patterns
    for pattern_name, pattern in sensitive_patterns.items():
        matches = re.finditer(pattern, prompt, re.IGNORECASE)
        for match in matches:
            sensitive_text = match.group()
            prompt = prompt.replace(sensitive_text, apply_differential_privacy(sensitive_text))
    
    # Detect keywords
    for keyword in sensitive_keywords:
        keyword_pattern = r'\b' + re.escape(keyword) + r'\b'
        matches = re.finditer(keyword_pattern, prompt, re.IGNORECASE)
        for match in matches:
            sensitive_text = match.group()
            prompt = prompt.replace(sensitive_text, apply_differential_privacy(sensitive_text))
    
    return prompt


# Function to remove old audio files
def remove_files(n_days):
    mp3_files = glob.glob("temp/*.mp3")
    now = time.time()
    cutoff = now - (n_days * 86400)
    for f in mp3_files:
        if os.stat(f).st_mtime < cutoff:
            os.remove(f)

# Function to convert text to speech
def text_to_speech(text, tld):
    tts = gTTS(text, lang='en', tld=tld, slow=False)
    my_file_name = "audio"
    tts.save(f"temp/{my_file_name}.mp3")
    return my_file_name

# Function to translate text
def text_translate(input_language, translate_language, text):
    translation = translator.translate(text, src=input_language, dest=translate_language)
    return translation.text

# Function to get related topics using AI
def generate_related_topics(prompt, n_topics=5):
    related_prompt_template = PromptTemplate(
        input_variables=['topic'], 
        template=f'Generate a list of {n_topics} related topics or articles based on this topic: {{topic}}'
    )

    related_topic_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6, top_p=0.85, google_api_key=st.session_state.google_api_key)
    related_topic_chain = LLMChain(llm=llm, prompt=related_prompt_template, verbose=True, output_key='related_topics', memory=related_topic_memory)
    related_topics_response = related_topic_chain.run({'topic': prompt})
    related_topics = related_topics_response.split('\n')
    return related_topics

# Function to retrieve and process Wikipedia information
def retrieve_and_process_wikipedia_info(prompt):
    wiki_template = PromptTemplate(
        input_variables=['topic'], 
        template='write me only all latest and maximum wikipedia pages titles about this:{topic}, give infomation estimation in numbers based on each titles, one line brief overview.must give me pages link, tell me how much information on wikipedia pages about this in numbers(response should follow this example format, dont add this example response in orignall response:: **Hennessey Venom F5** - estimated top speed of 301 mph, currently holds the title of the worlds fastest production car.\n https://en.wikipedia.org/wiki/Hennessey_Venom_F5 \n, Total contributors: 102)'
    )

    knowledge_graph_template = PromptTemplate(
        input_variables=['topic'], 
        template='get information from textual knowledge graph of {topic} from wikipedia, give fact and figures as much as you can in numbers as per example format(example output results and  Facts and Figures :Facts and Figures:Top speed: 1,227.986 km/h (763.035 mph). \n)'
    )

    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    knowledge_graph_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6, top_p=0.85, google_api_key=st.session_state.google_api_key)
    title_chain = LLMChain(llm=llm, prompt=wiki_template, verbose=True, output_key='title', memory=title_memory)
    knowledge_graph_chain = LLMChain(llm=llm, prompt=knowledge_graph_template, verbose=True, output_key='knowledge_graph', memory=knowledge_graph_memory)

    info = title_chain.run(prompt)
    knowledge_graph_info = knowledge_graph_chain.run(prompt)
    return info, knowledge_graph_info

# Function to configure sidebar
def configure_sidebar():
    out_lang = st.sidebar.selectbox(
        "Select your output language",
        list(output_language_dict.keys()),
        index=list(output_language_dict.values()).index(st.session_state.output_language)
    )
    output_language = output_language_dict[out_lang]

    english_accent = st.sidebar.selectbox(
        "Select your English accent",
        list(tld_dict.keys()),
        index=list(tld_dict.values()).index(st.session_state.english_accent)
    )
    tld = tld_dict[english_accent]

    st.sidebar.subheader("Google AI Key")
    google_api_key = st.sidebar.text_input('Enter your Google AI Key', type="password")
    st.sidebar.info("A Google API Key is required to access certain services from Google, such as the Google Knowledge Graph API. You can obtain a Google API Key by following the instructions [here](https://aistudio.google.com/app/apikey).")

    return output_language, tld, google_api_key

# Initialize session state
def initialize_session_state():
    if 'google_api_key' not in st.session_state:
        st.session_state.google_api_key = None
    if 'prompt' not in st.session_state:
        st.session_state.prompt = ""
    if 'explore_more_prompt' not in st.session_state:
        st.session_state.explore_more_prompt = ""
    if 'info' not in st.session_state:
        st.session_state.info = ""
    if 'knowledge_graph_info' not in st.session_state:
        st.session_state.knowledge_graph_info = ""
    if 'last_prompt' not in st.session_state:
        st.session_state.last_prompt = ""
    if 'translated_text' not in st.session_state:
        st.session_state.translated_text = ""
    if 'audio_file_path' not in st.session_state:
        st.session_state.audio_file_path = ""
    if 'output_language' not in st.session_state:
        st.session_state.output_language = "en"
    if 'english_accent' not in st.session_state:
        st.session_state.english_accent = "com"

# Dictionaries for output language and English accent mappings
output_language_dict = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Korean": "ko",
    "Chinese": "zh-cn",
    "Japanese": "ja"
}

tld_dict = {
    "Default": "com",
    "India": "co.in",
    "United Kingdom": "co.uk",
    "United States": "com",
    "Canada": "ca",
    "Australia": "com.au",
    "Ireland": "ie",
    "South Africa": "co.za"
}

def create_bar_chart(info):
    data = []
    for line in info.split("\n"):
        if '-' in line:
            parts = line.split('-')
            if len(parts) >= 2:  # Ensure there are at least two parts after splitting by '-'
                title = parts[0].strip()
                number_info = parts[1].strip().split(',')[0]
                # Check if number_info contains digits before attempting conversion
                if any(char.isdigit() for char in number_info):
                    number = int(''.join(filter(str.isdigit, number_info)))
                    data.append((title, number))

    if not data:
        st.warning("No valid numeric data found in Wikipedia information.")
        return None
    else:
        df = pd.DataFrame(data, columns=['Title', 'Number'])
        fig = px.bar(df, x='Title', y='Number', title='Wikipedia Information Overview')
        return fig

    

def create_knowledge_graph_visualizations(knowledge_graph_info):
    # Extract facts and figures dynamically
    facts_and_figures = {}
    lines = knowledge_graph_info.splitlines()
    for line in lines:
        if ':' in line:
            key_value = line.split(':', 1)
            facts_and_figures[key_value[0].strip()] = key_value[1].strip()

    # Convert facts and figures into a DataFrame
    df = pd.DataFrame(list(facts_and_figures.items()), columns=['Metric', 'Value'])

    # Create charts and graphs using Plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df['Metric'],
        x=df['Value'],
        orientation='h'
    ))
    fig.update_layout(title='Facts and Figures', xaxis_title='Value', yaxis_title='Metric')

    return fig

# Main application function
def main():
    # Configure sidebar and initialize session state
    initialize_session_state()
    output_language, tld, google_api_key = configure_sidebar()
    st.session_state.output_language = output_language
    st.session_state.english_accent = tld

    # Main content area
    st.markdown("<h1 style='text-align: center; color: white; padding: 10px; background-color: #002147;'>SmartTutor</h1>", unsafe_allow_html=True)
    st.markdown("---")  # Horizontal rule

    if google_api_key:
        st.session_state.google_api_key = google_api_key
        genai.configure(api_key=google_api_key)
        newmodel = genai.GenerativeModel(model_name="gemini-1.5-pro")

        # Input prompt
        st.session_state.prompt = st.text_input('Enter your topic here to get result', value=st.session_state.prompt)

        if st.session_state.google_api_key and st.session_state.prompt:
            if st.session_state.prompt != st.session_state.last_prompt:
                
                # Apply differential privacy to the input prompt
                dp_prompt = detect_and_apply_privacy(st.session_state.prompt)
                st.session_state.prompt = dp_prompt
                st.session_state.last_prompt = dp_prompt

                st.session_state.info, st.session_state.knowledge_graph_info = retrieve_and_process_wikipedia_info(st.session_state.prompt)

            # Display Wikipedia search results within an expander
            with st.expander("Wikipedia Search Results", expanded=False):
                st.write(wiki.run(st.session_state.prompt))

            st.markdown("---")

            # Split the info string into individual topic-response pairs
            topic_responses = st.session_state.info.split("\n\n") 

            # Set up columns for layout
            col1, col2 = st.columns(2)

            # Column 1: Wikipedia results
            with col1:
                st.write("### Wikipedia Results")
                for topic_response in topic_responses:
                    st.write(topic_response)
            
            # Column 2: Knowledge graph information
            with col2:
                st.write("### Knowledge Graph Information")
                st.write(st.session_state.knowledge_graph_info)

            st.markdown("---")

            # Bar chart
            st.write("### knowlegde graph Visualization")
            fig = create_knowledge_graph_visualizations(st.session_state.knowledge_graph_info)
            st.plotly_chart(fig)

            # Bar chart
            st.write("### Bar Chart Visualization")
            fig = create_bar_chart(st.session_state.info)
            st.plotly_chart(fig)

            st.markdown("---")

            # Suggest related topics based on the user's input
            related_topics = generate_related_topics(st.session_state.prompt)

            # Create a dropdown for related topics
            selected_related_topic = st.selectbox("Related Topics", related_topics)

            # Display information based on the selected related topic
            if selected_related_topic:
                st.write(f"You selected: {selected_related_topic}")
                st.write(wiki.run(selected_related_topic))

            st.markdown("---")
            # Explore more section
            st.session_state.explore_more_prompt = st.text_input('Explore more on the topic:', value=st.session_state.explore_more_prompt)

            if st.session_state.explore_more_prompt:
                # Apply differential privacy to the input prompt
                dp_prompt = detect_and_apply_privacy(st.session_state.explore_more_prompt)

                st.session_state.explore_more_prompt = dp_prompt

                responsenew = newmodel.generate_content(st.session_state.explore_more_prompt)
                updatedres = responsenew._result.candidates[0].content.parts[0].text
                with st.expander("AI Results", expanded=False):
                    st.write(updatedres)

                col3, col4 = st.columns([2, 3])
                with col3:
                    if st.button("Translate"):
                        translated_text = text_translate(input_language='en', translate_language=output_language, text=updatedres)
                        st.session_state.translated_text = translated_text
                        st.write(st.session_state.translated_text)
                    # Display translated text if it exists in session state
                    if st.session_state.translated_text:
                        st.write(st.session_state.translated_text)

                with col4:
                    if st.button("Convert to Speech"):
                        remove_files(1)
                        audio_file = text_to_speech(st.session_state.translated_text, tld)
                        st.session_state.audio_file_path = f"temp/{audio_file}.mp3"
                    # Display audio file if it exists in session state
                    if st.session_state.audio_file_path:
                        audio_file = open(st.session_state.audio_file_path, "rb")
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format="audio/mp3")

if __name__ == "__main__":
    main()
