import streamlit as st
import math
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import seaborn as sns
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set page config
st.set_page_config(
    page_title="OpenAI Token Probability Visualizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state to store settings
def init_session_state():
    defaults = {
        'model': "gpt-4o",
        'system_msg': "return a JSON containing 2 boolean variables: \"positive_sentiment\" that returns true if the adjective is positive and false otherwise, and \"confident_in_answer\" that returns true if you are confident in your assessment and false if you are uncertain",
        'temperature': 0.0,
        'prompt': "surprised",
        'analysis_results': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- Sidebar Configuration --- 
st.sidebar.header("⚙️ Configuration")

# Model configuration
with st.sidebar.expander("🤖 Model Settings", expanded=True):
    model_options = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "Custom"]
    selected_model_option = st.selectbox(
        "Select Model", 
        model_options, 
        index=model_options.index(st.session_state.model) if st.session_state.model in model_options else model_options.index("Custom"),
        key="model_selection"
    )

    # Custom model input
    if selected_model_option == "Custom":
        custom_model = st.text_input("Enter custom model name", value=st.session_state.model if st.session_state.model not in model_options[:-1] else "gpt-4-turbo", key="custom_model")
        current_model = custom_model
    else:
        current_model = selected_model_option

    # Temperature slider
    current_temperature = st.slider("🌡️ Temperature", min_value=0.0, max_value=2.0, value=st.session_state.temperature, step=0.1, key="temp_slider")

    if st.button("Apply Model Settings"):
        st.session_state.model = current_model
        st.session_state.temperature = current_temperature
        st.success(f"Model: {st.session_state.model}, Temp: {st.session_state.temperature} applied!")

# System message input
with st.sidebar.expander("📝 System Message", expanded=True):
    current_system_message = st.text_area(
        "Instructions for the AI", 
        value=st.session_state.system_msg,
        height=150,
        key="system_msg_input",
        help="This sets the behavior of the AI model."
    )

    if st.button("Apply System Message"):
        st.session_state.system_msg = current_system_message
        st.success("System message updated!")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("Visualizes token probabilities from OpenAI models.")

# --- Main Content Area --- 
st.title("📊 OpenAI Token Probability Visualizer")

# Display current settings
st.caption(f"Current Model: `{st.session_state.model}` | Temperature: `{st.session_state.temperature}`")

# User message input
st.subheader("💬 User Message")
user_prompt = st.text_area("Type your message here:", value=st.session_state.prompt, height=100, key="user_prompt_input", label_visibility="collapsed")

# --- Analysis Function --- 
def get_token_probabilities():
    try:
        st.session_state.analysis_results = None # Clear previous results
        st.info(f"Sending message to **{st.session_state.model}** with temperature **{st.session_state.temperature}**...")
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        with st.spinner("🧠 AI is thinking..."):
            completion = client.chat.completions.create(
                model=st.session_state.model,
                messages=[
                    {"role": "system", "content": st.session_state.system_msg},
                    {"role": "user", "content": st.session_state.prompt}
                ],
                temperature=st.session_state.temperature,
                max_tokens=50,
                logprobs=True,
                top_logprobs=6
            )
            st.session_state.analysis_results = completion # Store results
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.session_state.analysis_results = None

# --- Action Button --- 
if st.button("🚀 Generate and Analyze", type="primary", use_container_width=True):
    st.session_state.prompt = user_prompt # Update prompt in state before analysis
    get_token_probabilities()

# --- Display Results --- 
if st.session_state.analysis_results:
    completion = st.session_state.analysis_results
    
    st.divider()
    st.subheader("📝 Generated Response")
    st.markdown(completion.choices[0].message.content)
    
    st.divider()
    st.subheader("🔍 Token-by-Token Analysis")
    
    # Close any previously opened figures to avoid warning
    plt.close('all')
    
    logprobs_content = None
    # Safely access logprobs content, checking each part of the path
    if (completion.choices and 
        len(completion.choices) > 0 and
        hasattr(completion.choices[0], 'logprobs') and
        completion.choices[0].logprobs is not None and
        hasattr(completion.choices[0].logprobs, 'content')):
        logprobs_content = completion.choices[0].logprobs.content
    
    if not logprobs_content: # This handles if logprobs_content remained None or is an empty list
        st.warning("Logprobs content not found or is empty in the API response.")
    else:
        for i, token_info in enumerate(logprobs_content):
            with st.container(): # Container for each token's analysis
                token_prob = math.exp(token_info.logprob)
                
                st.write(f"**Token {i+1}:** `{token_info.token}` (Probability: {token_prob:.4f})")
                
                # Prepare data for the bar chart
                tokens = [token_info.token]
                probs = [token_prob]
                
                # Add alternative tokens
                if token_info.top_logprobs:
                    for alt in token_info.top_logprobs:
                        alt_prob = math.exp(alt.logprob)
                        tokens.append(alt.token)
                        probs.append(alt_prob)
                
                # Create DataFrame
                df = pd.DataFrame({
                    'Token': tokens,
                    'Probability': probs
                })
                
                # Sort by probability (descending)
                df = df.sort_values('Probability', ascending=False).reset_index(drop=True)
                
                # Create bar chart with seaborn
                fig, ax = plt.subplots(figsize=(10, max(3, len(df) * 0.5))) # Dynamic height
                bars = sns.barplot(y='Token', x='Probability', data=df, ax=ax, color='steelblue')
                ax.set_title(f"Token {i+1} Probabilities")
                ax.set_xlim(0, 1.1)  # Extend xlim slightly for labels
                
                # Add probability values on the bars
                for j, bar in enumerate(bars.patches):
                    width = bar.get_width()
                    probability = df.iloc[j]['Probability']
                    formatted_prob = f"{probability:.4f}"
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                            formatted_prob, ha='left', va='center', fontsize=9)
                
                ax.set_xlabel('Probability')
                ax.set_ylabel('Token')
                st.pyplot(fig)
                plt.close(fig) # Explicitly close the figure to free resources
