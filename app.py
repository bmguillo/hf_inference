import streamlit as st
import requests

# Access Hugging Face API token from Streamlit secrets
api_token = st.secrets["huggingface"]["api_token"]

# A list of models that the user can choose from
models = {
    "LLaMA 3B Instruct": "meta-llama/Llama-3B-Instruct",  # Replace with actual model paths if necessary
    "GPT-2": "gpt2",
    "GPT-Neo 1.3B": "EleutherAI/gpt-neo-1.3B",
    "DistilGPT-2": "distilgpt2"
}

# Function to call the Hugging Face Inference API
def generate_response(model_name, input_text):
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    
    headers = {
        "Authorization": f"Bearer {api_token}"
    }
    
    payload = {
        "inputs": input_text,
        "parameters": {
            "max_length": 150,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50
        }
    }
    
    # Send the request to Hugging Face API
    response = requests.post(url, headers=headers, json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        output = response.json()
        return output[0]['generated_text']
    else:
        return f"Error: {response.status_code} - {response.text}"

# Streamlit UI
st.title("LLM Inference API with Hugging Face")

# Dropdown for selecting a model
model_choice = st.selectbox("Select a model", list(models.keys()))

# Text input for the prompt
input_prompt = st.text_area("Enter your prompt:", "What are the health benefits of drinking water?")

# When the user submits, generate the response
if st.button("Generate Response"):
    if input_prompt.strip():
        # Get the model name from the selected dropdown option
        selected_model = models[model_choice]
        
        # Generate response using the Inference API
        with st.spinner(f"Generating response using {model_choice}..."):
            result = generate_response(selected_model, input_prompt)
        
        # Display the result
        st.subheader("Generated Response:")
        st.write(result)
    else:
        st.warning("Please enter a valid prompt.")
