import streamlit as st
#from transformers import AutoTokenizer
#from huggingface_hub import InferenceClient
import requests

# Access Hugging Face API token from Streamlit secrets
api_token = st.secrets["HF_ACCESS_TOKEN"]

# A list of models that the user can choose from
models = {
    "LLaMA": "meta-llama/Llama-3.2-3B-Instruct",  # Replace with actual model paths if necessary
    "Granite": "ibm-granite/granite-3.0-2b-instruct",
   # "Mistral": "mistralai/Mistral-7B-Instruct-v0.3"
    "Mistral":  "mistralai/Mixtral-8x7B-Instruct-v0.1" 
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
            "decoding_method": "greedy",
            "max_length": 150,
            "max_new_tokens": 1000,
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
