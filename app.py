#Imports
from huggingface_hub import InferenceClient
from transformers import pipeline
import gradio as gr
from gtts import gTTS
import os
import torch

# --- Initialization ---
# LLM Chatbot API Client
# Replace with your actual API key
HF_API_KEY = os.getenv("HF_API_KEY")
client = InferenceClient(api_key=os.getenv("HF_API_KEY"))

# Speech-to-Text Pipeline
MODEL_NAME = "openai/whisper-large-v3"
device = 0 if torch.cuda.is_available() else "cpu"
pipe = pipeline(task="automatic-speech-recognition", model=MODEL_NAME, chunk_length_s=30, device=device)

# --- Functions ---

# 1. Speech-to-Text Conversion
def speech_to_text(audio_file):
    if audio_file is None:
        return "Error: No audio provided!"
    result = pipe(audio_file)["text"]
    return result

# 2. Generate LLM Response
def generate_response(text_input):
    messages = [{"role": "user", "content": text_input}]
    completion = client.chat.completions.create(
        model="mistralai/Mistral-Nemo-Instruct-2407",
        messages=messages,
        max_tokens=500
    )
    response = completion.choices[0].message.content
    return response

# 3. Text-to-Speech Conversion
def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    file_path = "response.mp3"
    tts.save(file_path)
    return file_path

# 4. Final Process: Speech-to-LLM-to-Speech
def process_audio(audio_file):
    # Speech-to-Text
    text = speech_to_text(audio_file)
    
    # Generate Response from LLM
    llm_response = generate_response(text)
    
    # Convert LLM Response to Speech
    audio_response = text_to_speech(llm_response)
    
    return text, llm_response, audio_response


# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("## 🗣️ Speech-to-Speech App")
    
    with gr.Tab("Microphone"):
        audio_input = gr.Audio(sources="microphone", type="filepath", label="Speak Something")
        text_output = gr.Textbox(label="Text")
        llm_response_output = gr.Textbox(label="LLM Response")
        audio_response_output = gr.Audio(label="Audio Response")

    submit_button = gr.Button("Process")
    
    # Connect the button to the processing function
    submit_button.click(
        process_audio, 
        inputs=audio_input, 
        outputs=[text_output, llm_response_output, audio_response_output]
    )

# Launch the App
if __name__ == "__main__":
    demo.launch()
