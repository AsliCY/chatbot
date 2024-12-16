

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# GPU kontrolü
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Model yükleme
model_name = "microsoft/DialoGPT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def chatbot(message, history):
    try:
        # Tokenize ve model inputunu hazırla
        inputs = tokenizer.encode(message + tokenizer.eos_token, return_tensors='pt').to(device)
        
        # Yanıt oluştur
        outputs = model.generate(
            inputs,
            max_length=100,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Kullanıcı mesajını yanıttan çıkar
        response = response.replace(message, "").strip()
        
        # Eğer yanıt hala kullanıcı mesajıyla başlıyorsa, o kısmı kes
        if response.lower().startswith(message.lower()):
            response = response[len(message):].strip()
            
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"
# Arayüz oluşturma
interface = gr.ChatInterface(
    chatbot,
    title="AI Chatbot",
    description="GPT based chat bot",
    theme="soft",
    examples=["Hello!", "How are you?", "Can you tell me a story?"],
)

# Arayüzü başlat
interface.launch(share=True)
