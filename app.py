import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Model, GPT2Config, AutoModelForCausalLM

# Load the model and tokenizer from Hugging Face
def load_model():
    model = AutoModelForCausalLM.from_pretrained("Nishantc05/qa-gptmodel")
    device="cuda"
    model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("Nishantc05/qa-gptmodel", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
    tokenizer.padding_side = "left"
    tokenizer.pad_token=tokenizer.eos_token
    return tokenizer, model

st.title("Question Answer Generator")

# Text input from user
email_content = st.text_area("Enter A Question:", height=200)

if st.button("Generate Answer"):
    if email_content:
        # Tokenize and generate subject line
        model = AutoModelForCausalLM.from_pretrained("Nishantc05/qa-gptmodel")
        device="cuda"
        model.to(device)
        tokenizer = GPT2Tokenizer.from_pretrained("Nishantc05/qa-gptmodel", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
        tokenizer.padding_side = "left"
        tokenizer.pad_token=tokenizer.eos_token
        inputs = tokenizer.encode(email_content, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=20, num_beams=5, early_stopping=True)
        subject_line = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("Generated Answer...")
        st.write(subject_line)
    else:
        st.write("Please enter some email content to generate a subject line.")
