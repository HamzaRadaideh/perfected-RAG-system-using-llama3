import tkinter as tk
from tkinter import scrolledtext
import json
import torch
import ollama
import os
from openai import OpenAI
import argparse

import os
import tkinter as tk
from tkinter import filedialog
import PyPDF2
import re
import json

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Initialize variables
system_message = "Welcome to the chat."
conversation_history = []

"""upload"""

# Function to convert PDF to text and append to vault.txt
def convert_pdf_to_text():
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if file_path:
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            text = ''
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                if page.extract_text():
                    text += page.extract_text() + " "
            
            # Normalize whitespace and clean up text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split text into chunks by sentences, respecting a maximum chunk size
            sentences = re.split(r'(?<=[.!?]) +', text)  # split on spaces following sentence-ending punctuation
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                # Check if the current sentence plus the current chunk exceeds the limit
                if len(current_chunk) + len(sentence) + 1 < 1000:  # +1 for the space
                    current_chunk += (sentence + " ").strip()
                else:
                    # When the chunk exceeds 1000 characters, store it and start a new one
                    chunks.append(current_chunk)
                    current_chunk = sentence + " "
            if current_chunk:  # Don't forget the last chunk!
                chunks.append(current_chunk)
            with open("vault.txt", "a", encoding="utf-8") as vault_file:
                for chunk in chunks:
                    # Write each chunk to its own line
                    vault_file.write(chunk.strip() + "\n")  # Two newlines to separate cShunks
            print(f"PDF content appended to vault.txt with each chunk on a separate line.")

# Function to upload a text file and append to vault.txt
def upload_text_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, 'r', encoding="utf-8") as txt_file:
            text = txt_file.read()
            
            # Normalize whitespace and clean up text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split text into chunks by sentences, respecting a maximum chunk size
            sentences = re.split(r'(?<=[.!?]) +', text)  # split on spaces following sentence-ending punctuation
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                # Check if the current sentence plus the current chunk exceeds the limit
                if len(current_chunk) + len(sentence) + 1 < 1000:  # +1 for the space
                    current_chunk += (sentence + " ").strip()
                else:
                    # When the chunk exceeds 1000 characters, store it and start a new one
                    chunks.append(current_chunk)
                    current_chunk = sentence + " "
            if current_chunk:  # Don't forget the last chunk!
                chunks.append(current_chunk)
            with open("vault.txt", "a", encoding="utf-8") as vault_file:
                for chunk in chunks:
                    # Write each chunk to its own line
                    vault_file.write(chunk.strip() + "\n")  # Two newlines to separate chunks
            print(f"Text file content appended to vault.txt with each chunk on a separate line.")

# Function to upload a JSON file and append to vault.txt
def upload_jsonfile():
    file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
    if file_path:
        with open(file_path, 'r', encoding="utf-8") as json_file:
            data = json.load(json_file)
            
            # Flatten the JSON data into a single string
            text = json.dumps(data, ensure_ascii=False)
            
            # Normalize whitespace and clean up text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split text into chunks by sentences, respecting a maximum chunk size
            sentences = re.split(r'(?<=[.!?]) +', text)  # split on spaces following sentence-ending punctuation
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                # Check if the current sentence plus the current chunk exceeds the limit
                if len(current_chunk) + len(sentence) + 1 < 1000:  # +1 for the space
                    current_chunk += (sentence + " ").strip()
                else:
                    # When the chunk exceeds 1000 characters, store it and start a new one
                    chunks.append(current_chunk)
                    current_chunk = sentence + " "
            if current_chunk:  # Don't forget the last chunk!
                chunks.append(current_chunk)
            with open("vault.txt", "a", encoding="utf-8") as vault_file:
                for chunk in chunks:
                    # Write each chunk to its own line
                    vault_file.write(chunk.strip() + "\n")  # Two newlines to separate chunks
            print(f"JSON file content appended to vault.txt with each chunk on a separate line.")

# Create the main window
root = tk.Tk()
root.title("Upload .pdf, .txt, or .json")

# Create a button to open the file dialog for PDF
pdf_button = tk.Button(root, text="Upload PDF", command=convert_pdf_to_text)
pdf_button.pack(pady=10)

# Create a button to open the file dialog for text file
txt_button = tk.Button(root, text="Upload Text File", command=upload_text_file)
txt_button.pack(pady=10)

# Create a button to open the file dialog for JSON file
json_button = tk.Button(root, text="Upload JSON File", command=upload_jsonfile)
json_button.pack(pady=10)

# Run the main event loop
root.mainloop()

"""localrag"""

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Function to get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    # Encode the rewritten input
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

def rewrite_query(user_input_json, conversation_history, ollama_model):
    user_input = json.loads(user_input_json)["Query"]
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:
    
    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query
    
    Return ONLY the rewritten query text, without any additional formatting or explanations.
    
    Conversation History:
    {context}
    
    Original query: [{user_input}]
    
    Rewritten query: 
    """
    response = client.chat.completions.create(
        model=ollama_model,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200,
        n=1,
        temperature=0.1,
    )
    rewritten_query = response.choices[0].message.content.strip()
    return json.dumps({"Rewritten Query": rewritten_query})
   
def ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history):
    conversation_history.append({"role": "user", "content": user_input})
    
    if len(conversation_history) > 1:
        query_json = {
            "Query": user_input,
            "Rewritten Query": ""
        }
        rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history, ollama_model)
        rewritten_query_data = json.loads(rewritten_query_json)
        rewritten_query = rewritten_query_data["Rewritten Query"]
        print(PINK + "Original Query: " + user_input + RESET_COLOR)
        print(PINK + "Rewritten Query: " + rewritten_query + RESET_COLOR)
    else:
        rewritten_query = user_input
    
    relevant_context = get_relevant_context(rewritten_query, vault_embeddings, vault_content)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)
    
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str
    
    conversation_history[-1]["content"] = user_input_with_context
    
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    
    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=2000,
    )
    
    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
    
    return response.choices[0].message.content

# Function to save chat history to a JSON log file
def save_chat_to_log(message):
    log_dir = "json"  # Define log directory

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "data.json")

    # Load existing log if available
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        with open(log_file, "r") as f:
            log_entry = json.load(f)
    else:
        log_entry = []

    # Append new conversation entry
    log_entry.append({"You": message["You"], "LLaMA": message["LLaMA"]})

    # Write updated log
    with open(log_file, "w") as f:
        json.dump(log_entry, f, indent=4)

    print("Chat history saved to data.json")

# Parse command-line arguments
print(NEON_GREEN + "Parsing command-line arguments..." + RESET_COLOR)
parser = argparse.ArgumentParser(description="Ollama Chat")
parser.add_argument("--model", default="llama3", help="Ollama model to use (default: llama3)")
args = parser.parse_args()

# Configuration for the Ollama API client
print(NEON_GREEN + "Initializing Ollama API client..." + RESET_COLOR)
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='llama3'
)

# Load the vault content
print(NEON_GREEN + "Loading vault content..." + RESET_COLOR)
vault_content = []
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding='utf-8') as vault_file:
        vault_content = vault_file.readlines()

# Generate embeddings for the vault content using Ollama
print(NEON_GREEN + "Generating embeddings for the vault content..." + RESET_COLOR)
vault_embeddings = []
for content in vault_content:
    response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
    vault_embeddings.append(response["embedding"])

# Convert to tensor and print embeddings
print("Converting embeddings to tensor...")
vault_embeddings_tensor = torch.tensor(vault_embeddings) 
print("Embeddings for each line in the vault:")
print(vault_embeddings_tensor)

# GUI Integration
def send_message():
    user_input = user_input_text.get("1.0", tk.END).strip()
    if user_input:
        response = ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})
        update_chat_history()
        save_chat_to_log({"You": user_input, "LLaMA": response})
        user_input_text.delete("1.0", tk.END)

def update_chat_history():
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, f"You: {conversation_history[-2]['content']}\n", "user")
    chat_history.insert(tk.END, f"LLaMA: {conversation_history[-1]['content']}\n", "assistant")
    chat_history.config(state=tk.DISABLED)
    chat_history.see(tk.END)

root = tk.Tk()
root.title("LLaMA Chat")

chat_history = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=20)
chat_history.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
chat_history.tag_config("user", foreground="blue")
chat_history.tag_config("assistant", foreground="green")
chat_history.config(state=tk.DISABLED)

user_input_text = tk.Text(root, wrap=tk.WORD, width=50, height=2)
user_input_text.grid(row=1, column=0, padx=10, pady=10)

send_button = tk.Button(root, text="Send", width=10, command=send_message)
send_button.grid(row=1, column=1, padx=10, pady=10)

root.mainloop()
