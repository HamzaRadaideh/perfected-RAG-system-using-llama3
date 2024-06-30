import subprocess
import os
import json
import tkinter as tk
from tkinter import messagebox

def log_conversation(user_input, llama_output):
    log_dir = "json"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, "data.json")

    # Check if the file exists and is not empty
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        with open(log_file, "r") as f:
            conversation_log = json.load(f)
    else:
        conversation_log = []

    # Append new conversation entry
    conversation_log.append({"You": user_input, "LLaMA": llama_output})

    # Write updated log
    with open(log_file, "w") as f:
        json.dump(conversation_log, f, indent=4)

def send_message():
    user_input = entry.get()
    if user_input.lower() == "exit":
        messagebox.showinfo("Exit", "Exiting the conversation.")
        root.destroy()
        return

    response = run_llama3(conversation_history, user_input)
    if response:
        conversation_history.append(f"You: {user_input}")
        conversation_history.append(f"LLaMA (type exit to leave the program): {response}")
        log_conversation(user_input, response)

        text.config(state="normal")
        text.insert(tk.END, f"You: {user_input}\n")
        text.insert(tk.END, f"LLaMA: {response}\n")
        text.config(state="disabled")
        entry.delete(0, tk.END)

def mainApp():
    global root, entry, text, conversation_history

    root = tk.Tk()
    root.title("LLaMA Conversation")

    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    text = tk.Text(frame, height=10, width=50, state="disabled")
    text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scroll = tk.Scrollbar(frame, command=text.yview)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    text.config(yscrollcommand=scroll.set)

    entry = tk.Entry(root, width=50)
    entry.pack(fill=tk.X)

    button = tk.Button(root, text="Send", command=send_message)
    button.pack(fill=tk.X)

    conversation_history = []

    root.mainloop()


def run_llama3(history, prompt):
    try:
        full_prompt = "\n".join(history + [prompt])
        
        result = subprocess.run(
            ['ollama', 'run', 'llama3'],
            input=full_prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        output = result.stdout.decode()
        error = result.stderr.decode()

        if result.returncode != 0:
            messagebox.showerror("Error", error)
        else:
            return output.strip()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

if __name__ == "__main__":
    mainApp()