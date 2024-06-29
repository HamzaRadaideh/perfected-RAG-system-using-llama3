import tkinter as tk
from tkinter import messagebox
from llama_handler import run_llama3
from conversation_logger import log_conversation

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

