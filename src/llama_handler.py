import subprocess
from tkinter import messagebox

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
