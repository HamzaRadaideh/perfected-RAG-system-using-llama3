import os
import json

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
