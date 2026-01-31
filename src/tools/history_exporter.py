import json
import os

def export_to_markdown(input_file="conversation_history.jsonl", output_file="history.md"):
    if not os.path.exists(input_file):
        print(f"File {input_file} not found.")
        return
    
    with open(input_file, "r") as f, open(output_file, "w") as out:
        out.write("# Conversation History\n\n")
        for line in f:
            if line.strip():
                data = json.loads(line)
                role = data.get('role', 'Unknown').capitalize()
                content = data.get('content', '')
                out.write(f"### {role}\n{content}\n\n")
    print(f"Exported to {output_file}")

if __name__ == "__main__":
    export_to_markdown()
