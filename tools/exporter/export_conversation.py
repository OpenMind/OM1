import json
import argparse
import csv
import os
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="OM1 Conversation Exporter")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--format", choices=["markdown", "json", "csv", "html"], default="markdown")
    parser.add_argument("--input", required=True, help="Path to raw JSONL history")
    return parser.parse_args()

def load_history(path):
    history = []
    with open(path, "r") as f:
        for line in f:
            history.append(json.loads(line))
    return history

def to_markdown(history, agent, session):
    output = f"# Conversation: {agent}\nDate: {session}\n\n"
    for i, entry in enumerate(history):
        role = entry.get("role", "system")
        content = entry.get("content", "")
        ts = entry.get("timestamp", "")
        output += f"## Exchange {i+1} ({ts})\n"
        output += f"**{role.capitalize()}:** {content}\n\n"
    return output

def run_export():
    args = parse_args()
    history = load_history(args.input)
    filename = f"{args.agent}_{args.session}.{args.format if args.format != 'markdown' else 'md'}"
    
    if args.format == "markdown":
        res = to_markdown(history, args.agent, args.session)
        with open(filename, "w") as f: f.write(res)
    elif args.format == "json":
        with open(filename, "w") as f: json.dump(history, f, indent=2)
    
    print(f"✅ Exported to {filename}")

if __name__ == "__main__":
    run_export()
