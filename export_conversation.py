#!/usr/bin/env python3
"""
OM1 Conversation Exporter
Export agent conversations to Markdown format
"""

import json
import argparse
from pathlib import Path

# Example conversation data
EXAMPLE_CONVERSATION = {
    "agent": "spot",
    "session_id": "spot_20260130_121500",
    "start_time": "2026-01-30T12:15:00",
    "exchanges": [
        {
            "timestamp": "2026-01-30T12:15:23",
            "user_input": "Pick up the red ball",
            "vision": "Detected red sphere at (0.5, 0.2)",
            "llm_decision": "movement(0.5, 0, 0)",
            "action": "Moving forward..."
        },
        {
            "timestamp": "2026-01-30T12:15:45",
            "user_input": "Place it on the table",
            "vision": "Detected table at (1.2, 0.5)",
            "llm_decision": "movement(1.2, 0.5, 0) then place()",
            "action": "Placing object..."
        },
        {
            "timestamp": "2026-01-30T12:16:10",
            "user_input": "What do you see?",
            "vision": "No objects detected",
            "llm_decision": "speech('Task completed')",
            "action": "Speaking..."
        }
    ]
}


def export_markdown(conversation, output_file):
    """Export to Markdown"""
    
    content = f"# Conversation: {conversation['agent']}\n\n"
    content += f"**Session:** {conversation['session_id']}\n"
    content += f"**Date:** {conversation['start_time']}\n\n"
    content += "---\n\n"
    
    for i, ex in enumerate(conversation['exchanges'], 1):
        content += f"## Exchange {i}\n"
        content += f"**Time:** {ex['timestamp']}\n\n"
        content += f"**User:** {ex['user_input']}\n\n"
        content += f"**Vision:** {ex['vision']}\n\n"
        content += f"**LLM:** `{ex['llm_decision']}`\n\n"
        content += f"**Action:** {ex['action']}\n\n"
        content += "---\n\n"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Export OM1 conversations')
    parser.add_argument('--format', default='markdown', help='Export format')
    parser.add_argument('--output', default='conversations', help='Output directory')
    parser.add_argument('--example', action='store_true', help='Use example data')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Use example conversation
    conversation = EXAMPLE_CONVERSATION
    session_id = conversation['session_id']
    
    print("\n" + "="*50)
    print("  OM1 Conversation Exporter")
    print("="*50 + "\n")
    
    # Export to markdown
    output_file = output_dir / f"{session_id}.md"
    export_markdown(conversation, output_file)
    
    print(f"✓ Exported to: {output_file}")
    print(f"\nExport complete!\n")


if __name__ == "__main__":
    main()