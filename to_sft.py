#!/usr/bin/env python3
"""
Convert misleading samples to SFT format for training.

For misleading samples (k > 0):
- Full conversation: system -> user (prompt) -> assistant (questions) -> user (answers) -> assistant (standardized)

For clean samples (k = 0):
- Simplified: system -> user (prompt) -> assistant (is_misleading: false, standardized_prompt)
"""

import json
import argparse
import sys
import os
import random

# Add the CAD_Agent directory to the path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.clarification import ASK_AGENT_SYSTEM_PROMPT

# Use the system message from config
SYSTEM_MESSAGE = ASK_AGENT_SYSTEM_PROMPT


def parse_questions(questions_text):
    """Parse questions from the questions_to_ask field"""
    if not questions_text:
        return []
    # Remove the "ANSWER_TO_QUESTIONS" part if present
    questions_text = questions_text.split("ANSWER_TO_QUESTIONS")[0].strip()
    # Split by newlines and filter out empty lines
    questions = [q.strip() for q in questions_text.split("\n") if q.strip() and q.strip().startswith("-")]
    # Remove the leading "- " from each question
    questions = [q[2:].strip() if q.startswith("- ") else q.strip() for q in questions]
    return questions


def parse_answers(answers_text):
    """Parse answers from the answer_to_questions field"""
    if not answers_text:
        return ""
    # Remove leading "- " if present
    answers_text = answers_text.strip()
    if answers_text.startswith("- "):
        answers_text = answers_text[2:]
    return answers_text


def create_misleading_conversation(sample):
    """Create conversation format for misleading samples (k > 0)"""
    messages = []
    
    # System message
    messages.append({
        "role": "system",
        "content": SYSTEM_MESSAGE
    })
    
    # User message with misleading description
    user_prompt = f"Analyze the following CAD generation prompt:\n\n{sample.get('misleading_description', '')}"
    messages.append({
        "role": "user",
        "content": user_prompt
    })
    
    # Assistant response with questions
    questions_text = sample.get('questions_to_ask', '')
    questions = parse_questions(questions_text)
    
    questions_json = json.dumps({"is_misleading": True, "questions": questions}, indent=2)
    messages.append({
        "role": "assistant",
        "content": f"```json\n{questions_json}\n```"
    })
    
    # User message with answers
    answers_text = sample.get('answer_to_questions', '')
    answers = parse_answers(answers_text)
    messages.append({
        "role": "user",
        "content": f"- {answers}" if answers else "- (No answer provided)"
    })
    
    # Assistant response with standardized prompt (use original_prompt as the corrected version)
    standardized_prompt = sample.get('original_prompt', '')
    standardized_json = json.dumps({
        "is_misleading": False,
        "standardized_prompt": standardized_prompt
    }, indent=2)
    messages.append({
        "role": "assistant",
        "content": f"```json\n{standardized_json}\n```"
    })
    
    return messages


def create_clean_conversation(sample):
    """Create conversation format for clean samples (k = 0)"""
    messages = []
    
    # System message
    messages.append({
        "role": "system",
        "content": SYSTEM_MESSAGE
    })
    
    # User message with original prompt
    original_prompt = sample.get('original_prompt', '')
    user_prompt = f"Analyze the following CAD generation prompt:\n\n{original_prompt}"
    messages.append({
        "role": "user",
        "content": user_prompt
    })
    
    # Assistant response with is_misleading: false and standardized_prompt
    standardized_prompt = original_prompt  # For clean samples, standardized = original
    standardized_json = json.dumps({
        "is_misleading": False,
        "standardized_prompt": standardized_prompt
    }, indent=2)
    messages.append({
        "role": "assistant",
        "content": f"```json\n{standardized_json}\n```"
    })
    
    return messages


def convert_to_sft(input_file, output_file):
    """Convert misleading samples to SFT format"""
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    samples = data.get('samples', data if isinstance(data, list) else [])
    print(f"Total samples: {len(samples)}")
    
    # Convert to SFT format
    sft_samples = []
    for sample in samples:
        k = sample.get('k', -1)
        
        if k == 0:
            # Clean sample
            messages = create_clean_conversation(sample)
        else:
            # Misleading sample
            messages = create_misleading_conversation(sample)
        
        # Create SFT sample
        sft_sample = {
            "messages": messages,
            "uid": sample.get('uid'),
            "k": k
        }
        
        # Add additional fields if they exist
        if 'original_cd' in sample:
            sft_sample['original_cd'] = sample['original_cd']
        if 'misleading_cd' in sample:
            sft_sample['misleading_cd'] = sample['misleading_cd']
        if 'config_name' in sample:
            sft_sample['config_name'] = sample['config_name']
        
        sft_samples.append(sft_sample)
    
    print(f"Converted {len(sft_samples)} samples to SFT format")
    
    # Shuffle samples before saving
    print("Shuffling samples...")
    random.shuffle(sft_samples)
    
    # Save output
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(sft_samples, f, indent=2)
    
    print(f"✓ Successfully saved to {output_file}")
    print(f"  - Total samples: {len(sft_samples)}")
    print(f"  - Clean samples (k=0): {len([s for s in sft_samples if s.get('k') == 0])}")
    print(f"  - Misleading samples (k>0): {len([s for s in sft_samples if s.get('k', 0) > 0])}")


def main():
    parser = argparse.ArgumentParser(description='Convert misleading samples to SFT format')
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSON file with misleading samples')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file for SFT format')
    
    args = parser.parse_args()
    
    try:
        convert_to_sft(args.input, args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
