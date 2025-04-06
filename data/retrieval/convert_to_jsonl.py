#!/usr/bin/env python3

import sys
import yaml
import json


def convert_to_jsonl(yaml_path, jsonl_path):
    """
    Reads a YAML file containing the nested data structure,
    then writes a JSONL file with one or more lines per top-level entry.
    Specifically, this script creates two lines per top-level key: one
    for the normal question (q_str) and one for the reversed question (reversed_q_str).
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    with open(jsonl_path, 'w', encoding='utf-8') as out_f:
        for top_key, record in data.items():
            metadata = record.get('metadata', {})

            # Extract the two questions
            normal_q = metadata.get('q_str', None)
            reversed_q = metadata.get('reversed_q_str', None)

            # For convenience, you might store all metadata or just the keys you need.
            # Here we store the entire metadata block, but you can tailor fields if desired.
            # Example: prop_id, x_name, x_value, y_name, y_value, etc.
            out_metadata = {
                'prop_id': metadata.get('prop_id'),
                'comparison': metadata.get('comparison'),
                'x_name': metadata.get('x_name'),
                'x_value': metadata.get('x_value'),
                'y_name': metadata.get('y_name'),
                'y_value': metadata.get('y_value'),
                # Add more fields from metadata as needed
            }

            # 1) Normal question
            if normal_q:
                normal_entry = {
                    "id": f"{top_key}-normal",
                    "question": normal_q,
                    "metadata": out_metadata
                }
                out_f.write(json.dumps(normal_entry) + "\n")

            # 2) Reversed question
            if reversed_q:
                reversed_entry = {
                    "id": f"{top_key}-reversed",
                    "question": reversed_q,
                    "metadata": out_metadata
                }
                out_f.write(json.dumps(reversed_entry) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_data.py <input_yaml_path> <output_jsonl_path>")
        sys.exit(1)
    
    input_yaml = sys.argv[1]
    output_jsonl = sys.argv[2]
    convert_to_jsonl(input_yaml, output_jsonl)
