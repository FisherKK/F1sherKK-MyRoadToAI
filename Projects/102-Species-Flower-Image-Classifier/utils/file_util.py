import json

def load_json(json_filepath):
    with open(json_filepath, "r") as f:
        return json.load(f)