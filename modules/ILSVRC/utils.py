import re

def extract_number(filename: str):
    match = re.findall(r"\d+", filename)
    return int("".join(match)) if match else 0