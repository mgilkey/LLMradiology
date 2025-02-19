import os
import pandas as pd
import json
import subprocess
import re

# Define input and output file paths
INPUT_CSV_PATH = "input.csv"
OUTPUT_JSON_PATH = "output.json"
MODEL_NAME = "mistral:7b-instruct-q4_K_M"

def clean_json_output(output):
    """Clean markdown formatting from LLM output."""
    output = output.strip()
    if output.startswith("```"):
        lines = output.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        output = "\n".join(lines).strip()
    return output

def run_ollama(prompt):
    """Runs the LLM and ensures JSON output."""
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL_NAME],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True
        )

        # Handle errors if any
        if result.stderr:
            print(f"❌ LLM Error: {result.stderr.strip()}")
            return None
        
        content = result.stdout.strip()
        if not content:
            print("❌ Error: Empty response from LLM.")
            return None

        content = clean_json_output(content)

        # Validate JSON output
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print(f"❌ JSON Decode Error. Raw Output:\n{content}")
            return None

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def get_number_of_specimens(report_text):
    """Extracts the total number of biopsy specimens."""
    prompt = f"""
Extract the total number of biopsy specimens mentioned in the pathology report and list their names. Then record all of the relevant text for 
that specimen in a separate text field. The names may include the words (right, left, base, apex, mid, prostate, and biopsy).  It may also 
look like a numeric or alphabetic list. The name will end if a colon is present. Remove any commas in the name and reorder the name if necessary
 to begin with right or left, then vertical position, and then prostate biopsy. A valid response for a specimen name is right base prostate
 biopsy. If you see right, left prostate biopsy, this means there are two biopsies for right and left and these should be separate specimens. 
Below is an example response, do not infer answers.  If there is no specimen information, add "unknown" as the response. Respond only with a 
JSON object in the format shown below.

Example:
{{
  "number_of_specimens": 2,
  "specimen_names": ["right prostate biopsy", "left prostate biopsy"],
  "specimen_text": ["ADENOCARCINOMA, GLEASON GRADE 3+3 = 6.  - IN ONE SMALL FOCUS (<5%).  - NO EXTRAPROSTATIC EXTENSION SEEN.  ", 
  "GLANDULAR HYPERPLASIA."]
}}

Format:
{{
  "number_of_specimens": <integer>,
  "specimen_names": [text, text],
  "specimen_text": [text]
}}

Report:
{report_text}
"""

    response = run_ollama(prompt)
    if response and isinstance(response, dict):
        return response.get("number_of_specimens", 1), response.get("specimen_names", []), response.get(specimen_text, [])
    else:
        return 1, []


# Mock function to simulate the Mistral LLM processing locally
def mock_mistral_processing(prompt):
    # Add appropriate logic to handle the prompt and return a response
    # This is a placeholder function and should be replaced with actual logic
    if "gleason" in prompt.lower():
        return json.dumps({"gleason_score": "7", "gleason_pattern": "4+3"})
    elif "number of biopsy cores" in prompt.lower():
        return json.dumps({"num_cores": "3/10"})
    elif "percentage of cancer" in prompt.lower():
        return json.dumps({"percent_specimen": "30%"})
    elif "features from the pathology report" in prompt.lower():
        return json.dumps({"HGPIN": 0, "ASAP": 0, "ATYP": 0, "INF": 0, "ADC": 0, "PNI": 0, "Benign": 1})
    elif "comment section" in prompt.lower():
        return json.dumps({"comment": "No significant findings"})
    else:
        return json.dumps({"unknown": "unknown"})

def call_mistral(prompt):
    return mock_mistral_processing(prompt)

def extract_gleason(report_text):
    prompt = f"""
    Extract the Gleason score and pattern from the following pathology report:
    {report_text}
    Provide JSON output with keys 'gleason_score' and 'gleason_pattern'. If benign, set score to 'benign'. If unknown, return 'unknown'.
    """
    return json.loads(call_mistral(prompt))

def extract_cores(report_text):
    prompt = f"""
    Extract the number of biopsy cores with cancer from the following pathology report:
    {report_text}
    Provide JSON output with key 'num_cores' in the format 'X/Y' (cancer-containing cores/total cores). If unknown, return 'unknown'.
    """
    return json.loads(call_mistral(prompt))

def extract_percent(report_text):
    prompt = f"""
    Extract the percentage of cancer in the specimen from the following pathology report:
    {report_text}
    Provide JSON output with key 'percent_specimen'. If unknown, return 'unknown'.
    """
    return json.loads(call_mistral(prompt))

def extract_features(report_text):
    prompt = f"""
    Extract features from the pathology report:
    {report_text}
    Provide JSON output with keys: 
  - "HGPIN": 0 (absent) or 1 (present) - high grade prostatic intraepithelial neoplasia. If no data is found, set to 0.
  - "ASAP": 0 (absent) or 1 (present) - atypical small acinar proliferation. If no data is found, set to 0.
  - "ATYP": 0 (absent) or 1 (present) - any other mention of atypical glands besides ASAP specifically. If no data is found, set to 0.
  - "INF": 0 (absent) or 1 (present) - inflammation or prostatitis. If no data is found, set to 0.
  - "ADC": 0 (absent) or 1 (present) - adenocarcinoma described but no GS given. If no data is found, set to 0.
  - "PNI": 0 (absent) or 1 (present) - perineural invasion. If no data is found, set to 0.
    """
    return json.loads(call_mistral(prompt))

def extract_comment(report_text):
    prompt = f"""
    Extract the comment section from the following pathology report:
    {report_text}
    Provide JSON output with key 'comment'. If no comment, return 'none'.
    """
    return json.loads(call_mistral(prompt))

def process_report(report_text):
    structured_data = {
        "gleason": extract_gleason(report_text),
        "num_cores": extract_cores(report_text),
        "percent_specimen": extract_percent(report_text),
        "features": extract_features(report_text),
        "comment": extract_comment(report_text)
    }
    return structured_data

# Read CSV and process each pathology report
input_file = "input.csv"
output_file = "output.json"
data = []

with open(input_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        report_id = row["accession_number"]
        report_text = row["report"]
        specimens = report_text.split("  ")  # Assuming specimens are separated by double spaces or similar
        specimen_data = {"accession_number": report_id, "specimens": []}
        
        for specimen in specimens:
            specimen_data["specimens"].append(process_report(specimen))
        
        data.append(specimen_data)

# Write output JSON
with open(output_file, "w", encoding='utf-8') as jsonfile:
    json.dump(data, jsonfile, indent=4)

print("Processing complete. Output saved to", output_file)