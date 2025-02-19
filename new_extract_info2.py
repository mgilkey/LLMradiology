import json
import csv
import openai  # Assuming OpenAI's API is being used for Mistral; update if using another LLM

# Configure API Key (Ensure you set this up securely, do not hardcode in production)
openai.api_key = "your_api_key_here"

# Function to call Mistral LLM with a structured prompt
def call_mistral(prompt):
    response = openai.ChatCompletion.create(
        model="mistral-7b-instruct",
        messages=[{"role": "system", "content": "You are a medical NLP model designed to extract structured pathology data."},
                  {"role": "user", "content": prompt}],
        temperature=0.3  # Lower temperature for more deterministic responses
    )
    return response["choices"][0]["message"]["content"]

# Hierarchical prompt structure
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
    Provide JSON output with keys: 'HGPIN', 'ASAP', 'ATYP', 'INF', 'ADC', 'PNI', 'Benign', each having a value of 0 (absent) or 1 (present).
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
