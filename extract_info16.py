import os
import pandas as pd
import json
import subprocess
import re

def fix_malformed_json(json_str):
    """
    Attempts to fix common JSON format issues:
    - Adds missing commas before new keys
    - Closes unclosed brackets
    - Removes invalid characters
    """
    try:
        json_str = json_str.strip()

        # Ensure brackets are balanced
        if json_str.count("{") > json_str.count("}"):
            json_str += "}"
        if json_str.count("[") > json_str.count("]"):
            json_str += "]"

        # Replace common errors (missing commas)
        json_str = re.sub(r'("\w+":\s?"[^"]+)"\s*("\w+":)', r'\1,\2', json_str)

        return json.loads(json_str)  # Try parsing the fixed JSON
    except json.JSONDecodeError as e:
        print(f"Error fixing JSON: {e}\nRaw output:\n{json_str}")
        return None

def get_number_of_specimens(report_text):
    """
    Sends a pathology report to the local LLM via Ollama and extracts the number of specimens.
    """
    prompt = f"""
You are a data scientist tasked with extracting clinical data from a pathology report. Please extract the number of specimens mentioned in the report. Respond with just the number.

### Pathology Report:
{report_text}
"""

    model_name = "medllama2"

    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True
        )

        content = result.stdout.strip()
        print("\nRaw LLM Output for number of specimens:\n", content)

        try:
            number_of_specimens = int(re.search(r'\d+', content).group())
            return number_of_specimens
        except (ValueError, AttributeError) as e:
            print(f"Error parsing number of specimens: {e}\nRaw Output:\n{content}")
            return 0
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 0

def extract_info_from_report(report_text, accession_number, number_of_specimens):
    """
    Sends a pathology report to the local LLM via Ollama and extracts structured data for each specimen.
    """
    prompt = f"""
You are a data scientist tasked with extracting clinical data from a pathology report. The goal is to create a structured clinical dataset in the form of a JSON object. Please extract the following details from the provided pathology report for each of the {number_of_specimens} specimens:

- "study_id": Use the provided accession number "{accession_number}". If no study_id is present, add "subject_<line number from input.csv>".
- "specimen_name": Specimen type (e.g., "Right Apex Biopsy"). Remove commas and rearrange the words as necessary to get the order to match the example. Ignore specimens from other tissues and diagnostic information.
- "gleason_score": Numeric value from 6 to 10. If missing but "gleason_pattern" is present, sum the two patterns. If the specimen is benign, set to "benign".
- "gleason_pattern": Score between 1-5 or "Grade Group". If benign, set to "benign".
- "num_cores": Fraction format (e.g., "3/7").
- "percent_specimen": Cancer percentage (e.g., "<5%"). If a range is given, record the range. If no percentage is mentioned, report "unknown".
- "features": Dictionary containing:
  - "HGPIN": 0 (absent) or 1 (present) - high grade prostatic intraepithelial neoplasia
  - "ASAP": 0 (absent) or 1 (present) - atypical small acinar proliferation
  - "ATYP": 0 (absent) or 1 (present) - any other mention of atypical glands besides ASAP specifically
  - "INF": 0 (absent) or 1 (present) - inflammation or prostatitis
  - "ADC": 0 (absent) or 1 (present) - adenocarcinoma described but no GS given
- "comment": Unique findings not included elsewhere. If benign, include "benign" in comments.

Ensure the response is a valid JSON object with a "specimens" list, where each entry follows the specified format.

### Pathology Report:
{report_text}
"""

    model_name = "llama3.2"

    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True
        )

        content = result.stdout.strip()
        print("\nRaw LLM Output for specimen details:\n", content)

        try:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                json_text = re.sub(r",\s*([\]}])", r"\1", json_text)
                data = json.loads(json_text)

                if "specimens" in data and isinstance(data["specimens"], list):
                    return data["specimens"]
                else:
                    print("Error: JSON output did not contain expected 'specimens' list.")
                    return None
            else:
                print("Error: No valid JSON found.")
                return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}\nRaw Output:\n{content}")
            return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Define file paths for input and output CSVs
input_csv_path = "input.csv"
output_csv_path = "output.csv"

try:
    df_input = pd.read_csv(input_csv_path)
except FileNotFoundError:
    print(f"Error: The input file '{input_csv_path}' was not found.")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: The input file '{input_csv_path}' is empty.")
    exit(1)
except pd.errors.ParserError:
    print(f"Error: The input file '{input_csv_path}' could not be parsed correctly.")
    exit(1)

if "report" not in df_input.columns or "accession_number" not in df_input.columns:
    print("Error: The input CSV must contain columns named 'report' and 'accession_number'.")
    exit(1)

extracted_data = []

for index, row in df_input.iterrows():
    report_text = row['report']
    accession_number = row['accession_number']
    line_number = index + 1
    print(f"\nProcessing report {line_number} with accession number {accession_number}...")

    number_of_specimens = get_number_of_specimens(report_text)
    if number_of_specimens == 0:
        print(f"⚠️ Warning: No specimens found for report {line_number}.")
        number_of_specimens = 1  # Default to 1 to ensure at least one set of columns is created

    extracted_specimens = extract_info_from_report(report_text, accession_number, number_of_specimens)

    if extracted_specimens is None:
        print(f"⚠️ Warning: No valid data extracted for report {line_number}.")
        extracted_specimens = [{
            "study_id": accession_number if accession_number else f"subject_{line_number}",
            "specimen_name": "Specimen Unknown",
            "gleason_score": "No GS",
            "gleason_pattern": "No GP",
            "num_cores": "No #C",
            "percent_specimen": "unknown",
            "features": {"HGPIN": 0, "ASAP": 0, "ATYP": 0, "INF": 0, "ADC": 0},
            "comment": "No comment"
        }]

    for specimen in extracted_specimens:
        if 'benign' in report_text.lower():
            specimen['gleason_score'] = 'benign'
            specimen['gleason_pattern'] = 'benign'
            specimen['comment'] = 'benign'

    extracted_data.append(extracted_specimens)

# Flatten extracted data and create dynamic columns for multiple specimens
flattened_data = []
max_specimens = max(len(report) for report in extracted_data)
columns = ["StudyID"]

for i in range(max_specimens):
    columns += [f"Specimen_{i+1}", f"GS_{i+1}", f"GP_{i+1}", f"#C_{i+1}", f"%Spec_{i+1}", f"HGPIN_{i+1}", f"ASAP_{i+1}", f"ATYP_{i+1}", f"INF_{i+1}", f"ADC_{i+1}", f"Comment_{i+1}"]

for report_data in extracted_data:
    row = [report_data[0]["study_id"]]  # StudyID
    for i in range(max_specimens):
        if i < len(report_data):
            specimen = report_data[i]
            row += [
                specimen.get("specimen_name", ""),
                specimen.get("gleason_score", ""),
                specimen.get("gleason_pattern", ""),
                specimen.get("num_cores", ""),
                specimen.get("percent_specimen", ""),
                specimen.get("features", {}).get("HGPIN", 0),
                specimen.get("features", {}).get("ASAP", 0),
                specimen.get("features", {}).get("ATYP", 0),
                specimen.get("features", {}).get("INF", 0),
                specimen.get("features", {}).get("ADC", 0),
                specimen.get("comment", "")
            ]
        else:
            row += [""] * 11
    flattened_data.append(row)

df_output = pd.DataFrame(flattened_data, columns=columns)

try:
    df_output.to_csv(output_csv_path, index=False)
    print(f"\n✅ Extraction complete. New CSV saved at: {output_csv_path}")
except Exception as e:
    print(f"Error writing to output CSV: {e}")