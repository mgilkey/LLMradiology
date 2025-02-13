import os
import pandas as pd
import json
import subprocess
import re

# Define input and output file paths
INPUT_CSV_PATH = "input.csv"
OUTPUT_JSON_PATH = "output.json"
MODEL_NAME = "medllama2"

def clean_json_output(output):
    """Clean markdown formatting from LLM output."""
    output = output.strip()
    # If output is wrapped in triple backticks, remove them.
    if output.startswith("```"):
        lines = output.splitlines()
        # Remove the first line if it starts with ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove the last line if it ends with ```
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        output = "\n".join(lines).strip()
    return output

def get_number_of_specimens(report_text):
    """
    Extracts the total number of biopsy specimens from the pathology report and their names.
    """
    prompt = f"""
Extract the total number of biopsy specimens mentioned in the pathology report and list their names. The names may include the words (right, left, base, apex, mid, prostate, and biopsy).  It may also look like a numeric or alphabetic list. The name will end if a colon is present.
Respond only with a JSON object in the following format:

{{
  "number_of_specimens": <integer>,
  "specimen_names": keep only the first 4 words and nothing after a colon. Remove any commas between the words.
}}

Report:
{report_text}
"""
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL_NAME],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True
        )
        content = result.stdout.strip()
        print("\nRaw LLM Output for number of specimens:\n", content)
        content = clean_json_output(content)
        data = json.loads(content)
        return data.get("number_of_specimens", 1), data.get("specimen_names", [])
    except Exception as e:
        print(f"❌ Error extracting number of specimens: {e}")
        return 1, []

def extract_info_from_report(report_text, accession_number, number_of_specimens, line_number):
    """
    Sends a pathology report to the local LLM via Ollama and extracts structured data for each specimen.
    """
    prompt = f"""
You are a data scientist tasked with extracting clinical data from a pathology report. Use only the data provided for a specific specimen name and do not infer data.
Below is the pathology report:
{report_text}

Extract the following details for each of the {number_of_specimens} specimens:

- "gleason_score": A numeric value from 6 to 10. If missing but "gleason_pattern" is present, sum the two patterns. If the specimen is benign, set to "benign". If no data is found, set to "unknown".
- "gleason_pattern": A set of 2 numbers generally with a plus sign between them between 1-5 or a grade group with a value between 1 and 5. If the specimen is benign, set this value to "benign". If no data is found, set to "unknown".
- "num_cores": Look for the number of biopsy cores named in the {report_text} and report the fraction of cancer containing cores over the total cores (e.g., "3/7"). If no data is found, set to "unknown".
- "percent_specimen": Look for a number followed by (%) or the word (percentage) for the amount of cancer on the specimen (e.g., "<5%"). If a range is given, record the range. If no number with (%) or the word (percentage) is mentioned, set to "unknown".
- "features": A dictionary containing:
    - "HGPIN": 0 (absent) or 1 (present).
    - "ASAP": 0 (absent) or 1 (present).
    - "ATYP": 0 (absent) or 1 (present).
    - "INF": 0 (absent) or 1 (present).
    - "ADC": 0 (absent) or 1 (present).
    - "PNI": 0 (absent) or 1 (present).
    - "Benign": 0 (absent) or 1 (present).
- "comment": Include the text following the "specimen_names" after the colon and stop if you encounter a period.  Do not include follow up details by the doctor. If no data is found, set to "none".

Do not include any placeholder text from this prompt in your output. Only use data explicitly found in the report (or "unknown" if missing) and do not infer any data.

Respond **only** with a valid JSON object in the following format:

{{
  "specimens": [
    {{
      "gleason_score": "",
      "gleason_pattern": "",
      "num_cores": "",
      "percent_specimen": "",
      "features": {{
        "HGPIN": 0,
        "ASAP": 0,
        "ATYP": 0,
        "INF": 0,
        "ADC": 0,
        "PNI": 0,
        "Benign": 0
      }},
      "comment": ""
    }}
  ]
}}
"""
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL_NAME],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True
        )

        content = result.stdout.strip()
        print("\nRaw LLM Output for specimen details:\n", content)
        content = clean_json_output(content)
        data = json.loads(content)

        if isinstance(data, dict) and "specimens" in data and isinstance(data["specimens"], list):
            return data["specimens"]
        else:
            print("⚠️ JSON missing 'specimens' key or not in expected format.")
            return None
    except json.JSONDecodeError:
        print(f"❌ JSON Decode Error. Raw Output:\n{content}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def main():
    # Load input CSV 
    try:
        df_input = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"❌ Error: The input file '{INPUT_CSV_PATH}' was not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"❌ Error: The input file '{INPUT_CSV_PATH}' is empty.")
        return
    except pd.errors.ParserError:
        print(f"❌ Error: The input file '{INPUT_CSV_PATH}' could not be parsed correctly.")
        return
    
    # Validate columns
    if "report" not in df_input.columns or "accession_number" not in df_input.columns:
        print("❌ Error: The input CSV must contain columns named 'report' and 'accession_number'.")
        return

    extracted_data = []

    # Process each report
    for index, row in df_input.iterrows():
        report_text = row['report']
        accession_number = row['accession_number']
        print(f"\n🔍 Processing report {index+1} with accession number {accession_number}...")

        # Get number of specimens and their names
        number_of_specimens, specimen_names = get_number_of_specimens(report_text)

        # Extract pathology details
        extracted_specimens = extract_info_from_report(report_text, accession_number, number_of_specimens, index+1)

        if extracted_specimens is None:
            print(f"⚠️ No valid data extracted for report {index+1}. Adding default entry.")
            extracted_specimens = [{
                "study_id": accession_number,
                "specimen_name": "Specimen Unknown",
                "gleason_score": "unknown",
                "gleason_pattern": "unknown",
                "num_cores": "unknown",
                "percent_specimen": "unknown",
                "features": {"HGPIN": 0, "ASAP": 0, "ATYP": 0, "INF": 0, "ADC": 0, "PNI": 0, "Benign": 0},
                "comment": "unknown"
            }]
        else:
            for i, specimen in enumerate(extracted_specimens):
                specimen['study_id'] = accession_number
                specimen['specimen_name'] = specimen_names[i] if i < len(specimen_names) else "Unknown Specimen"
                specimen['comment'] = specimen['comment'][:200]

        extracted_data.extend(extracted_specimens)

    # Save output
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(extracted_data, f, indent=4)

    print(f"\n✅ Extraction complete. Data saved in: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
