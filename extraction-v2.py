import os
import glob
import json
import sys
from bs4 import BeautifulSoup
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o"
MAX_RETRIES = 3  # How many times to try to self-heal before giving up

def extract_tables_from_html(file_content):
    """Parses HTML and finds table blocks."""
    soup = BeautifulSoup(file_content, 'html.parser')
    tables = soup.find_all('table')
    significant_tables = []
    for table in tables:
        if len(table.get_text(strip=True)) > 100:
            significant_tables.append(str(table))
    return significant_tables

def validate_financial_logic(data):
    """
    Performs hard logic checks on the extracted JSON.
    Returns None if valid, or an error string if invalid.
    """
    # 1. Check if the LLM identified this as a Balance Sheet / Assets & Liabilities
    table_type = data.get("table_type", "").lower()
    
    if "balance sheet" in table_type or "assets and liabilities" in table_type:
        try:
            # Safely get values, defaulting to 0 if missing (but we want them to exist)
            assets = float(str(data.get("total_assets", 0)).replace(',', ''))
            liabilities = float(str(data.get("total_liabilities", 0)).replace(',', ''))
            net_assets = float(str(data.get("net_assets", 0)).replace(',', ''))
            
            # THE MATH CHECK: Assets should equal Liabilities + Net Assets
            # We use a small epsilon (0.1) for floating point rounding safety
            calculated_total = liabilities + net_assets
            difference = abs(assets - calculated_total)
            
            if difference > 1.0: # Allow $1 variance for rounding
                return f"Math Error: Total Assets ({assets}) does not equal Liabilities ({liabilities}) + Net Assets ({net_assets}). Difference is {difference}."
            
        except ValueError:
            return "Validation Error: Could not convert financial fields to numbers."
        except Exception as e:
            return f"Validation Error: {str(e)}"

    # we can add other validators here like the Schedule of Investments summation
    
    return None # No errors found

def process_table_agentic_loop(table_html, filename):
    """
    The Agentic Loop: Extracts, Validates, and Retries if necessary.
    """
    
    system_prompt = "You are a specialized financial auditor. You extract data precisely."
    base_prompt = f"""
    Analyze this HTML table from file: {filename}.
    
    Goal: Extract into JSON.
    
    CRITICAL SCHEMA RULES:
    1. Identify 'table_type'.
    2. If it is a Balance Sheet, you MUST extract keys: "total_assets", "total_liabilities", "net_assets".
    3. Remove commas from numbers.
    
    HTML:
    {table_html[:15000]}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": base_prompt}
    ]

    for attempt in range(MAX_RETRIES):
        print(f"      > Attempt {attempt + 1}...")
        
        # 1. Call LLM
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0
            )
            extracted_data = json.loads(response.choices[0].message.content)
            
            # 2. Deterministic Validation (Python checks the Math)
            error_message = validate_financial_logic(extracted_data)
            
            if error_message is None:
                extracted_data["validation_status"] = "passed"
                extracted_data["attempts_needed"] = attempt + 1
                return extracted_data
            
            # 3. FAILURE: Validation failed
            print(f"      ! Validation Failed: {error_message}")
            
            # 4. The "Self-Healing" Step: Feed error back to LLM
            # We append the assistant's wrong answer and our error message to history
            messages.append({"role": "assistant", "content": json.dumps(extracted_data)})
            messages.append({
                "role": "user", 
                "content": f"AUDIT FAILURE: {error_message}. Please re-examine the table image and fix your JSON output to satisfy the math check."
            })
            
        except Exception as e:
            print(f"      ! API Error: {e}")
            break

    # If we exhaust retries, mark as failed but return what we have
    print("      ! Exhausted retries. Returning last attempt.")
    return {"error": "Validation failed after max retries", "last_attempt": extracted_data}

def main(input_path):
    # This list will hold all extracted data from all files.
    # The script accumulates data in memory and writes it to a single JSON file at the very end.
    all_data = []
    
    # Ensure output directory exists
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    output_filename = os.path.join(output_dir, "final_output.json")
    if not os.path.exists(input_path):
        print("Input path not found.")
        return

    if os.path.isdir(input_path):
        files = glob.glob(os.path.join(input_path, "*.txt"))
    else:
        files = [input_path]
        base_name = os.path.basename(input_path)
        output_filename = os.path.join(output_dir, f"{base_name}_extracted.json")

    for file_path in files:
        print(f"Processing: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            tables = extract_tables_from_html(content)
            print(f"  - Found {len(tables)} tables.")

            file_record = {"filename": os.path.basename(file_path), "extracted_tables": []}

            for i, table in enumerate(tables):
                print(f"    - Processing Table {i+1} (Agentic Loop)...")
                
                # --- NEW FUNCTION CALL ---
                extracted_json = process_table_agentic_loop(table, os.path.basename(file_path))
                
                file_record["extracted_tables"].append(extracted_json)

            all_data.append(file_record)
            
        except Exception as e:
            print(f"Failed file {file_path}: {e}")

    # The script writes the entire accumulated list 'all_data' to the JSON file in one go.
    # It does not append incrementally; it overwrites/creates the file at the end of execution.
    with open(output_filename, 'w') as f:
        json.dump(all_data, f, indent=4)
    print(f"Done. Saved to {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 extract_tables.py <file_or_folder>")
        sys.exit(1)
    main(sys.argv[1])