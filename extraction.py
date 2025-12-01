import os
import glob
import json
import sys  # <--- Added sys to read command line arguments
from bs4 import BeautifulSoup
from openai import OpenAI

# --- CONFIGURATION ---
# Set your API Key here or in your environment variables
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY_HERE"
client = OpenAI()

# Select a model with a large context window (4o is efficient and smart)
MODEL = "gpt-4o" 

def extract_tables_from_html(file_content):
    """
    Parses HTML content and returns a list of stringified <table> blocks.
    """
    # Use lxml if installed for speed, otherwise html.parser
    soup = BeautifulSoup(file_content, 'html.parser')
    tables = soup.find_all('table')
    
    # Filter out tiny tables (often used for formatting/spacing in old HTML)
    significant_tables = []
    for table in tables:
        # Heuristic: Table must have > 100 chars of text to be considered data
        if len(table.get_text(strip=True)) > 100:
            significant_tables.append(str(table))
            
    return significant_tables

def process_table_with_llm(table_html, filename):
    """
    Sends the HTML table to OpenAI to convert to JSON.
    """
    prompt = f"""
    You are a financial data extraction engine. 
    Analyze the following HTML table from an SEC N-CSR filing ({filename}).
    
    Your Goal: Extract the data into a structured JSON format.
    
    Rules:
    1. Identify the 'table_type' (e.g., "Schedule of Investments", "Balance Sheet", "Operations", or "Other").
    2. Extract the headers and the rows accurately.
    3. Return ONLY a JSON object.
    
    HTML Table:
    {table_html[:15000]} # Truncating to 15k chars for safety in this simple script version
    """

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            # Enforce JSON mode
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error processing table chunk: {e}")
        return {"error": str(e), "table_snippet": table_html[:100]}

def main(input_path):
    # This list will hold all extracted data from all files.
    # The script accumulates data in memory and writes it to a single JSON file at the very end.
    all_data = []
    
    # Ensure output directory exists
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine output filename based on input
    output_filename = os.path.join(output_dir, "final_output.json")

    # Check if input exists
    if not os.path.exists(input_path):
        print(f"Error: The path '{input_path}' does not exist.")
        return

    # Determine if input is a single file or directory
    if os.path.isdir(input_path):
        files = glob.glob(os.path.join(input_path, "*.txt"))
        print(f"Directory detected. Found {len(files)} .txt files.")
    else:
        files = [input_path]
        # Make output name specific to input file
        base_name = os.path.basename(input_path)
        output_filename = os.path.join(output_dir, f"{base_name}_extracted.json")

    for file_path in files:
        print(f"Processing: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 1. Chunking Strategy: Isolate Tables
            tables = extract_tables_from_html(content)
            print(f"  - Found {len(tables)} significant tables.")

            file_record = {
                "filename": os.path.basename(file_path),
                "extracted_tables": []
            }

            # 2. Loop through chunks
            for i, table in enumerate(tables):
                print(f"    - Extracting Table {i+1}/{len(tables)}...")
                
                # Call LLM
                extracted_json = process_table_with_llm(table, os.path.basename(file_path))
                
                file_record["extracted_tables"].append(extracted_json)

            all_data.append(file_record)
            
        except Exception as e:
            print(f"Failed to process file {file_path}: {e}")

    # 3. Save Final Output
    # The script writes the entire accumulated list 'all_data' to the JSON file in one go.
    # It does not append incrementally; it overwrites/creates the file at the end of execution.
    with open(output_filename, 'w') as f:
        json.dump(all_data, f, indent=4)
    
    print(f"Extraction complete. Data saved to {output_filename}")

if __name__ == "__main__":
    # Check if user provided an argument
    if len(sys.argv) < 2:
        print("Usage: python3 extract_tables.py <file_or_folder_path>")
        print("Example: python3 extract_tables.py my_filing.txt")
        sys.exit(1)

    # Get the filename/folder from the command line argument
    input_arg = sys.argv[1]
    
    main(input_arg)