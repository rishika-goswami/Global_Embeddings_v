import os
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# 1. Load Environment Variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
OUTPUT_FILE = "no_pii_grievance_extension.json"
BATCH_SIZE = 25 
TOTAL_RECORDS_NEEDED = 20000 
MODEL_ENGINE = "gpt-4o-mini"
MAX_WORKERS = 5 # Number of parallel API calls (Keep at 5 to avoid OpenAI Rate Limits)

def generate_batch(batch_size):
    """Calls OpenAI to generate a batch of records."""
    system_prompt = f"""
    You are an expert data synthesizer for the Indian Ministry of Railways (MORLY).
    Generate exactly {batch_size} unique grievance records in JSON format.
    Output MUST be a valid JSON array of objects. Do not use markdown blocks like ```json. Just output the raw array.

    CRITICAL REQUIREMENTS FOR THE DATA:
    1. "_id" and "registration_no": Must be in format "MORLY/E/2023/XXXXXXX" (Random 7 digit number).
    2. "org_code": Must ALWAYS be "MORLY".
    3. "subject_content_text": The citizen's complaint. Vary the length significantly (10 to 100 words). Use a mix of English and 'Hinglish' (phonetic Hindi). Topics: Operations, Payment, Personnel, Infrastructure.
    4. "remarks_text": The official response. Vary the length significantly. Use a mix of formal English and formal Hindi written in English (Hinglish). 
    5. PII Masking: If using phone numbers, account numbers, or PNRs, mask them using 'X' (e.g., X0X5X, X-X-X-X).
    6. "sex": "M" or "F".
    7. "v7_target": "Yes" or "No".
    8. Dates: Must use the format `{{"$date": "YYYY-MM-DDTHH:MM:SS.000Z"}}`.

    Make sure EVERY object in the array has these exact keys:
    _id, CategoryV7, DiaryDate, UserCode, closing_date, dist_name, org_code, pincode, recvd_date, registration_no, remarks_text, resolution_date, sex, state, subject_content_text, v7_target
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_ENGINE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate the JSON array now."}
            ],
            temperature=0.8, 
        )
        
        raw_output = response.choices[0].message.content.strip()
        
        # Clean up markdown if the model hallucinates it
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:]
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]
            
        return json.loads(raw_output)
    
    except json.JSONDecodeError as e:
        # Silently fail and return empty array so the worker thread doesn't crash the script
        return []
    except Exception as e:
        # Handle Rate limits (429) or connection drops
        if "rate_limit" in str(e).lower() or "429" in str(e):
            time.sleep(5) # Brief pause if hitting rate limits
        return []

def main():
    print(f"Starting Generation: Target = {TOTAL_RECORDS_NEEDED} records using {MODEL_ENGINE} with {MAX_WORKERS} workers.")
    
    all_records = []
    
    # Check if we are resuming a previous run
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                all_records = json.load(f)
            print(f"Found existing file with {len(all_records)} records. Resuming...")
        except json.JSONDecodeError:
            print("Existing file is corrupted. Starting fresh.")
            all_records = []

    records_to_generate = TOTAL_RECORDS_NEEDED - len(all_records)
    
    if records_to_generate <= 0:
        print("Target already reached.")
        return

    # Create a list of task sizes (e.g., [25, 25, 25...])
    batches = [BATCH_SIZE] * (records_to_generate // BATCH_SIZE)
    if records_to_generate % BATCH_SIZE != 0:
        batches.append(records_to_generate % BATCH_SIZE)

    # Execute in parallel
    with tqdm(total=TOTAL_RECORDS_NEEDED, initial=len(all_records), desc="Generating Data", unit="rec") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all batches to the worker pool
            future_to_batch = {executor.submit(generate_batch, b_size): b_size for b_size in batches}
            
            # As each API call finishes, process the result
            for future in as_completed(future_to_batch):
                new_records = future.result()
                
                if new_records:
                    # Enforce the cap in case of slight over-generation
                    remaining_slots = TOTAL_RECORDS_NEEDED - len(all_records)
                    records_to_add = new_records[:remaining_slots]
                    
                    all_records.extend(records_to_add)
                    
                    # File IO is safe here because `as_completed` yields on the main thread
                    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                        json.dump(all_records, f, indent=4)
                    
                    pbar.update(len(records_to_add))
                
                if len(all_records) >= TOTAL_RECORDS_NEEDED:
                    # Cancel any pending futures if we hit the target early
                    for f in future_to_batch:
                        f.cancel()
                    break

    print(f"\nFinished! Total records generated and saved to {OUTPUT_FILE}: {len(all_records)}")

if __name__ == "__main__":
    main()