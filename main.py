import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# 1. Load Environment Variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2. LLM Classification Function
def analyze_complaint_with_llm(text):
    if not text or not isinstance(text, str):
        return {"Aspect": "Unknown", "Frustration": "Unknown", "Request": "Unknown"}

    system_prompt = """
    You are an expert public grievance analyzer for the Indian Government.
    You will analyze citizens' complaints, which may be in English, Hindi, or 'Hinglish' (Hindi written in English).
    Note: The text contains redacted data marked with 'X' (e.g., X0X5, XOXPXNXAXIXN). Ignore these redactions and focus on the meaning.
    
    Extract the following three facets from the complaint:
    1. Aspect: The core category of the issue (e.g., Operations, Personnel, Payment, Infrastructure, Technical Issue, Legal/Harassment).
    2. Frustration: Is the user emotionally tense because a goal is blocked? Answer strictly as 'Frustrated' or 'Neutral'.
    3. Request: Is the user explicitly asking for an action to be taken (e.g., refund, fix, action against someone)? Answer strictly as 'Action_Required' or 'Statement'.
    
    Return the result strictly as a JSON object with keys: "Aspect", "Frustration", "Request".
    """

    try:
        response = client.chat.completions.create(
            model="gpt-5.4-nano", # GPT-5.4 Nano engine
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this complaint:\n\n{text}"}
            ],
            response_format={ "type": "json_object" },
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        # If a single API call fails (e.g., rate limit), it won't crash the whole script
        print(f"\n[Error] OpenAI API call failed: {e}")
        return {"Aspect": "Error", "Frustration": "Error", "Request": "Error"}

def assign_priority(frustration, request):
    if frustration == "Frustrated" and request == "Action_Required":
        return "INCIDENT (P1)"
    elif frustration == "Frustrated":
        return "ESCALATION (P2)"
    elif request == "Action_Required":
        return "ROUTINE (P3)"
    else:
        return "RECORD (P4)"

# Worker Function for Concurrent Execution
def process_single_record(record, index):
    """Processes a single record. Designed to be run by a worker thread."""
    reg_no = record.get('registration_no', f"UNK_{index}")
    subject = record.get('subject_content_text', '')
    
    analysis = analyze_complaint_with_llm(subject)
    priority = assign_priority(analysis.get("Frustration"), analysis.get("Request"))
    
    return {
        "ID": reg_no,
        "Aspect": analysis.get("Aspect", "Unknown"),
        "Frustration": analysis.get("Frustration", "Unknown"),
        "Request": analysis.get("Request", "Unknown"),
        "Priority": priority
    }

# 3. Data Processing Loop (With 70/30 Split)
def process_data(file_path, limit=None, output_dir=".", max_workers=5): 
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter for Railways (MORLY)
    transport_records = [r for r in data if r.get('org_code') == 'MORLY']
    
    total_records = len(transport_records)
    print(f"Total MORLY records found: {total_records}")
    
    # Calculate the 70% split index
    split_index = int(total_records * 0.70)
    
    # Split the dataset
    baseline_records = transport_records[:split_index]
    holdout_records = transport_records[split_index:]
    
    # Apply command-line limit to the baseline set if provided (for quick testing)
    if limit is not None:
        baseline_records = baseline_records[:limit]
        
    print(f"Using {len(baseline_records)} records (First 70%) for baseline processing...")
    print(f"Reserving {len(holdout_records)} records (Remaining 30%) for future smoothing tests.")
    
    # Save the 30% holdout set to a new JSON file for later use
    holdout_path = os.path.join(output_dir, "holdout_test_set.json")
    with open(holdout_path, 'w', encoding='utf-8') as f:
        json.dump(holdout_records, f, indent=4)
    print(f"Saved holdout dataset to: {holdout_path}")
    
    results = []
    
    # Execute in parallel using ThreadPoolExecutor on the 70% split
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_record = {
            executor.submit(process_single_record, record, i): i 
            for i, record in enumerate(baseline_records)
        }
        
        # Use tqdm to track completed futures as they finish
        for future in tqdm(as_completed(future_to_record), total=len(baseline_records), desc="Analyzing 70% Baseline", unit="ticket"):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"\n[Warning] A worker thread generated an exception: {exc}")
        
    df = pd.DataFrame(results)
    
    # Save the processed 70% inside the designated output directory
    csv_path = os.path.join(output_dir, "baseline_zero_shot_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved 70% baseline results to {csv_path}")
    return df

# 4. Graph Generation
def generate_graphs(df, output_dir=".", suffix="baseline"):
    # Set a bold, clean aesthetic
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Graph 1: Distribution of Aspects
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, y="Aspect", order=df['Aspect'].value_counts().index, color="#2b2d42")
    plt.title(f"Complaint Aspects Distribution ({suffix.capitalize()})", fontsize=14, fontweight='bold')
    plt.xlabel("Count")
    plt.ylabel("Aspect Category")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"aspect_distribution_{suffix}.png"), dpi=300)
    plt.close()

    # Graph 2: Frustration vs Request Matrix (Heatmap)
    plt.figure(figsize=(8, 6))
    matrix_data = pd.crosstab(df['Frustration'], df['Request'])
    sns.heatmap(matrix_data, annot=True, fmt="d", cmap="YlOrRd", cbar=False, annot_kws={"size": 16})
    plt.title(f"Frustration vs Request Signals ({suffix.capitalize()})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"signal_matrix_{suffix}.png"), dpi=300)
    plt.close()

    # Graph 3: Priority Level Breakdown
    plt.figure(figsize=(8, 5))
    priority_order = ["INCIDENT (P1)", "ESCALATION (P2)", "ROUTINE (P3)", "RECORD (P4)"]
    sns.countplot(data=df, x="Priority", order=priority_order, palette="dark:salmon_r")
    plt.title(f"Operational Priority Levels ({suffix.capitalize()})", fontsize=14, fontweight='bold')
    plt.ylabel("Number of Complaints")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"priority_levels_{suffix}.png"), dpi=300)
    plt.close()

    print(f"Generated and saved 3 graphs in {output_dir}.")

# --- Execute ---
if __name__ == "__main__":
    # Setup argparse to handle the --records flag
    parser = argparse.ArgumentParser(description="Run zero-shot baseline classification on complaints.")
    parser.add_argument("--records", type=int, default=None, 
                        help="Number of records to process from the 70% split for quick testing.")
    args = parser.parse_args()

    # Setup Directory
    output_directory = os.path.join("results", "baseline")
    os.makedirs(output_directory, exist_ok=True)
    print(f"Outputting files to: {output_directory}")

    # 1. Run the LLM baseline on the NEW synthetic dataset (70% split)
    df_baseline = process_data(
        file_path='no_pii_grievance_extension.json', 
        limit=args.records, 
        output_dir=output_directory,
        max_workers=5 # 5 Concurrent API calls
    )
    
    # 2. Generate the graphs
    if not df_baseline.empty:
        generate_graphs(df_baseline, output_dir=output_directory, suffix="baseline")