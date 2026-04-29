import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# 1. Load Environment Variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2. Mathematical Functions
def calculate_global_priors(df, column):
    """Calculates P_global(c) for a given categorical column."""
    return df[column].value_counts(normalize=True).to_dict()

def apply_dirichlet_smoothing(local_label, global_priors, mu):
    """
    Applies Dirichlet smoothing to a single hard-label observation.
    P_smooth = (Local_Count + mu * Global_Prior) / (1 + mu)
    """
    smoothed_probs = {}
    
    for label, prior in global_priors.items():
        # Local count is 1 if it matches the LLM's choice, else 0
        local_count = 1.0 if label == local_label else 0.0
        
        # Dirichlet calculation
        smoothed_prob = (local_count + mu * prior) / (1.0 + mu)
        smoothed_probs[label] = smoothed_prob
        
    # Return the label with the highest smoothed probability (argmax)
    return max(smoothed_probs, key=smoothed_probs.get)

def assign_priority(frustration, request):
    """Re-calculates priority based on smoothed signals."""
    if frustration == "Frustrated" and request == "Action_Required":
        return "INCIDENT (P1)"
    elif frustration == "Frustrated":
        return "ESCALATION (P2)"
    elif request == "Action_Required":
        return "ROUTINE (P3)"
    else:
        return "RECORD (P4)"

# 3. LLM Functions for the Holdout Set
def analyze_complaint_with_llm(text):
    if not text or not isinstance(text, str):
        return {"Aspect": "Unknown", "Frustration": "Unknown", "Request": "Unknown"}

    system_prompt = """
    You are an expert public grievance analyzer for the Indian Government.
    You will analyze citizens' complaints, which may be in English, Hindi, or 'Hinglish'.
    Note: The text contains redacted data marked with 'X'. Ignore these redactions.
    
    Extract the following three facets from the complaint:
    1. Aspect: Operations, Personnel, Payment, Infrastructure, Technical Issue, Legal/Harassment.
    2. Frustration: Answer strictly as 'Frustrated' or 'Neutral'.
    3. Request: Answer strictly as 'Action_Required' or 'Statement'.
    
    Return the result strictly as a JSON object with keys: "Aspect", "Frustration", "Request".
    """

    try:
        response = client.chat.completions.create(
            model="gpt-5.4-nano", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this complaint:\n\n{text}"}
            ],
            response_format={ "type": "json_object" },
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"Aspect": "Error", "Frustration": "Error", "Request": "Error"}

def process_single_record(record, index):
    reg_no = record.get('registration_no', f"UNK_{index}")
    subject = record.get('subject_content_text', '')
    
    analysis = analyze_complaint_with_llm(subject)
    
    return {
        "ID": reg_no,
        "Raw_Aspect": analysis.get("Aspect", "Unknown"),
        "Raw_Frustration": analysis.get("Frustration", "Unknown"),
        "Raw_Request": analysis.get("Request", "Unknown"),
    }

def fetch_or_process_holdout(holdout_json_path, raw_csv_path, max_workers=5):
    """Processes the holdout JSON through the LLM, or loads the CSV if already processed."""
    if os.path.exists(raw_csv_path):
        print(f"Loading pre-processed raw holdout data from {raw_csv_path} (Saves API calls!)")
        return pd.read_csv(raw_csv_path)
    
    print(f"Raw holdout data not found. Processing {holdout_json_path} via API...")
    with open(holdout_json_path, 'r', encoding='utf-8') as f:
        holdout_records = json.load(f)
        
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_record = {
            executor.submit(process_single_record, record, i): i 
            for i, record in enumerate(holdout_records)
        }
        
        for future in tqdm(as_completed(future_to_record), total=len(holdout_records), desc="Extracting Holdout Zero-Shot", unit="ticket"):
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"[Warning] Worker exception: {exc}")
                
    df_raw = pd.DataFrame(results)
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"Saved raw holdout predictions to {raw_csv_path}")
    return df_raw

# 4. Graph Generation
def generate_graphs(df, mu, output_dir):
    sns.set_theme(style="whitegrid", palette="muted")
    mu_str = str(mu)
    
    # 1. Aspect Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y="Aspect", order=df['Aspect'].value_counts().index, color="#2b2d42")
    plt.title(f"Holdout Aspects (Dirichlet Smoothed, μ={mu_str})", fontsize=14, fontweight='bold')
    plt.xlabel("Count")
    plt.ylabel("Aspect Category")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"d_{mu_str}_aspect_distribution.png"), dpi=300)
    plt.close()

    # 2. Frustration vs Request Matrix
    plt.figure(figsize=(8, 6))
    matrix_data = pd.crosstab(df['Frustration'], df['Request'])
    sns.heatmap(matrix_data, annot=True, fmt="d", cmap="YlOrRd", cbar=False, annot_kws={"size": 16})
    plt.title(f"Holdout Frustration vs Request (Dirichlet Smoothed, μ={mu_str})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"d_{mu_str}_signal_matrix.png"), dpi=300)
    plt.close()

    # 3. Priority Level Breakdown
    plt.figure(figsize=(8, 5))
    priority_order = ["INCIDENT (P1)", "ESCALATION (P2)", "ROUTINE (P3)", "RECORD (P4)"]
    sns.countplot(data=df, x="Priority", order=priority_order, palette="dark:salmon_r")
    plt.title(f"Holdout Priority Levels (Dirichlet Smoothed, μ={mu_str})", fontsize=14, fontweight='bold')
    plt.ylabel("Number of Complaints")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"d_{mu_str}_priority_levels.png"), dpi=300)
    plt.close()
    
    print(f"Generated 3 graphs in {output_dir}")

# --- Execute ---
if __name__ == "__main__":
    # 1. Load the 70% Baseline data to calculate Priors
    baseline_path = os.path.join("results", "baseline", "baseline_zero_shot_results.csv")
    try:
        df_baseline = pd.read_csv(baseline_path)
    except FileNotFoundError:
        print(f"Error: Baseline CSV not found at {baseline_path}. Run main.py first.")
        exit()

    print("Calculating Global Priors from the 70% baseline set...")
    aspect_priors = calculate_global_priors(df_baseline, 'Aspect')
    frust_priors = calculate_global_priors(df_baseline, 'Frustration')
    req_priors = calculate_global_priors(df_baseline, 'Request')

    # 2. Fetch or Generate the Raw Holdout Predictions
    holdout_json_path = os.path.join("results", "baseline", "holdout_test_set.json")
    raw_holdout_csv_path = os.path.join("results", "baseline", "holdout_zero_shot_raw.csv")
    
    if not os.path.exists(holdout_json_path):
        print(f"Error: {holdout_json_path} not found. Run main.py to split the dataset.")
        exit()
        
    df_holdout = fetch_or_process_holdout(holdout_json_path, raw_holdout_csv_path)

    # 3. Apply Dirichlet Smoothing
    MU = 1.5 
    print(f"\nApplying Dirichlet Smoothing to Holdout Set with μ = {MU}")
    
    output_dir = os.path.join("results", "dirichlet", str(MU))
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply smoothing using the baseline priors against the raw holdout labels
    df_holdout['Aspect'] = df_holdout['Raw_Aspect'].apply(lambda x: apply_dirichlet_smoothing(x, aspect_priors, MU))
    df_holdout['Frustration'] = df_holdout['Raw_Frustration'].apply(lambda x: apply_dirichlet_smoothing(x, frust_priors, MU))
    df_holdout['Request'] = df_holdout['Raw_Request'].apply(lambda x: apply_dirichlet_smoothing(x, req_priors, MU))

    # Re-evaluate Priority based on smoothed labels
    df_holdout['Priority'] = df_holdout.apply(lambda row: assign_priority(row['Frustration'], row['Request']), axis=1)

    # 4. Save CSV and Graphs
    csv_path = os.path.join(output_dir, f"d_{MU}_holdout_smoothed_results.csv")
    df_holdout.to_csv(csv_path, index=False)
    print(f"Saved smoothed holdout data to {csv_path}")
    
    generate_graphs(df_holdout, MU, output_dir)