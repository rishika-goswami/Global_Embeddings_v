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

# ==========================================
# 1. SETUP & ENVIRONMENT
# ==========================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = os.path.join("results", "variable_confidence")
os.makedirs(BASE_DIR, exist_ok=True)

# ==========================================
# 2. MATHEMATICAL CORE (SOFT LABELS)
# ==========================================
def calculate_global_priors(df, column):
    return df[column].value_counts(normalize=True).to_dict()

def get_unique_classes(df, column):
    return df[column].unique().tolist()

def get_soft_local_probs(local_label, confidence, classes):
    """Distributes confidence to the winner and remainder to the losers."""
    probs = {}
    n_classes = len(classes)
    remainder = (1.0 - confidence) / max(1, (n_classes - 1)) if n_classes > 1 else 0.0
    
    for c in classes:
        probs[c] = confidence if c == local_label else remainder
    return probs

def apply_jm_soft(local_label, confidence, classes, global_priors, jm_lambda):
    local_probs = get_soft_local_probs(local_label, confidence, classes)
    smoothed_probs = {}
    for label in classes:
        prior = global_priors.get(label, 0.0)
        smoothed_probs[label] = (1.0 - jm_lambda) * local_probs[label] + jm_lambda * prior
    return max(smoothed_probs, key=smoothed_probs.get)

def apply_dirichlet_soft(local_label, confidence, classes, global_priors, mu):
    local_probs = get_soft_local_probs(local_label, confidence, classes)
    smoothed_probs = {}
    for label in classes:
        prior = global_priors.get(label, 0.0)
        # N=1 because it's a single ticket observation
        smoothed_probs[label] = (local_probs[label] + mu * prior) / (1.0 + mu)
    return max(smoothed_probs, key=smoothed_probs.get)

def apply_bayesian_soft(local_label, confidence, classes, b_constant):
    local_probs = get_soft_local_probs(local_label, confidence, classes)
    smoothed_probs = {}
    V = len(classes)
    for label in classes:
        smoothed_probs[label] = (local_probs[label] + b_constant) / (1.0 + b_constant * V)
    return max(smoothed_probs, key=smoothed_probs.get)

def assign_priority(frustration, request):
    if frustration == "Frustrated" and request == "Action_Required":
        return "INCIDENT (P1)"
    elif frustration == "Frustrated":
        return "ESCALATION (P2)"
    elif request == "Action_Required":
        return "ROUTINE (P3)"
    else:
        return "RECORD (P4)"

# ==========================================
# 3. LLM EXTRACTION & CATCHING
# ==========================================
def analyze_complaint_with_llm(text):
    if not text or not isinstance(text, str):
        return {"Aspect": "Unknown", "Frustration": "Unknown", "Request": "Unknown", "Confidence": 1.0}

    system_prompt = """
    Analyze the public grievance and return a JSON object with:
    1. Aspect: Operations, Personnel, Payment, Infrastructure, Technical Issue, Legal/Harassment.
    2. Frustration: 'Frustrated' or 'Neutral'.
    3. Request: 'Action_Required' or 'Statement'.
    4. Confidence: A float between 0.5 and 1.0 representing how sure you are of these labels.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-5.4-nano", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze:\n\n{text}"}
            ],
            response_format={ "type": "json_object" },
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"Aspect": "Error", "Frustration": "Error", "Request": "Error", "Confidence": 1.0}

def process_single_record(record, index):
    reg_no = record.get('registration_no', f"UNK_{index}")
    subject = record.get('subject_content_text', '')
    analysis = analyze_complaint_with_llm(subject)
    
    # Ensure confidence is a float
    try:
        conf = float(analysis.get("Confidence", 1.0))
    except (ValueError, TypeError):
        conf = 1.0

    return {
        "ID": reg_no,
        "Raw_Aspect": analysis.get("Aspect", "Unknown"),
        "Raw_Frustration": analysis.get("Frustration", "Unknown"),
        "Raw_Request": analysis.get("Request", "Unknown"),
        "Confidence": conf
    }

def fetch_or_process_holdout(holdout_json_path, raw_csv_path, max_workers=5):
    if os.path.exists(raw_csv_path):
        print(f"Loading cached raw holdout data WITH confidence from {raw_csv_path}")
        return pd.read_csv(raw_csv_path)
    
    print(f"Cache not found. Processing via API to extract Soft Labels...")
    with open(holdout_json_path, 'r', encoding='utf-8') as f:
        holdout_records = json.load(f)
        
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_record = {executor.submit(process_single_record, rec, i): i for i, rec in enumerate(holdout_records)}
        for future in tqdm(as_completed(future_to_record), total=len(holdout_records), desc="Extracting API"):
            try:
                results.append(future.result())
            except Exception as exc:
                pass
                
    df_raw = pd.DataFrame(results)
    df_raw.to_csv(raw_csv_path, index=False)
    return df_raw

# ==========================================
# 4. GRAPH GENERATION
# ==========================================
def generate_graphs(df, method_name, param_val, output_dir, prefix):
    sns.set_theme(style="whitegrid", palette="muted")
    param_str = str(param_val)
    title_suffix = f"({method_name}, param={param_str})"
    
    # 1. Aspect
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y="Aspect", order=df['Aspect'].value_counts().index, color="#2b2d42")
    plt.title(f"Aspect Distribution {title_suffix}", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_{param_str}_aspect_distribution.png"), dpi=300)
    plt.close()

    # 2. Matrix
    plt.figure(figsize=(8, 6))
    matrix_data = pd.crosstab(df['Frustration'], df['Request'])
    sns.heatmap(matrix_data, annot=True, fmt="d", cmap="YlOrRd", cbar=False, annot_kws={"size": 16})
    plt.title(f"Frustration vs Request {title_suffix}", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_{param_str}_signal_matrix.png"), dpi=300)
    plt.close()

    # 3. Priority
    plt.figure(figsize=(8, 5))
    priority_order = ["INCIDENT (P1)", "ESCALATION (P2)", "ROUTINE (P3)", "RECORD (P4)"]
    sns.countplot(data=df, x="Priority", order=priority_order, palette="dark:salmon_r")
    plt.title(f"Priority Levels {title_suffix}", fontweight='bold')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_{param_str}_priority_levels.png"), dpi=300)
    plt.close()

# ==========================================
# 5. EXCEL COMPILATION
# ==========================================
def extract_matrix_data(df, algo_name, param_name):
    counts = df.groupby(['Frustration', 'Request']).size().to_dict()
    return {
        'Algorithm': algo_name,
        'Parameter': param_name,
        'Frustrated + Action_Required': counts.get(('Frustrated', 'Action_Required'), 0),
        'Frustrated + Statement': counts.get(('Frustrated', 'Statement'), 0),
        'Neutral + Action_Required': counts.get(('Neutral', 'Action_Required'), 0),
        'Neutral + Statement': counts.get(('Neutral', 'Statement'), 0)
    }

def compile_all_results(summary_data):
    # Priority Excel
    priority_df = pd.DataFrame([s['priority'] for s in summary_data])
    exp_p = ['Algorithm', 'Parameter', 'INCIDENT (P1)', 'ESCALATION (P2)', 'ROUTINE (P3)', 'RECORD (P4)']
    for c in exp_p: 
        if c not in priority_df: priority_df[c] = 0
    priority_df = priority_df[exp_p].fillna(0)

    p_path = os.path.join(BASE_DIR, "variable_compiled_priority_bucket.xlsx")
    with pd.ExcelWriter(p_path, engine='openpyxl') as writer:
        priority_df.to_excel(writer, index=False)
        ws = writer.sheets['Sheet1']
        for idx, col in enumerate(priority_df.columns):
            ws.column_dimensions[chr(65 + idx)].width = max(priority_df[col].astype(str).map(len).max(), len(col)) + 2

    # Matrix Excel
    matrix_df = pd.DataFrame([s['matrix'] for s in summary_data])
    exp_m = ['Algorithm', 'Parameter', 'Frustrated + Action_Required', 'Frustrated + Statement', 'Neutral + Action_Required', 'Neutral + Statement']
    for c in exp_m: 
        if c not in matrix_df: matrix_df[c] = 0
    matrix_df = matrix_df[exp_m].fillna(0)

    m_path = os.path.join(BASE_DIR, "variable_compiled_signal_matrix.xlsx")
    with pd.ExcelWriter(m_path, engine='openpyxl') as writer:
        matrix_df.to_excel(writer, index=False)
        ws = writer.sheets['Sheet1']
        for idx, col in enumerate(matrix_df.columns):
            ws.column_dimensions[chr(65 + idx)].width = max(matrix_df[col].astype(str).map(len).max(), len(col)) + 2

    print(f"\n✅ Excel Reports Generated in {BASE_DIR}")

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("--- Starting Variable Confidence Pipeline ---")

    # 1. Load Baseline Priors & Classes
    baseline_path = os.path.join("results", "baseline", "baseline_zero_shot_results.csv")
    df_base = pd.read_csv(baseline_path)
    
    aspect_priors = calculate_global_priors(df_base, 'Aspect')
    frust_priors = calculate_global_priors(df_base, 'Frustration')
    req_priors = calculate_global_priors(df_base, 'Request')
    
    aspect_classes = get_unique_classes(df_base, 'Aspect')
    frust_classes = get_unique_classes(df_base, 'Frustration')
    req_classes = get_unique_classes(df_base, 'Request')

    # 2. Get Soft Label Holdout Data
    holdout_json = os.path.join("results", "baseline", "holdout_test_set.json")
    raw_conf_csv = os.path.join(BASE_DIR, "raw_holdout_with_confidence.csv")
    df_raw = fetch_or_process_holdout(holdout_json, raw_conf_csv)

    # Prepare data collector for Excel
    summary_data = []

    # Capture Raw Baseline for Excel
    df_raw['Priority'] = df_raw.apply(lambda r: assign_priority(r['Raw_Frustration'], r['Raw_Request']), axis=1)
    
    p_base = df_raw['Priority'].value_counts().to_dict()
    p_base.update({'Algorithm': 'Zero-Shot Baseline', 'Parameter': 'None'})
    
    # Temporarily rename columns for extract_matrix_data helper
    df_raw_temp = df_raw.rename(columns={'Raw_Frustration': 'Frustration', 'Raw_Request': 'Request'})
    m_base = extract_matrix_data(df_raw_temp, 'Zero-Shot Baseline', 'None')
    
    summary_data.append({'priority': p_base, 'matrix': m_base})

    # 3. Define Hyperparameters
    experiments = [
        {"name": "Jelinek-Mercer", "prefix": "jm", "folder": "jelinek_mercer", "func": apply_jm_soft, "params": [0.2, 0.3, 0.4, 0.6, 0.8], "needs_priors": True},
        {"name": "Bayesian", "prefix": "bayesian", "folder": "bayesian", "func": apply_bayesian_soft, "params": [0.5, 1.0], "needs_priors": False},
        {"name": "Dirichlet", "prefix": "dirichlet", "folder": "dirichlet", "func": apply_dirichlet_soft, "params": [0.5, 1.0, 1.5], "needs_priors": True}
    ]

    # 4. Run the Pipeline
    for exp in experiments:
        exp_dir = os.path.join(BASE_DIR, exp["folder"])
        os.makedirs(exp_dir, exist_ok=True)
        print(f"\nProcessing {exp['name']} Smoothing...")

        for val in tqdm(exp["params"], desc=exp['name']):
            df_smooth = df_raw.copy()
            
            # Apply Smoothing
            if exp["needs_priors"]:
                df_smooth['Aspect'] = df_smooth.apply(lambda r: exp["func"](r['Raw_Aspect'], r['Confidence'], aspect_classes, aspect_priors, val), axis=1)
                df_smooth['Frustration'] = df_smooth.apply(lambda r: exp["func"](r['Raw_Frustration'], r['Confidence'], frust_classes, frust_priors, val), axis=1)
                df_smooth['Request'] = df_smooth.apply(lambda r: exp["func"](r['Raw_Request'], r['Confidence'], req_classes, req_priors, val), axis=1)
            else:
                df_smooth['Aspect'] = df_smooth.apply(lambda r: exp["func"](r['Raw_Aspect'], r['Confidence'], aspect_classes, val), axis=1)
                df_smooth['Frustration'] = df_smooth.apply(lambda r: exp["func"](r['Raw_Frustration'], r['Confidence'], frust_classes, val), axis=1)
                df_smooth['Request'] = df_smooth.apply(lambda r: exp["func"](r['Raw_Request'], r['Confidence'], req_classes, val), axis=1)

            # Assign Priority
            df_smooth['Priority'] = df_smooth.apply(lambda r: assign_priority(r['Frustration'], r['Request']), axis=1)

            # Save CSV
            csv_path = os.path.join(exp_dir, f"{exp['prefix']}_{val}_smoothed_results.csv")
            df_smooth.to_csv(csv_path, index=False)

            # Generate Graphs
            generate_graphs(df_smooth, exp["name"], val, exp_dir, exp["prefix"])

            # Collect for Excel
            p_counts = df_smooth['Priority'].value_counts().to_dict()
            p_counts.update({'Algorithm': exp['name'], 'Parameter': str(val)})
            
            m_counts = extract_matrix_data(df_smooth, exp['name'], str(val))
            
            summary_data.append({'priority': p_counts, 'matrix': m_counts})

    # 5. Compile Excel
    compile_all_results(summary_data)