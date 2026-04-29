import os
import pandas as pd

def assign_raw_priority(frustration, request):
    """Calculates priority for the raw baseline since it wasn't saved in the cache."""
    if frustration == "Frustrated" and request == "Action_Required":
        return "INCIDENT (P1)"
    elif frustration == "Frustrated":
        return "ESCALATION (P2)"
    elif request == "Action_Required":
        return "ROUTINE (P3)"
    else:
        return "RECORD (P4)"

def extract_matrix_data(df, algo_name, param_name, frust_col="Frustration", req_col="Request"):
    """Extracts the 2x2 grid data for the Signal Matrix."""
    # Group by the two columns to get the intersection counts
    matrix_counts = df.groupby([frust_col, req_col]).size().to_dict()
    
    return {
        'Algorithm': algo_name,
        'Parameter': param_name,
        'Frustrated + Action_Required': matrix_counts.get(('Frustrated', 'Action_Required'), 0),
        'Frustrated + Statement': matrix_counts.get(('Frustrated', 'Statement'), 0),
        'Neutral + Action_Required': matrix_counts.get(('Neutral', 'Action_Required'), 0),
        'Neutral + Statement': matrix_counts.get(('Neutral', 'Statement'), 0)
    }

def compile_results():
    print("Gathering data from results folders...")
    priority_data = []
    matrix_data = []

    # 1. Fetch Zero-Shot Baseline (The Raw Holdout Predictions)
    raw_path = os.path.join("results", "baseline", "holdout_zero_shot_raw.csv")
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)
        # Reconstruct Priority
        df_raw['Priority'] = df_raw.apply(lambda row: assign_raw_priority(row['Raw_Frustration'], row['Raw_Request']), axis=1)
        
        # Priority Buckets
        p_counts = df_raw['Priority'].value_counts().to_dict()
        p_counts['Algorithm'] = "Zero-Shot Baseline"
        p_counts['Parameter'] = "None"
        priority_data.append(p_counts)
        
        # Signal Matrix
        m_counts = extract_matrix_data(df_raw, "Zero-Shot Baseline", "None", frust_col="Raw_Frustration", req_col="Raw_Request")
        matrix_data.append(m_counts)
    else:
        print(f"[Warning] Raw baseline not found at {raw_path}")

    # 2. Fetch Bayesian Results
    bayesian_vals = [0.5, 1.0]
    for b in bayesian_vals:
        path = os.path.join("results", "bayesian", str(b), f"b_{b}_holdout_smoothed_results.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            
            p_counts = df['Priority'].value_counts().to_dict()
            p_counts['Algorithm'] = "Bayesian Additive"
            p_counts['Parameter'] = f"b = {b}"
            priority_data.append(p_counts)
            
            m_counts = extract_matrix_data(df, "Bayesian Additive", f"b = {b}")
            matrix_data.append(m_counts)

    # 3. Fetch Dirichlet Results
    dirichlet_vals = [1.5]
    for mu in dirichlet_vals:
        path = os.path.join("results", "dirichlet", str(mu), f"d_{mu}_holdout_smoothed_results.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            
            p_counts = df['Priority'].value_counts().to_dict()
            p_counts['Algorithm'] = "Dirichlet"
            p_counts['Parameter'] = f"μ = {mu}"
            priority_data.append(p_counts)
            
            m_counts = extract_matrix_data(df, "Dirichlet", f"μ = {mu}")
            matrix_data.append(m_counts)

    # 4. Fetch Jelinek-Mercer Results
    jm_vals = [0.2, 0.4, 0.6, 0.8]
    for jm in jm_vals:
        path = os.path.join("results", "jelinek-mercer", str(jm), f"jm_{jm}_holdout_smoothed_results.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            
            p_counts = df['Priority'].value_counts().to_dict()
            p_counts['Algorithm'] = "Jelinek-Mercer"
            p_counts['Parameter'] = f"λ = {jm}"
            priority_data.append(p_counts)
            
            m_counts = extract_matrix_data(df, "Jelinek-Mercer", f"λ = {jm}")
            matrix_data.append(m_counts)

    # --- Process Priority DataFrame ---
    priority_df = pd.DataFrame(priority_data)
    expected_p_cols = ['Algorithm', 'Parameter', 'INCIDENT (P1)', 'ESCALATION (P2)', 'ROUTINE (P3)', 'RECORD (P4)']
    
    for col in expected_p_cols:
        if col not in priority_df.columns:
            priority_df[col] = 0

    priority_df = priority_df[expected_p_cols].fillna(0).astype({
        'INCIDENT (P1)': int, 'ESCALATION (P2)': int, 'ROUTINE (P3)': int, 'RECORD (P4)': int
    })

    # --- Process Matrix DataFrame ---
    matrix_df = pd.DataFrame(matrix_data)
    expected_m_cols = ['Algorithm', 'Parameter', 'Frustrated + Action_Required', 'Frustrated + Statement', 'Neutral + Action_Required', 'Neutral + Statement']
    
    for col in expected_m_cols:
        if col not in matrix_df.columns:
            matrix_df[col] = 0
            
    matrix_df = matrix_df[expected_m_cols].fillna(0).astype({
        'Frustrated + Action_Required': int, 'Frustrated + Statement': int, 'Neutral + Action_Required': int, 'Neutral + Statement': int
    })

    # --- Save to Excel ---
    file_priority = os.path.join("results", "compiled_priority_buckets.xlsx")
    file_matrix = os.path.join("results", "compiled_signal_matrix.xlsx")
    
    # Save Priority Excel
    with pd.ExcelWriter(file_priority, engine='openpyxl') as writer:
        priority_df.to_excel(writer, index=False, sheet_name='Priority Shifts')
        worksheet = writer.sheets['Priority Shifts']
        for idx, col in enumerate(priority_df.columns):
            max_len = max(priority_df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.column_dimensions[chr(65 + idx)].width = max_len

    # Save Matrix Excel
    with pd.ExcelWriter(file_matrix, engine='openpyxl') as writer:
        matrix_df.to_excel(writer, index=False, sheet_name='Signal Matrix')
        worksheet = writer.sheets['Signal Matrix']
        for idx, col in enumerate(matrix_df.columns):
            max_len = max(matrix_df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.column_dimensions[chr(65 + idx)].width = max_len

    print(f"\n✅ Success! Compiled runs into:")
    print(f" 1. {file_priority}")
    print(f" 2. {file_matrix}")

if __name__ == "__main__":
    compile_results()