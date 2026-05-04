"""LLM-based open-coded annotation of all 15,582 grievances using gpt-4o-mini.

Two-pass design so categories *emerge* rather than being imposed:

  Pass A (discovery): sample ~300 rows, ask the LLM for a free-text 1-3 word
  category each. Cluster these labels into a consolidated taxonomy of 5-10
  buckets (the LLM does the consolidation itself in a single call).

  Pass B (full set): 1 row per request, 40 parallel workers, gpt-4o-mini
  assigns each row to one of the consolidated buckets and returns a
  confidence score. tqdm progress bar over completions; partial saves every
  500 rows so a crash can resume.

Outputs:
  outputs/llm_taxonomy.json  — consolidated category list
  outputs/llm_labels.csv     — registration_no, llm_category, llm_confidence
  outputs/llm_labels_partial.csv — checkpoint during Pass B

Run:  python llm_annotate.py [--workers 40] [--limit 0] [--skip-confirm]
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

OUTPUT_DIR = "outputs"
MODEL = "gpt-4o-mini"
DISCOVERY_SAMPLE = 300
CHECKPOINT_EVERY = 500
SEED = 42

# gpt-4o-mini pricing (approx, USD per 1M tokens — used for the upfront estimate)
PRICE_INPUT_PER_1M = 0.15
PRICE_OUTPUT_PER_1M = 0.60


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------- Pass A: open-coded discovery ----------

DISCOVERY_SYSTEM = """You are an expert at categorizing Indian Railway citizen grievances.
Complaints may be in English, Hindi, or 'Hinglish' (Hindi written in Latin script).
Read the complaint and respond with a SHORT 1-3 word category that captures the core issue.
Examples: "refund delay", "ticket booking", "train cleanliness", "staff misbehavior", "platform infrastructure".
Return strictly as JSON: {"category": "..."}"""


def discover_label(text):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": DISCOVERY_SYSTEM},
                    {"role": "user", "content": text[:2000]},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            return json.loads(resp.choices[0].message.content).get("category", "unknown")
        except Exception as e:
            if attempt == 2:
                return f"error:{type(e).__name__}"
            time.sleep(2 ** attempt)


CONSOLIDATE_SYSTEM = """You are organizing a list of free-text complaint categories into a clean taxonomy.
You will receive ~300 raw category labels. Consolidate them into 5-10 broad, mutually-exclusive buckets that cover all the labels.
Each bucket should have a short snake_case id and a one-line description.
Return strictly as JSON: {"buckets": [{"id": "payment_refund", "description": "..."}, ...]}"""


def consolidate_taxonomy(raw_labels):
    joined = "\n".join(f"- {l}" for l in raw_labels)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": CONSOLIDATE_SYSTEM},
            {"role": "user", "content": f"Raw labels:\n{joined}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(resp.choices[0].message.content)["buckets"]


# ---------- Pass B: assign each row to a bucket ----------

def make_assign_system(taxonomy):
    bucket_lines = "\n".join(f"  - {b['id']}: {b['description']}" for b in taxonomy)
    return f"""You are categorizing Indian Railway citizen grievances into a fixed taxonomy.
Complaints may be in English, Hindi, or 'Hinglish' (Hindi written in Latin script).

Choose exactly one bucket from this list:
{bucket_lines}

Also report your confidence (0.0 to 1.0).
Return strictly as JSON: {{"category": "<bucket_id>", "confidence": 0.0-1.0}}"""


RETRY_BACKOFF = [2, 4, 8, 16, 30]  # seconds, with jitter; 5 attempts total


def _sleep_with_jitter(seconds):
    time.sleep(seconds + random.uniform(0, seconds * 0.3))


def _log_retry(where, exc, attempt, sleep_for):
    """Print a one-line warning about a retry. Uses tqdm.write so it doesn't
    break the active progress bar."""
    err_type = type(exc).__name__
    msg = str(exc)
    if len(msg) > 120:
        msg = msg[:117] + "..."
    label = "RATE_LIMIT" if "429" in msg or "rate" in msg.lower() else err_type
    line = f"[{where}] {label} on attempt {attempt+1} (sleeping {sleep_for:.1f}s): {msg}"
    try:
        tqdm.write(line)
    except Exception:
        print(line, flush=True)


def assign_label(text, system_prompt):
    for attempt in range(len(RETRY_BACKOFF) + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text[:2000]},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            data = json.loads(resp.choices[0].message.content)
            return data.get("category", "unknown"), float(data.get("confidence", 0.0))
        except Exception as e:
            if attempt >= len(RETRY_BACKOFF):
                _log_retry("assign", e, attempt, 0.0)
                return "error", 0.0
            _log_retry("assign", e, attempt, RETRY_BACKOFF[attempt])
            _sleep_with_jitter(RETRY_BACKOFF[attempt])


# ---------- Pass B batched: N rows per request ----------

def make_assign_system_batched(taxonomy, batch_size):
    bucket_lines = "\n".join(f"  - {b['id']}: {b['description']}" for b in taxonomy)
    return f"""You are categorizing Indian Railway citizen grievances into a fixed taxonomy.
Complaints may be in English, Hindi, or 'Hinglish' (Hindi written in Latin script).

You will receive a JSON array of complaints under the key "rows", indexed 0..N-1.
For EACH row, choose exactly one bucket from this list:
{bucket_lines}

Also report your confidence (0.0 to 1.0) per row.

Return strictly JSON with EXACTLY {batch_size} entries in the SAME ORDER as the input,
under the key "labels":
{{"labels": [{{"index": 0, "category": "<bucket_id>", "confidence": 0.0-1.0}}, ...]}}"""


def assign_labels_batch(texts, system_prompt, expected_size):
    """Send a batch of texts in one request. Returns list[(cat, conf)] of length expected_size,
    or None on failure (caller should fall back to per-row).
    """
    payload = {"rows": [t[:1500] for t in texts]}
    for attempt in range(len(RETRY_BACKOFF) + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            data = json.loads(resp.choices[0].message.content)
            labels = data.get("labels", [])
            if len(labels) != expected_size:
                mismatch = ValueError(
                    f"batch size mismatch: expected {expected_size}, got {len(labels)}"
                )
                if attempt >= len(RETRY_BACKOFF):
                    _log_retry("batch", mismatch, attempt, 0.0)
                    return None
                _log_retry("batch", mismatch, attempt, RETRY_BACKOFF[attempt])
                _sleep_with_jitter(RETRY_BACKOFF[attempt])
                continue
            try:
                labels = sorted(labels, key=lambda d: int(d.get("index", 0)))
            except Exception:
                pass
            out = []
            for d in labels:
                cat = d.get("category", "unknown")
                try:
                    conf = float(d.get("confidence", 0.0))
                except Exception:
                    conf = 0.0
                out.append((cat, conf))
            return out
        except Exception as e:
            if attempt >= len(RETRY_BACKOFF):
                _log_retry("batch", e, attempt, 0.0)
                return None
            _log_retry("batch", e, attempt, RETRY_BACKOFF[attempt])
            _sleep_with_jitter(RETRY_BACKOFF[attempt])


def estimate_cost(n_rows, avg_input_tokens=200, avg_output_tokens=30):
    in_cost = n_rows * avg_input_tokens / 1_000_000 * PRICE_INPUT_PER_1M
    out_cost = n_rows * avg_output_tokens / 1_000_000 * PRICE_OUTPUT_PER_1M
    return in_cost + out_cost


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="preprocessed_grievances.csv")
    parser.add_argument("--workers", type=int, default=10,
                        help="Concurrent workers. Default 10 — high concurrency on Tier-1 wastes calls in retry storms.")
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    parser.add_argument("--skip-confirm", action="store_true")
    parser.add_argument("--taxonomy-only", action="store_true",
                        help="Run discovery+consolidation only, skip Pass B")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Rows per OpenAI request. >1 enables batched mode.")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(args.csv)
    # Apply the SAME filtering as embeddings_utils.load_sample so row_idx
    # values line up across the LLM labels and the clustering sample.
    df = df.dropna(subset=["cleaned_text"]).reset_index(drop=True)
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)
    df = df.dropna(subset=["subject_content_text"]).reset_index(drop=True)
    # Source data has duplicate registration_nos (each ID appears ~10x).
    # Use positional row_idx as the true unique key for dedup/resume.
    df["row_idx"] = df.index.astype(int)
    if args.limit:
        df = df.head(args.limit)
    print(f"[llm] {len(df)} rows total ({df['registration_no'].nunique()} unique registration_nos)")

    est = estimate_cost(len(df) + DISCOVERY_SAMPLE + 1)
    print(f"[llm] estimated cost: ${est:.2f} (gpt-4o-mini, {len(df)} + {DISCOVERY_SAMPLE} discovery calls)")
    if not args.skip_confirm:
        ans = input("Proceed? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            sys.exit(0)

    taxonomy_path = os.path.join(OUTPUT_DIR, "llm_taxonomy.json")

    # Pass A: discovery
    if os.path.exists(taxonomy_path):
        print(f"[llm] reusing existing taxonomy at {taxonomy_path}")
        with open(taxonomy_path) as f:
            taxonomy = json.load(f)
    else:
        print(f"[llm] Pass A: discovering categories on {DISCOVERY_SAMPLE}-row sample ...")
        sample = df.sample(n=min(DISCOVERY_SAMPLE, len(df)), random_state=SEED)
        raw_labels = []
        with ThreadPoolExecutor(max_workers=min(args.workers, 20)) as ex:
            futures = {ex.submit(discover_label, t): i for i, t in enumerate(sample["subject_content_text"])}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="discovery"):
                raw_labels.append(fut.result())
        print(f"[llm] {len(set(raw_labels))} unique raw labels — consolidating into taxonomy ...")
        taxonomy = consolidate_taxonomy(raw_labels)
        with open(taxonomy_path, "w") as f:
            json.dump(taxonomy, f, indent=2)
        print(f"[llm] saved taxonomy: {len(taxonomy)} buckets -> {taxonomy_path}")
        for b in taxonomy:
            print(f"  - {b['id']}: {b['description']}")

    if args.taxonomy_only:
        print("[llm] --taxonomy-only set; stopping after Pass A.")
        return

    # Pass B: full assignment
    batch_size = max(1, args.batch_size)
    mode = f"batched (batch_size={batch_size})" if batch_size > 1 else "per-row"
    print(f"[llm] Pass B: assigning {len(df)} rows with {args.workers} workers, {mode}")
    if batch_size > 1:
        system_prompt = make_assign_system_batched(taxonomy, batch_size)
        per_row_fallback_prompt = make_assign_system(taxonomy)
    else:
        system_prompt = make_assign_system(taxonomy)
        per_row_fallback_prompt = system_prompt

    partial_path = os.path.join(OUTPUT_DIR, "llm_labels_partial.csv")
    final_path = os.path.join(OUTPUT_DIR, "llm_labels.csv")

    done_idx = set()
    if os.path.exists(partial_path):
        prior = pd.read_csv(partial_path)
        if "row_idx" in prior.columns:
            done_idx = set(prior["row_idx"].astype(int).tolist())
            print(f"[llm] resuming: {len(done_idx)} rows already labeled")
        else:
            print(f"[llm] {partial_path} has old schema (no row_idx) — discarding and starting fresh")
            os.remove(partial_path)

    todo = df[~df["row_idx"].isin(done_idx)].reset_index(drop=True)
    if len(todo) == 0:
        print("[llm] nothing to do")
        os.replace(partial_path, final_path) if os.path.exists(partial_path) else None
        return

    results = []
    completed = 0
    last_checkpoint = 0
    pbar = tqdm(total=len(todo), desc="assignment")

    def submit_single(row):
        cat, conf = assign_label(row["subject_content_text"], per_row_fallback_prompt)
        return [(int(row["row_idx"]), row["registration_no"], cat, conf)]

    def submit_batch(rows):
        texts = [r["subject_content_text"] for r in rows]
        idxs = [int(r["row_idx"]) for r in rows]
        regs = [r["registration_no"] for r in rows]
        labels = assign_labels_batch(texts, system_prompt, expected_size=len(rows))
        if labels is None:
            out = []
            for r in rows:
                cat, conf = assign_label(r["subject_content_text"], per_row_fallback_prompt)
                out.append((int(r["row_idx"]), r["registration_no"], cat, conf))
            return out
        return [(idx, reg, cat, conf) for idx, reg, (cat, conf) in zip(idxs, regs, labels)]

    rows_list = todo.to_dict(orient="records")
    if batch_size > 1:
        work_units = [rows_list[i:i + batch_size] for i in range(0, len(rows_list), batch_size)]
        submit_fn = submit_batch
    else:
        work_units = rows_list
        submit_fn = submit_single

    cols = ["row_idx", "registration_no", "llm_category", "llm_confidence"]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(submit_fn, unit) for unit in work_units]
        for fut in as_completed(futures):
            tuples = fut.result()
            for idx, reg, cat, conf in tuples:
                results.append((idx, reg, cat, conf))
                completed += 1
                pbar.update(1)

                if completed - last_checkpoint >= CHECKPOINT_EVERY:
                    last_checkpoint = completed
                    cur = pd.DataFrame(results, columns=cols)
                    if os.path.exists(partial_path):
                        prior = pd.read_csv(partial_path)
                        cur = pd.concat([prior, cur], ignore_index=True)
                    # Dedup on row_idx (truly unique), not registration_no
                    cur = cur.drop_duplicates(subset="row_idx", keep="last")
                    cur.to_csv(partial_path, index=False)
                    results = []
                    err_pct = 100.0 * (cur["llm_category"] == "error").mean()
                    pbar.set_postfix({"saved": len(cur), "err%": f"{err_pct:.1f}"})

    pbar.close()

    # final flush + rename
    cur = pd.DataFrame(results, columns=cols)
    if os.path.exists(partial_path):
        prior = pd.read_csv(partial_path)
        cur = pd.concat([prior, cur], ignore_index=True)
    cur = cur.drop_duplicates(subset="row_idx", keep="last").sort_values("row_idx")
    cur.to_csv(final_path, index=False)
    if os.path.exists(partial_path) and partial_path != final_path:
        os.remove(partial_path)
    err_pct = 100.0 * (cur["llm_category"] == "error").mean()
    print(f"[llm] saved {len(cur)} labels to {final_path}  (errors: {err_pct:.1f}%)")

    # quick distribution report
    print("\n[llm] category distribution:")
    print(cur["llm_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
