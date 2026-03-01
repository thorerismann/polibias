import os
import json
import subprocess
import concurrent.futures
from pathlib import Path
import time

PROMPT = """You are a political-bias scoring tool.

Read the article text and output ONLY valid JSON according to the schema below.

All bias scores are floats in [-1.0, +1.0].

Sign convention (consistent):
-1.0 = left-leaning or favorable to the left
+1.0 = right-leaning or favorable to the right
0.0 = neutral or unclear

Definitions:
1) subject_bias:
   Does the topic selection itself lean left or right?

2) framing_bias:
   Is the framing, tone, or narrative left-leaning or right-leaning?

3) treatment_bias:
   Does the article treat the left or the right more favorably?

4) guests_bias:
   Are quoted speakers or invited voices more left or more right?
   (If no clear political actors, return 0.0.)


=== OUTPUT JSON ONLY ===
Schema:
{
  "subject_bias": <float>,
  "framing_bias": <float>,
  "treatment_bias": <float>,
  "guests_bias": <float>,
  "confidence": <float>,
  "comment": <MAX 2 sentence string>
}
"""

def extract_json(text):
    text = text.strip()
    if text.startswith('```json'):
        text = text[7:]
    elif text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    return text.strip()

def process_file(source, filepath, run_id):
    filename = filepath.name
    out_dir = Path(f"data/runs/comparisons/{source}_results/gemini3/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    
    if out_path.exists():
        return f"Skipped {source}/{filename} (run {run_id})"
    
    with open(filepath, 'r') as f:
        data = json.load(f)
        body = data.get('body', data.get('lead', ''))
        
    cmd = ["gemini", "-p", PROMPT, "-y", "-o", "json"]
    
    for attempt in range(3):
        try:
            result = subprocess.run(cmd, input=body, capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            
            start_idx = output.find('{')
            if start_idx == -1:
                return f"CLI Format Error {source}/{filename} (run {run_id}) (No braces found)"
                
            parsed = json.loads(output[start_idx:])
            response_text = parsed.get("response", "")
            final_json = extract_json(response_text)
            
            final_obj = json.loads(final_json)
            output_data = {
                "subject_bias": final_obj.get("subject_bias"),
                "framing_bias": final_obj.get("framing_bias"),
                "treatment_bias": final_obj.get("treatment_bias"),
                "guests_bias": final_obj.get("guests_bias"),
                "confidence": final_obj.get("confidence"),
                "comment": final_obj.get("comment"),
                "status": "ok",
                "model": "gemini3",
                "article": filename,
                "prompt_hash": "gemini-cli-adhoc"
            }
            
            with open(out_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            return f"Success {source}/{filename} (run {run_id})"
            
        except subprocess.CalledProcessError as e:
            return f"Subprocess Error {source}/{filename} (run {run_id}): {e.stderr}"
        except json.JSONDecodeError as e:
            if attempt == 2:
                # Fallback to save empty
                empty_data = {
                    "subject_bias": None, "framing_bias": None, "treatment_bias": None, "guests_bias": None,
                    "confidence": None, "comment": None, "status": "fallback", "model": "gemini3", "article": filename
                }
                with open(out_path, 'w') as f:
                    json.dump(empty_data, f, indent=2)
                return f"Parse Error {source}/{filename} (run {run_id}): {e} | response text was: {response_text}"
        except Exception as e:
            return f"Unexpected Error {source}/{filename} (run {run_id}): {str(e)}"
        
        time.sleep(1) # Backoff before retry

def main():
    webdata_dir = Path("data/webdata")
    tasks = []
    
    for source_dir in webdata_dir.iterdir():
        if source_dir.is_dir():
            source = source_dir.name
            for filepath in source_dir.glob("*.json"):
                tasks.append((source, filepath, 1))
                tasks.append((source, filepath, 2))
                
    print(f"Total tasks: {len(tasks)}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_file, s, f, r): (s, f, r) for s, f, r in tasks}
        
        success = 0
        failed = 0
        skipped = 0
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            print(f"[{i+1}/{len(tasks)}] {res}")
            if "Success" in res:
                success += 1
            elif "Skipped" in res:
                skipped += 1
            else:
                failed += 1
                
        print(f"Finished! Success: {success}, Skipped: {skipped}, Failed: {failed}")

if __name__ == "__main__":
    main()
