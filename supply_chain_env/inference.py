import os
import sys
import json
import urllib.request
import urllib.error
import ssl

# Fix for macOS SSL certificate verification error
ssl_context = ssl._create_unverified_context()

# Add parent directory to sys.path so 'supply_chain_env' is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
# Load .env file manually if it exists (avoids extra dependencies)
env_path = os.path.join(current_dir, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                parts = line.strip().split("=", 1)
                if len(parts) == 2:
                    key, val = parts
                    os.environ[key] = val.strip('"').strip("'")

from supply_chain_env.client import SupplyChainEnv
from supply_chain_env.models import SupplyChainAction

# 1. Read API credentials from environment variables
# Auto-loads from .env if present
API_KEY = os.environ.get("GROQ_API_KEY") or os.environ.get("API_KEY")

if API_KEY and API_KEY.startswith("gsk_"):
    # Groq Configuration (Free & Fast)
    API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
    MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
else:
    # OpenAI Configuration (Standard)
    API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

SYSTEM_PROMPT = "You are a procurement manager. Available actions: query_supplier, place_order, negotiate_price, notify_customer, expedite_shipment, cancel_order, declare_done. Respond with ONLY valid JSON matching the SupplyChainAction schema."

def call_llm(prompt: str) -> dict:
    url = f"{API_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }
    
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60.0, context=ssl_context) as resp:
            resp_body = resp.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        raise Exception(f"HTTPError: {e.code} {e.reason}")
    except Exception as e:
        raise Exception(f"Request failed: {str(e)}")
        
    content = json.loads(resp_body)["choices"][0]["message"]["content"]
    
    # Cleanup markdown JSON blocks in case LLM adds them
    content = content.strip()

    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
        
    return json.loads(content.strip())

def get_action_from_llm(obs_str: str) -> SupplyChainAction:
    try:
        data = call_llm(obs_str)
        return SupplyChainAction(**data)
    except Exception as e:
        # If API call fails or JSON is invalid, retry once
        print(f"Agent thinking failed (retrying): {e}")
        retry_prompt = f"{obs_str}\n\nYour previous response was invalid. Error: {str(e)}. Respond ONLY with valid JSON."
        try:
            data = call_llm(retry_prompt)
            return SupplyChainAction(**data)
        except Exception as retry_e:
            # Skip step and use declare_done action if it fails again
            print(f"Agent fallback to declare_done due to error: {retry_e}")
            return SupplyChainAction(action_type="declare_done")

def run_task(task_id: int, env=None, callback=None):
    """Run a single task and return the score. Optional callback(step, action, obs, done, score)."""
    if env is None:
        env = SupplyChainEnv(base_url=ENV_URL).sync()
        
    try:
        # b. Call reset(task_id=task_id) to get initial observation
        result = env.reset(task_id=task_id)
        obs = result.observation
        done = result.done
        
        if callback:
            callback(0, None, obs, done, 0.0)

        # c. Run up to 30 steps
        for step_idx in range(30):
            if done:
                break
                
            obs_str = obs.model_dump_json(indent=2)
            action = get_action_from_llm(obs_str)
            
            try:
                # Call env.step(action)
                result = env.step(action)
                obs = result.observation
                done = result.done
                if callback:
                    callback(step_idx + 1, action, obs, done, 0.0)
                print(f"Step {step_idx+1}: {action.action_type} -> done={done}")
            except Exception as e:
                print(f"Step {step_idx+1} failed: {e}")
                # Environment error: skip and use declare_done
                try:
                    action = SupplyChainAction(action_type="declare_done")
                    result = env.step(action)
                    obs = result.observation
                    done = result.done
                    if callback:
                        callback(step_idx + 1, action, obs, done, 0.0)
                except Exception:
                    done = True
                break
                
        # d. Call /grade endpoint to get final score
        final_state = env.state()
        grade_result = env.grade(task_id=task_id, final_state=final_state)
        score = grade_result.get("score", 0.0)
        
        if callback:
            callback(step_idx + 1, None, obs, True, score)
            
        return score
            
    except Exception as e:
        print(f"Task {task_id} failed with error: {e}")
        return 0.0

def main():
    env = SupplyChainEnv(base_url=ENV_URL).sync()
    scores = []
    
    for task_id in [0, 1, 2]:
        print(f"\n--- Starting Task {task_id} ---")
        score = run_task(task_id, env=env)
        print(f"Task {task_id}: score={score:.4f}")
        scores.append(score)
        
    print("\n" + "="*40)
    print("FINAL BASELINE SCORES")
    print("="*40)
    print(f"Easy: {scores[0]:.4f} | Medium: {scores[1]:.4f} | Hard: {scores[2]:.4f}")
    print(f"Average: {(sum(scores) / 3):.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
