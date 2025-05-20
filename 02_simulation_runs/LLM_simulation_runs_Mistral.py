import os
import time
import json
import yaml
import pandas as pd
import re
from datetime import datetime
from openai import OpenAI



# ===== LLaMA 3 (Together.ai) settings =====
client = OpenAI(
    api_key='',
    base_url='https://api.together.xyz'
)

# ===== experiment settings =====
MODEL_PATH = "mistral"
# MODEL_NAME = "meta-llama/Llama-3-8b-chat-hf"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
TEMPERATURE = 0.7
PERSONA_VERSION = "v2"
QUESTION_VERSION = "v4"
INSTRUCTION_VERSION = "v1"
SAMPLE_NUM = 8280
RUN_DATE = datetime.now().strftime("%Y-%m-%d")
SAVE_DIR = f"ANES/03_results/{MODEL_PATH}/{MODEL_NAME}/{RUN_DATE}"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===== save experiment settings =====
config = {
    "model": MODEL_NAME,
    "temperature": TEMPERATURE,
    "persona_version": PERSONA_VERSION,
    "question_version": QUESTION_VERSION,
    "instruction_version": INSTRUCTION_VERSION,
    "sample_num": SAMPLE_NUM,
    "date": RUN_DATE
}
with open(os.path.join(SAVE_DIR, "config.yaml"), "w") as f:
    yaml.dump(config, f)


# ===== load data =====
persona_texts = {}
with open(f"ANES/01_data/prompt/personas/personas_texts_{PERSONA_VERSION}.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        persona_texts[obj["persona_id"]] = obj["text"]

with open(f"ANES/01_data/prompt/questions/questions_{QUESTION_VERSION}.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

with open(f"ANES/01_data/prompt/instructions/instructions_{INSTRUCTION_VERSION}.json", "r", encoding="utf-8") as f:
    instruction = json.load(f)


# ===== LLaMA API call function =====
def ask_llama(prompt, model=MODEL_NAME, temperature=TEMPERATURE, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=10
            )
            content = response.choices[0].message.content
            if content:
                match = re.search(r"\b[1-3]\b", content)
                return match.group(0) if match else content.strip()
            else:
                print(f"⚠️ empty response (attempt {attempt+1})")
        except Exception as e:
            print(f"❌ Error: {e} (attempt {attempt + 1})")
        time.sleep(3)
    return "ERROR_NO_RESPONSE"

# ===== simulation run =====
start_time = time.time()
results = []

persona_ids = list(persona_texts.keys())[:SAMPLE_NUM]

for i, pid in enumerate(persona_ids):
    persona_text = persona_texts[pid]
    print(f"\n=== Persona {i+1}/{len(persona_ids)} (pid:{pid}) ===")

    for qid, qdata in questions.items():
        prompt = (
            f"{persona_text} {instruction['text']}\n\n"
            f"{qid.upper()}: {qdata['text']}\n" +
            "\n".join(qdata["options"]) +
            "\n\nAnswer with: 1, 2, or 3 only."
        )

        response = ask_llama(prompt)
        print(f"{qid.upper()}: {response}")

        results.append({
            "persona_id": pid,
            "question_id": qid,
            "model": MODEL_NAME,
            "temperature": TEMPERATURE,
            "persona_version": PERSONA_VERSION,
            "question_version": QUESTION_VERSION,
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })

        time.sleep(1)

    if i < 3:
        print("=== Prompt Preview ===")
        print(prompt)

    if (i + 1) % 100 == 0:
        with open(os.path.join(SAVE_DIR, f"checkpoint_{i+1}.jsonl"), "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✅ {i+1} finished")



# ===== save final results =====
with open(os.path.join(SAVE_DIR, "responses_final.jsonl"), "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

elapsed = (time.time() - start_time) / 60
print(f"\n=== all finished! time: {elapsed:.2f} minutes ===")
