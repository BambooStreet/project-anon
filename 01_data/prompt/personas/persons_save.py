import pandas as pd
import json

personas_version = "v2"

# load demographic data
df = pd.read_csv("01_data/ANES_demographics.csv")

# generate persona description
def generate_persona_description(row):
    desc = []
    if pd.notnull(row['age']):
        desc.append(f"{int(row['age'])} years old")
    if pd.notnull(row['gender']):
        desc.append(f"{row['gender'].lower()}")
    if 'race' in row and pd.notnull(row['race']):
        desc.append(f"{row['race'].lower()}")
    if pd.notnull(row['household_income']):
        desc.append(f"in the income bracket '{row['household_income']}'")
    if pd.notnull(row['education']):
        desc.append(f"with an education level of {row['education'].lower()}")
    # if pd.notnull(row['political_ideology']):
    #     desc.append(f"who identifies as {row['political_ideology'].lower()}")
    if pd.notnull(row['religion']):
        desc.append(f"and religiously identifies as {row['religion'].lower()}")

    return "I am " + ", ".join(desc) + "."

with open("01_data/prompt/personas/personas_texts.jsonl", "w", encoding="utf-8") as f:
    for idx, row in df.iterrows():
        persona_id = f"{idx:06d}"
        persona_text = generate_persona_description(row)

        record = {
            "persona_id": persona_id,
            "text": persona_text
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")