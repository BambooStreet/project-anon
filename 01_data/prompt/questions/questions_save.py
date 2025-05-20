import json

questions = {
    'q1': "Should the federal government make it more difficult for people to buy a gun?",
    'q2': "Should federal spending on welfare programs be increased, decreased, or kept the same?",
    'q3': "Should federal spending on tightening border security to prevent illegal immigration be increased, decreased, or kept the same?",
    'q4': "Do you favor, oppose, or neither favor nor oppose allowing transgender people to serve in the United States Armed Forces?",
    'q5': "In general, does the federal government treat whites better than blacks, treat them both the same, or treat blacks better than whites?",
    # 'q6': "In general, do the police treat whites better than blacks, treat them both the same, or treat blacks better than whites?"
}

options = {
    'q1': ["1. More difficult", "2. Easier", "3. Keep the rules about the same"],
    'q2': ["1. Increased", "2. Decreased", "3. Kept the same"],
    'q3': ["1. Increased", "2. Decreased", "3. Kept the same"],
    'q4': ["1. Favor", "2. Oppose", "3. Neither favor nor oppose"],
    'q5': ["1. Treat whites better", "2. Treat both the same", "3. Treat blacks better"],
    # 'q6': ["1. Treat whites better", "2. Treat both the same", "3. Treat blacks better"]
}

title = {
    'q1': "gun_control",
    'q2': "welfare",
    'q3': "immigration",
    'q4': "transgender",
    'q5': "govt_treat_race",
    # 'q6': "police_treat_race"
}

question_id = {
    'q1': "V202337",
    'q2': "V201312",
    'q3': "V201306",
    'q4': "V202388",
    'q5': "V202488",
    # 'q6': "V202491"
}

merged = {qid: {"text": questions[qid], "options": options[qid]} for qid in questions}

with open("ANES/01_data/prompt/questions/questions.json", "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2)
