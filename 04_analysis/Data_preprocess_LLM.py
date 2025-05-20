import pandas as pd
import json

# load CSV (modify local path)
df = pd.read_csv("01_data/ANES/anes_timeseries_2020_csv_20220210/anes_timeseries_2020_csv_20220210.csv")

# model = 'gpt'
# model = 'claude'
model = 'gemini'
#model = 'mistral/mistralai'
# model = 'llama/meta-llama'
#model = 'deepseek'

# model_name = 'gpt-3.5-turbo'
# model_name = 'gpt-4o-mini'
# model_name = 'claude-3-haiku-20240307'
#model_name = 'models_gemini-1.5-flash-latest'
model_name = 'models_gemini-2.0-flash-001'
# model_name = 'Llama-3-8b-chat-hf'
# model_name = 'Mistral-7B-Instruct-v0.3'
#model_name = 'Meta-Llama-3.1-8B-Instruct-Turbo'
# model_name = 'deepseek-coder'
date = '2025-05-17'

# load JSON file
df_llm = pd.read_json(f"03_results/{model}/{model_name}/{date}/responses_final.jsonl", lines=True)
# df_llm = pd.read_json(f"03_results/{model}/{model_name}/{date}/merged_final.jsonl", lines=True)

# select necessary columns
df_simple = df_llm[['persona_id', 'question_id', 'response']]

# pivot to wide-format
df_llm_ques = df_simple.pivot(index='persona_id', columns='question_id', values='response')
df_llm_ques = df_llm_ques.rename_axis(columns=None).reset_index()

# sort columns (optional)
df_llm_ques = df_llm_ques[['persona_id'] + sorted([col for col in df_llm_ques.columns if col != 'persona_id'])]

# check types (before conversion)
print("===== before conversion =====")
print(df_llm_ques.dtypes)

# correct float conversion method
# method 1: use pd.to_numeric
for col in df_llm_ques.columns:
    if col != 'persona_id':  # persona_id is not converted
        df_llm_ques[col] = pd.to_numeric(df_llm_ques[col], errors='coerce')

# check types (after conversion)
print("\n===== after conversion =====")
print(df_llm_ques.dtypes)

# select necessary columns
df_IRT = df_llm_ques[['q1', 'q2', 'q3', 'q4', 'q5', 
                      # 'q6'
                      ]].copy()
df_IRT.columns = ['gun_control', 'welfare','immigration', 'transgender', 'govt_treat_race', 
                  #'police_treat_race'
                  ]

# check missing values and outliers
print("=== dataframe structure ===")
print(df_IRT.info())
print("\n=== number of missing values ===")
print(df_IRT.isnull().sum())
print("\n=== check outliers (values out of 1~3 range) ===")
for col in df_IRT.columns:
    invalid_values = df_IRT[~df_IRT[col].isin([1, 2, 3]) & df_IRT[col].notnull()]
    if not invalid_values.empty:
        print(f"[warning] {col} has outliers:")
        print(invalid_values[col].value_counts(dropna=False))
    else:
        print(f"{col}: no outliers")
        
        
# recoding mapping for each question
recode_maps = {
    'gun_control': {1: 1, 2: 3, 3: 2},
    'welfare':     {1: 1, 2: 3, 3: 2},
    'immigration': {1: 3, 2: 1, 3: 2},
    'transgender': {1: 1, 2: 3, 3: 2},
    'govt_treat_race': {1: 3, 2: 2, 3: 1}
}

# perform recoding
for col, mapping in recode_maps.items():
    df_IRT[col] = df_IRT[col].map(mapping)

# check missing values and outliers
print("=== check missing values ===")
print(df_IRT.isnull().sum())


# load demographic data
df_demo = pd.read_csv('01_data/ANES_demographics.csv')

# merge based on ID (assume 'case_id' is the common key)
df_full = pd.concat([df_IRT, df,df_demo], axis=1)  # merge based on index (only if indices are the same)

# remove rows with missing values in political questions or demographic data
cols_for_analysis = ['gun_control', 'welfare', 'immigration', 'transgender', 'govt_treat_race']  # V201600 = gender
df_clean = df_full.dropna(subset=cols_for_analysis).reset_index(drop=True)

print(f"number of final analysis targets: {len(df_clean)}")


# save
df_clean.to_csv(f'01_data/ANES_{model_name}_{date}.csv', index=False)
print(f"saved: 01_data/ANES_{model_name}_{date}.csv")
