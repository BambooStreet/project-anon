import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# model_name = 'gpt-3.5-turbo'
# model_name = 'gpt-4o-mini'
# model_name = 'claude-3-haiku-20240307'
# model_name = 'models_gemini-1.5-flash-latest'
# model_name = 'Llama-3-8b-chat-hf'


models = [
    'gpt-4o-mini', 
    'gpt-3.5-turbo', 
    'claude-3-haiku-20240307', 
    'models_gemini-2.0-flash-001', 
    'Meta-Llama-3.1-8B-Instruct-Turbo',
    'Mistral-7B-Instruct-v0.3',
 ]

date = "2025-05-18"




for model_name in models:
    # load CSV (modify the local path)
    df = pd.read_csv(f'02_results/LLM/{model_name}_IRT_GRM_{date}/theta_{model_name}_robustness2.csv')
    
    # specify the category order to force the baseline value
    df['gender_group'] = pd.Categorical(df['gender_group'], categories=['Female', 'Male'], ordered=False)
    df['education_group'] = pd.Categorical(df['education_group'], categories=['High', 'Mid', 'Low'], ordered=False)
    df['ideology_group'] = pd.Categorical(df['ideology_group'], categories=['Moderate', 'Liberal', 'Conservative'], ordered=False)
    df['age_group'] = pd.Categorical(df['age_group'], categories=['35~59', 'Under 35', 'Over 60'], ordered=False)
    df['income_group'] = pd.Categorical(df['income_group'], categories=['Low', 'Middle', 'High'], ordered=False)
    df['marital_group'] = pd.Categorical(df['marital_group'], categories=['Married', 'Never married', 'Previously married'], ordered=False)
    df['religion_group'] = pd.Categorical(df['religion_group'], categories=['Western religion', 'Non-religious', 'Non-Western religion', 'Other religion'], ordered=False)
    df['race_group'] = pd.Categorical(df['race_group'], categories=['White', 'Black', 'Asian', 'Other'], ordered=False)
    
    # list of questions to analyze
    binary_targets = ['gun_control', 'welfare', 'immigration', 'transgender', 
                      #'govt_treat_race'
                      ]

    # preprocessing: set the binary standard (example: liberal=1, conservative=0)
    binary_maps = {
        'gun_control': {1.0: 1, 2.0: 0, 3.0: 0},
        'welfare': {1.0: 1, 2.0: 0, 3.0: 0},
        'immigration': {1.0: 1, 2.0: 0, 3.0: 0},
        'transgender': {1.0: 1, 2.0: 0, 3.0: 0},
        # 'govt_treat_race': {3.0: 1, 2.0: 0, 1.0: 0}
    }

    # dictionary for saving regression results
    regression_results = {}

    # set common categories
    category_orders = {
        'gender_group': ['Female', 'Male'],
        'education_group': ['High', 'Mid', 'Low'],
        'ideology_group': ['Moderate', 'Liberal', 'Conservative'],
        'age_group': ['35~59', 'Under 35', 'Over 60'],
        'income_group': ['Low', 'Middle', 'High'],
        'marital_group': ['Married', 'Never married', 'Previously married'],
        'religion_group': ['Western religion', 'Non-religious', 'Non-Western religion', 'Other religion'],
        'race_group': ['White', 'Black', 'Asian', 'Other']
    }
    categorical_vars = list(category_orders.keys())

    # repeat analysis for each question
    for target in binary_targets:
        print(f'{model_name} {target} regression started')
        df_temp = df.copy()
        df_temp[target + '_binary'] = df_temp[target].map(binary_maps[target])

        # remove missing values
        df_clean = df_temp.dropna(subset=[target + '_binary', 'theta'] + categorical_vars).copy()
        df_clean[target + '_binary'] = df_clean[target + '_binary'].astype(int)

        # apply category order
        for var, cats in category_orders.items():
            df_clean[var] = pd.Categorical(df_clean[var], categories=cats, ordered=False)

        # one-hot encoding
        df_encoded = pd.get_dummies(df_clean[categorical_vars + ['theta', target + '_binary']],
                                    drop_first=True, dtype='float')

        # regression modeling
        X = sm.add_constant(df_encoded.drop(columns=target + '_binary'))
        y = df_encoded[target + '_binary']
        
        
        try:
            model = sm.Logit(y, X)
            result = model.fit(disp=0)
            regression_results[target] = result.summary2().tables[1]  # Coef table
            print(f'{model_name} {target} regression finished')

        except Exception as e:
            regression_results[target] = f"Error: {e}"
            print(f'{model_name} {target} regression error occurred')
            
    # extract only the questions that were successfully regressed
    combined_results = []

    for target, result in regression_results.items():
        if isinstance(result, pd.DataFrame):
            df_result = result.copy()
            df_result['variable'] = df_result.index
            df_result['question'] = target
            combined_results.append(df_result)

    # combine into one dataframe
    df_all_results = pd.concat(combined_results, ignore_index=True)

    # rearrange columns
    cols = ['question', 'variable'] + [col for col in df_all_results.columns if col not in ['question', 'variable']]
    df_all_results = df_all_results[cols]
    
    df_all_results.to_csv(f'04_analysis/data/DIF_results/LLM_DIF_BinaryLogistic_{model_name}_robustness2.csv', index=False)
    print(f'{model_name} DIF results saved')