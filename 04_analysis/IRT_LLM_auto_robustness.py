import pandas as pd
import numpy as np

from scipy.optimize import minimize
import time
import os
from datetime import datetime

### 1. prepare data
def prepare_irt_data(df_clean):
    valid_responses = [1, 2, 3]
    irt_data = []

    # rearrange user index (continuous integers starting from 0)
    df_clean = df_clean.reset_index(drop=True)
    
    for idx, row in df_clean.iterrows():
        for item_id, response in enumerate(row, start=1):
            if pd.notnull(response) and response in valid_responses:
                irt_data.append([idx, item_id-1, response-1])  # 0-indexed
    return np.array(irt_data)

### 2. GRM probability calculation (vectorized)
# calculate the probability that each user will respond to each item in each category
def grm_probabilities(theta, alpha, beta, n_categories):
    n_users = theta.shape[0]
    n_items = alpha.shape[0]

    prob = np.zeros((n_users, n_items, n_categories))
    prob[:, :, 0] = 1.0

    # calculate cumulative probabilities
    for c in range(1, n_categories):
        z = alpha[None, :] * (theta[:, None] - beta[:, c-1])
        prob[:, :, c] = 1 / (1 + np.exp(-z))
    
    # category probability
    cat_probs = np.zeros_like(prob)
    cat_probs[:, :, 0] = prob[:, :, 0] - prob[:, :, 1]

    # calculate category probabilities
    for c in range(1, n_categories-1):
        cat_probs[:, :, c] = prob[:, :, c] - prob[:, :, c+1]
    cat_probs[:, :, n_categories-1] = prob[:, :, n_categories-1]

    return np.clip(cat_probs, 1e-10, 1-1e-10) # 수치 안정성을 위해 확률을 1e-10과 1-1e-10 사이로 클리핑


### 3. NLL (loss function)
# negative log likelihood function, calculate the negative log likelihood of the observed response data in the current parameter setting
# the goal is to find the parameter combination that best explains the observed response data by estimating the user's ability (theta), the discrimination of the item (alpha), and the difficulty of the category boundary (beta)
def grm_nll(params, responses, n_users, n_items, n_categories):
    
    # extract parameters
    theta = params[:n_users] # user ability parameter
    alpha = params[n_users:n_users+n_items] # item discrimination parameter
    beta = params[n_users+n_items:].reshape((n_items, n_categories-1)) # item difficulty parameter (category boundary)
    
    # extract actual response data
    user_idx = responses[:, 0].astype(int)
    item_idx = responses[:, 1].astype(int)
    actual_resp = responses[:, 2].astype(int)

    # calculate probabilities and return log likelihood
    probs = grm_probabilities(theta, alpha, beta, n_categories) # calculate the probability of all user-item-category combinations
    selected_probs = probs[user_idx, item_idx, actual_resp] # select the probability of the actual response
    return -np.sum(np.log(selected_probs)) # calculate the negative log likelihood (to minimize)


### 4. model execution
def run_grm_model(df_IRT, n_categories=3): # categories = number of response categories
    print("preparing data...")
    irt_data = prepare_irt_data(df_IRT)
    user_ids = np.unique(irt_data[:, 0]) # [0,0,0,1,1,1...,8263,8263,8263] -> [0,1,..,8263]
    item_ids = np.unique(irt_data[:, 1]) # [0,1,2,0,1,2,...] -> [0,1,2]
    n_users = len(user_ids) # 8264
    n_items = len(item_ids) # 3
    print(f"initializing parameters: {n_users} users, {n_items} items, {n_categories} categories")

    # initial parameters
    n_params = n_users + n_items + n_items*(n_categories-1) # each item needs n-1 beta parameters, so we subtract one
    init_params = np.zeros(n_params) # initialize all parameters to 0
    init_params[:n_users] = np.random.normal(0, 1, n_users) # initialize user parameters to random normal values with mean 0 and standard deviation 1
    init_params[n_users:n_users+n_items] = 1.0 # initialize item parameters to 1.0
    init_params[n_users+n_items:] = np.linspace(-1, 1, n_items*(n_categories-1)) # initialize category boundary parameters to evenly distributed values between -1 and 1

    # parameter constraints, currently (None,None)
    # the minimum value of 0.2 ensures that the item has some discrimination
    bounds = [(None, None)]*n_users + [(0.2, None)]*n_items + [(None, None)]*(n_items*(n_categories-1))

    # use SciPy's minimize function to estimate model parameters by minimizing the negative log-likelihood

    print(f"{model_name} model estimation started...")
    start_time = time.time()
    result = minimize(
        grm_nll, # the objective function to minimize, calculate the negative log-likelihood
        init_params, # the initial parameters
        args=(irt_data, n_users, n_items, n_categories), # additional arguments to pass to the objective function (response data, number of users, number of categories)
        method='L-BFGS-B', # use the limited memory BFGS algorithm, which is efficient for large-scale data
        bounds=bounds, # apply parameter constraints
        options={'maxiter': 500, 'disp': True} # optimization process settings, maximum 500 iterations and display intermediate results
    )
    end_time = time.time()
    print(f"{model_name} model estimation completed! time: {end_time - start_time:.2f} seconds")

    # summarize results
    theta_estimates = result.x[:n_users]
    alpha_estimates = result.x[n_users:n_users+n_items]
    beta_estimates = result.x[n_users+n_items:].reshape((n_items, n_categories-1))

    theta_df = pd.DataFrame({'user_id': user_ids, 'theta': theta_estimates})
    item_df = pd.DataFrame({'item_id': item_ids, 'alpha': alpha_estimates})
    for j in range(n_categories-1):
        item_df[f'beta_{j+1}'] = beta_estimates[:, j]

    
    # calculate model fit
    n_params = len(result.x)  # number of parameters (theta + alpha + beta)
    n_obs = len(irt_data)     # total number of responses
    nll = result.fun          # minimized negative log-likelihood

    aic = 2 * n_params + 2 * nll
    bic = np.log(n_obs) * n_params + 2 * nll

    print("\n📊 model fit metrics")
    print(f"- total number of responses: {n_obs}")
    print(f"- number of parameters: {n_params}")
    print(f"- -2 Log Likelihood: {2 * nll:.2f}")
    print(f"- AIC: {aic:.2f}")
    print(f"- BIC: {bic:.2f}")


    # save results
    today = datetime.now().strftime('%Y-%m-%d')
    save_dir = f"data/results/LLM/{model_name}_IRT_GRM_{today}"
    os.makedirs(save_dir, exist_ok=True)
    theta_df.to_csv(f"{save_dir}/theta_LLM_grm_robustness2.csv", index=False)
    item_df.to_csv(f"{save_dir}/item_parameters_LLM_grm_robustness2.csv", index=False)

    print(f"\n✅ {model_name} GRM estimation results saved! location: {save_dir}")
    return theta_df, item_df, result

### 5. example execution
# theta_df, item_df, result = run_grm_model(df_IRT)


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

date = '2025-05-17'


for model_name in models:
    # load CSV (modify the local path as needed)
    df = pd.read_csv(f'01_data/ANES_{model_name}_{date}.csv')
    print("file loaded")
    
    #### re-categorize based on .map ####

    # 1. gender (Binary)
    df['gender_encoded'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['gender_group'] = df['gender'].map({'Male': 'Male', 'Female': 'Female'})

    # 2. education level
    df['education_group'] = df['education'].map({
        'Less than high school': 'Low',
        'High school': 'Low',
        'Some post-high school': 'Mid',
        'Bachelor\'s degree': 'High',
        'Graduate degree': 'High'
    })

    # 3. political ideology
    df['ideology_group'] = df['political_ideology'].map({
        'Extremely liberal': 'Liberal',
        'Liberal': 'Liberal',
        'Slightly liberal': 'Liberal',
        'Moderate': 'Moderate',
        "Haven't thought much about this": 'Moderate',
        'Slightly conservative': 'Conservative',
        'Conservative': 'Conservative',
        'Extremely conservative': 'Conservative'
    })

    # 4. age group
    df['age_encoded'] = df['age'].map(lambda x: 1 if x < 35 else 2 if x < 60 else 3)
    df['age_group'] = df['age'].map(lambda x: 'Under 35' if x < 35 else '35~59' if x < 60 else 'Over 60')

    # 5. income group
    income_mapping = {}
    for val in ['Under $9,999', '$10,000-14,999', '$15,000-19,999', '$20,000-24,999', '$25,000-29,999',
                '$30,000-34,999', '$35,000-39,999', '$40,000-44,999', '$45,000-49,999']:
        income_mapping[val] = 'Low'
    for val in ['$50,000-59,999', '$60,000-64,999', '$65,000-69,999',
                '$70,000-74,999', '$75,000-79,999', '$80,000-89,999', '$90,000-99,999']:
        income_mapping[val] = 'Middle'
    for val in ['$100,000-109,999', '$110,000-124,999', '$125,000-149,999', '$150,000-174,999',
                '$175,000-249,999', '$250,000 or more']:
        income_mapping[val] = 'High'

    df['income_group'] = df['household_income'].map(income_mapping)

    # 6. marital status
    df['marital_group'] = df['marital_status'].map({
        'Married: spouse present': 'Married',
        'Married: spouse absent': 'Previously married',
        'Widowed': 'Previously married',
        'Divorced': 'Previously married',
        'Separated': 'Previously married',
        'Never married': 'Never married'
    })

    # 7. religion group
    df['religion_group'] = df['religion'].map({
        'Protestant': 'Western religion',
        'Roman Catholic': 'Western religion',
        'Orthodox Christian': 'Western religion',
        'Latter-Day Saints': 'Western religion',
        'Jewish': 'Non-Western religion',
        'Muslim': 'Non-Western religion',
        'Buddhist': 'Non-Western religion',
        'Hindu': 'Non-Western religion',
        'Atheist': 'Non-religious',
        'Agnostic': 'Non-religious',
        'Nothing in particular': 'Non-religious',
        'Something else': 'Other religion'
    })

    # 8. race group
    df['race_group'] = df['race'].map({
        'White': 'White',
        'Black': 'Black',
        'Asian': 'Asian',
        'Native American': 'Other',
        'Other': 'Other'
    })
    
    
    df_clean = df.dropna(subset=[
    'gender_group', 'education_group', 'ideology_group', 'age_group',
    'income_group', 'marital_group', 'religion_group', 'race_group'
    ])
    
    # apply GRM model
    df_selected = df_clean[['gun_control', 'welfare', 'immigration', 'transgender', 
                            # 'govt_treat_race' 제거
                            ]].dropna()
    theta_df, item_df, result = run_grm_model(df_selected)
    
    # combine theta values
    # reset index before combining theta values
    df_clean = df_clean.reset_index(drop=True)

    # combine theta values
    df_clean['theta'] = theta_df['theta'].values
    
    # save results
    today = datetime.now().strftime('%Y-%m-%d')
    save_dir = f"data/results/LLM/{model_name}_IRT_GRM_{today}"
    df_clean.to_csv(f'{save_dir}/theta_{model_name}_robustness2.csv',index=False)