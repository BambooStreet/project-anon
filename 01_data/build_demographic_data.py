import pandas as pd

# load CSV file
df = pd.read_csv("ANES/anes_timeseries_2020_csv_20220210/anes_timeseries_2020_csv_20220210.csv")

# add analysis target variables
cols = [
    'V201600',     # gender
    'V201511x',    # education
    'V201507x',    # age
    'V201200',     # political ideology
    'V201617x',    # household income
    'V201508',     # marital status
    'V201435',     # religion
    'V201549x',    # race
]

# select and remove missing values
df_demographic = df[cols].dropna()

# rename variables
df_demographic = df_demographic.rename(columns={
    'V201600': 'gender',
    'V201511x': 'education',
    'V201507x': 'age',
    'V201200': 'political_ideology',
    'V201617x': 'household_income',
    'V201508': 'marital_status',
    'V201435': 'religion',
    'V201549x': 'race'
})


# save results
df_demographic.to_csv("raw_demographics.csv", index=False)