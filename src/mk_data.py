# 생성한 데이터셋을 CSV로 저장하고 일부를 표로 보여줍니다.
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N = 2000
start_date = datetime(2025, 10, 1)
dates = [start_date + timedelta(days=int(x)) for x in np.random.uniform(0, 45, N)]

user_id = np.arange(10000, 10000+N)
group = np.random.choice(['A', 'B'], size=N, p=[0.5, 0.5])
age = np.random.randint(18, 66, size=N)
gender = np.random.choice(['M', 'F'], size=N, p=[0.52, 0.48])
region = np.random.choice(['North','South','East','West'], size=N)
sessions = np.random.poisson(3, size=N) + 1

# baseline log-odds
logodds = -2.0 + (age - 40) * (-0.01) + (sessions - 3) * 0.15 + (gender == 'F')*0.05
# hidden heterogeneous treatment effects (embedded, not revealed here)
treatment_effect = np.zeros(N)
# create some non-linear/time-dependent/segment-dependent effects
for i in range(N):
    d = dates[i]
    # time bucket
    if d >= datetime(2025,10,25):
        treatment_effect[i] += 0.12  # increased effect later
    # age/gender interaction
    if age[i] < 35 and gender[i] == 'F':
        treatment_effect[i] += 0.18
    if age[i] > 55 and gender[i] == 'M':
        treatment_effect[i] -= 0.08
    # sessions amplify effect
    if sessions[i] >= 6:
        treatment_effect[i] += 0.05

# apply only to group B (treatment)
logodds_treated = logodds + treatment_effect * (group == 'B')

# convert to probabilities
p_control = 1 / (1 + np.exp(-logodds))
p_treated = 1 / (1 + np.exp(-logodds_treated))

conversion = np.random.binomial(1, np.where(group=='B', p_treated, p_control))
revenue = conversion * (np.round(np.random.exponential(scale=30, size=N) + 5, 2))

df = pd.DataFrame({
    'user_id': user_id,
    'group': group,
    'signup_date': [d.strftime('%Y-%m-%d') for d in dates],
    'age': age,
    'gender': gender,
    'region': region,
    'sessions_last_7d': sessions,
    'conversion': conversion,
    'revenue': revenue
})

# shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# save to CSV
path = '/Users/jun/GitStudy/commerceAB/data/practice_data/test_data.csv'
df.to_csv(path, index=False)