import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)


path = '/Users/jun/GitStudy/commerceAB/data/practice_data/'


# -----------------------------
# 1. 초기 A 그룹 데이터 생성 (Before)
# -----------------------------
N = 2000
start_date = datetime(2025, 10, 1)
dates = [start_date + timedelta(days=int(x)) for x in np.random.uniform(0, 45, N)]

user_id = np.arange(10000, 10000 + N)
age = np.random.randint(18, 66, size=N)
gender = np.random.choice(['M', 'F'], size=N, p=[0.52, 0.48])
region = np.random.choice(['North', 'South', 'East', 'West'], size=N)
sessions = np.random.poisson(3, size=N) + 1

# KPI: conversion, revenue (baseline)
logodds = -2.0 + (age - 40) * (-0.01) + (sessions - 3) * 0.15 + (gender == 'F')*0.05
p_control = 1 / (1 + np.exp(-logodds))
conversion = np.random.binomial(1, p_control)
revenue = conversion * (np.round(np.random.exponential(scale=30, size=N) + 5, 2))

df_before = pd.DataFrame({
    'user_id': user_id,
    'group': 'A',
    'signup_date': [d.strftime('%Y-%m-%d') for d in dates],
    'age': age,
    'gender': gender,
    'region': region,
    'sessions_last_7d': sessions,
    'conversion': conversion,
    'revenue': revenue
})



df_before.to_csv(path+'/ab_before.csv', index=False)

# -----------------------------
# 2. A/B Test용 After 데이터 생성
# -----------------------------
df_after = df_before.copy()
df_after['group'] = np.random.choice(['A', 'B'], size=N, p=[0.5, 0.5])

# Treatment effect: B 그룹만 적용, 다양한 특징 삽입
treatment_effect = np.zeros(N)
for i in range(N):
    d = dates[i]
    
    # 1) 가입일 시기별 차이: 2025-10-25 이후 가입자만 B에서 효과 상승
    if d >= datetime(2025, 10, 25):
        treatment_effect[i] += 0.12

    # 2) 연령/성별 상호작용
    if age[i] < 30 and gender[i] == 'F':
        treatment_effect[i] += 0.20  # 젊은 여성에게 강한 효과
    elif age[i] > 55 and gender[i] == 'M':
        treatment_effect[i] -= 0.10  # 나이 많은 남성에게 부정 효과

    # 3) 세션수 기반 증폭
    if sessions[i] >= 6:
        treatment_effect[i] += 0.05

    # 4) 지역별 효과: 특정 지역에서 B 효과 강화
    if region[i] == 'South':
        treatment_effect[i] += 0.08
    elif region[i] == 'East':
        treatment_effect[i] -= 0.05

logodds_treated = logodds + treatment_effect * (df_after['group'] == 'B')
p_treated = 1 / (1 + np.exp(-logodds_treated))
df_after['conversion'] = np.random.binomial(1, p_treated)
df_after['revenue'] = df_after['conversion'] * (np.round(np.random.exponential(scale=30, size=N) + 5, 2))

df_after.to_csv(path+'/ab_after.csv', index=False)

print("데이터셋 준비 완료:")
print("1) AA 테스트용 Before CSV: aa_test_before.csv")
print("2) AB 테스트용 After CSV: ab_test_after.csv")
