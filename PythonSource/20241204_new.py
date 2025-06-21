import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler

# 데이터 로드
file_path = "D:\\Data\\20241204_NEW\\20241204_NEW_DATA_2.csv"
data = pd.read_csv(file_path)

# 'TRAVEL_MISSION_INT' 생성
data.loc[:, 'TRAVEL_MISSION_INT'] = data['TRAVEL_MISSION_CHECK'].str.split(';').str[0].astype(int)

# 필요한 컬럼만 선택
data = data.loc[:, [
    'GENDER',
    'AGE_GRP',
    'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
    'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
    'TRAVEL_MOTIVE_1', 'TRAVEL_MOTIVE_2', 'TRAVEL_MOTIVE_3',
    'TRAVEL_COMPANIONS_NUM',
    'TRAVEL_MISSION_INT',
    'DGSTFN', "REVISIT_INTENTION", "RCMDTN_INTENTION",
    'TEMPERATURE', 'PRECIPITATION', 'SUNSHINE_HOURS'
]]

# 결측치 제거 후 복사본 생성
data_cleaned = data.dropna().copy()

numerical_features = ['TEMPERATURE', 'PRECIPITATION', 'SUNSHINE_HOURS']
scaler = StandardScaler()
data_cleaned[numerical_features] = scaler.fit_transform(data_cleaned[numerical_features])

# 범주형 피처
categorical_features_names = [
    'GENDER',
    'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
    'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
    'TRAVEL_MOTIVE_1', 'TRAVEL_MOTIVE_2', 'TRAVEL_MOTIVE_3',
    'TRAVEL_MISSION_INT',
    "REVISIT_INTENTION", "RCMDTN_INTENTION"
]

# 범주형 데이터를 문자열로 변환 및 결측치 처리
for col in categorical_features_names:
    if col in data_cleaned.columns:
        data_cleaned[col] = data_cleaned[col].fillna('Unknown').astype(str)

# 데이터 분할
train_data, test_data = train_test_split(data_cleaned, test_size=0.2, random_state=42)

# CatBoost Pool 생성
train_pool = Pool(
    data=train_data.drop(['DGSTFN'], axis=1),
    label=train_data['DGSTFN'],
    cat_features=[f for f in categorical_features_names if f in train_data.columns]
)

test_pool = Pool(
    data=test_data.drop(['DGSTFN'], axis=1),
    label=test_data['DGSTFN'],
    cat_features=[f for f in categorical_features_names if f in test_data.columns]
)

# 모델 학습
model = CatBoostRegressor(
    loss_function='Huber:delta=1.0',
    eval_metric='RMSE',
    task_type='GPU',
    depth=8,
    learning_rate=0.01,
    n_estimators=2000,
    early_stopping_rounds=100  # 성능 개선 없을 시 조기 종료
)

# 모델 학습 및 평가
model.fit(
    train_pool,
    eval_set=test_pool,
    verbose=500
)

# 학습 및 평가 결과 시각화
eval_metrics = model.get_evals_result()
plt.figure(figsize=(10, 5))
plt.plot(eval_metrics['learn']['RMSE'], label='Train RMSE')
plt.plot(eval_metrics['validation']['RMSE'], label='Validation RMSE')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()
