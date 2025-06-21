import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 데이터 로드
file_path = r"D:\Data\20241130_2_데이터전처리.csv"
df = pd.read_csv(file_path)

# 필요한 컬럼 선택
categorical_columns = ['GENDER', 'AGE_GRP', 'VISIT_AREA_TYPE_CD']
target_column = 'DGSTFN'  # 만족도

# 결측치 처리
df = df[categorical_columns + [target_column]].dropna()

# 범주형 데이터 전처리 (One-Hot Encoding)
encoder = OneHotEncoder(sparse=False, drop='first')
X = encoder.fit_transform(df[categorical_columns])
y = df[target_column]

# 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost 모델 학습
model = XGBRegressor()
model.fit(X_train, y_train)

# 변수 중요도 시각화
feature_importance = model.feature_importances_
plt.barh(encoder.get_feature_names_out(), feature_importance)
plt.title("Feature Importance")
plt.show()
