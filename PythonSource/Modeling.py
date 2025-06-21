import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

# 1. 데이터 로드
file_path = r"D:\Data\20241130_2_데이터전처리.csv"
data = pd.read_csv(file_path)

# 2. 필요한 컬럼 선택
# 가정: 데이터에는 아래 컬럼이 포함되어 있음
columns_to_use = ['GENDER', 'AGE_GRP', 'TEMPERATURE', 'PRECIPITATION', 'SUNSHINE_HOURS']
target_column = 'RCMDTN_INTENTION'  # 추천 의향을 목표 변수로 설정

# 3. 독립 변수와 종속 변수 분리
X = data[columns_to_use]
y = data[target_column]

# 4. 범주형 데이터 처리 (One-Hot Encoding)
categorical_columns = ['GENDER', 'AGE_GRP']  # 범주형 변수
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_features = encoder.fit_transform(X[categorical_columns])

# 수치형 데이터 스케일링 (기온, 강수, 일조량 등)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X[['TEMPERATURE', 'PRECIPITATION', 'SUNSHINE_HOURS']])

# 독립 변수 결합
X_processed = pd.concat([
    pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns)),
    pd.DataFrame(scaled_features, columns=['TEMPERATURE', 'PRECIPITATION', 'SUNSHINE_HOURS'])
], axis=1)

# 5. 종속 변수 처리 (One-Hot Encoding)
y_processed = to_categorical(y)

# 6. 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)

# 데이터 준비 완료
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
