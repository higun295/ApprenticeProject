import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드 (파일 경로에 맞게 수정)
file_path = r"D:\Data\20241203_데이터전처리.csv"
data = pd.read_csv(file_path)

# 필요한 데이터 선택
columns_to_analyze = ['TEMPERATURE', 'AGE_GRP', 'GENDER', 'visit_count']
data_subset = data[columns_to_analyze]

# 연령대와 성별 데이터를 숫자로 변환 (상관계수 계산을 위해)
data_subset['AGE_GRP'] = pd.to_numeric(data_subset['AGE_GRP'], errors='coerce')  # 연령대를 숫자로
data_subset['GENDER'] = data_subset['GENDER'].map({'남': 0, '여': 1})  # 성별을 숫자로 매핑

# 데이터의 결측치 제거
data_subset = data_subset.dropna()

# 1. 상관관계 계산
correlation_matrix = data_subset.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# 2. 상관관계 히트맵 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation between Temperature, Age Group, Gender, and Visit Count")
plt.show()

# 3. 기온과 방문 횟수의 연령대/성별별 관계 시각화
sns.scatterplot(data=data_subset, x='TEMPERATURE', y='visit_count', hue='AGE_GRP', style='GENDER', palette='viridis')
plt.title("Visit Count vs. Temperature by Age Group and Gender")
plt.xlabel("Temperature (°C)")
plt.ylabel("Visit Count")
plt.show()
