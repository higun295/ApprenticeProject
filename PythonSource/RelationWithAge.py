import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 파일 경로
file_path = r"D:\Data\20241203_데이터전처리_AGE_20.csv"

# 데이터 로드
data = pd.read_csv(file_path, names=['AGE_GRP', 'ROAD_NM_CD', 'VISIT_COUNT'])

# 연령대와 방문 횟수를 숫자로 변환
data['AGE_GRP'] = pd.to_numeric(data['AGE_GRP'], errors='coerce')
data['VISIT_COUNT'] = pd.to_numeric(data['VISIT_COUNT'], errors='coerce')

# 1. AGE_GRP가 20인 데이터 필터링
filtered_data = data[data['AGE_GRP'] == 20]

# 도로명주소코드(ROAD_NM_CD)를 범주형 숫자로 변환
data['ROAD_NM_CD'] = pd.Categorical(data['ROAD_NM_CD']).codes

# 2. 다변수 상관관계 계산
correlation_matrix = data.corr()

# 상관관계 출력
print("Correlation Matrix for AGE_GRP = 20:")
print(correlation_matrix)

# 3. 상관관계 히트맵 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, fmt='.2f')
plt.title("Correlation Heatmap for AGE_GRP = 20")
plt.show()

# 4. 다변수 산점도 매트릭스 (쌍변수 관계 시각화)
sns.pairplot(data)
plt.suptitle("Pairplot for AGE_GRP = 20")
plt.show()
