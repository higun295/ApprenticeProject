import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 파일 경로
file_path = r"D:\Data\20241203_DATA_1.csv"

# 데이터 로드
data = pd.read_csv(file_path)

# 1. 상관관계 계산
correlation_matrix = data[['AVG_TEMPERATURE', 'AVG_PRECIPITATION', 'AVG_SUNSHINE_HOURS', 'VISIT_COUNT']].corr()

# 상관관계 출력
print("Correlation Matrix:")
print(correlation_matrix)

# 2. 히트맵 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# 3. 산점도 시각화: 온도와 방문 횟수
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='AVG_TEMPERATURE', y='VISIT_COUNT', hue='AVG_PRECIPITATION', palette='viridis', size='AVG_SUNSHINE_HOURS', sizes=(20, 200))
plt.title("Visit Count vs. Temperature")
plt.xlabel("Average Temperature (°C)")
plt.ylabel("Visit Count")
plt.legend(title="Precipitation")
plt.show()