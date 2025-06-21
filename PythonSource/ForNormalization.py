import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# CSV 파일 경로
file_path = r"D:\Data\Using\CSV_TEST.csv"

# CSV 파일 읽기
data = pd.read_csv(file_path)

# TEMPERATURE 컬럼의 데이터 표준화
temperature_data = data[['TEMPERATURE']]
scaler = StandardScaler()
standardized_temperature = scaler.fit_transform(temperature_data)

# 표준화된 데이터를 히스토그램으로 시각화
plt.figure(figsize=(10, 6))
sns.histplot(standardized_temperature, kde=True)
plt.title('Standardized Temperature Distribution')
plt.xlabel('Standardized Temperature')
plt.ylabel('Frequency')
plt.grid(True)

# 그래프 표시
plt.show()
