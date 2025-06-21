import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# 데이터 로드
file_path = r"D:\Data\20241130_2_데이터전처리.csv"
data = pd.read_csv(file_path)

# 필요한 컬럼 선택
columns_to_use = ['GENDER', 'AGE_GRP', 'VISIT_AREA_TYPE_CD', 'DGSTFN', 'RCMDTN_INTENTION', 'REVISIT_INTENTION']
df = data[columns_to_use].dropna()

# 범주형 변수 선택
categorical_columns = ['GENDER', 'AGE_GRP', 'VISIT_AREA_TYPE_CD']
dependent_variables = ['DGSTFN', 'RCMDTN_INTENTION', 'REVISIT_INTENTION']

# Cramer's V 계산 함수
def cramers_v(contingency_table):
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    k = min(contingency_table.shape)
    return np.sqrt(chi2 / (n * (k - 1)))

# 독립 변수와 종속 변수 간의 Cramer's V 계산
results = []
for dep_var in dependent_variables:
    for cat_col in categorical_columns:
        contingency_table = pd.crosstab(df[cat_col], df[dep_var])
        cramer_v_value = cramers_v(contingency_table)
        results.append({'Dependent Variable': dep_var,
                        'Categorical Variable': cat_col,
                        'Cramer\'s V': cramer_v_value})

# 결과 출력
results_df = pd.DataFrame(results)
print(results_df)
