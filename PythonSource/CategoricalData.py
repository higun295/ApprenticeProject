import pandas as pd
import scipy.stats as stats

# 데이터 로드
file_path = r"D:\Data\20241130_2_데이터전처리.csv"
data = pd.read_csv(file_path)

# 필요한 컬럼 선택 (범주형 변수 + 종속 변수)
columns_to_use = ['GENDER', 'AGE_GRP', 'VISIT_AREA_TYPE_CD', 'DGSTFN', 'RCMDTN_INTENTION', 'REVISIT_INTENTION']
df = data[columns_to_use]

# 결측치 처리
df = df.dropna()

# 범주형 변수 선택
categorical_columns = ['GENDER', 'AGE_GRP', 'VISIT_AREA_TYPE_CD']

# 종속 변수
dependent_variables = ['DGSTFN', 'RCMDTN_INTENTION', 'REVISIT_INTENTION']


# 범주형 변수와 종속 변수 간의 연관성 분석 함수
def analyze_categorical_association(df, categorical_columns, dependent_variables):
    results = []

    for dep_var in dependent_variables:
        print(f"\n=== Analyzing associations with {dep_var} ===")
        for cat_col in categorical_columns:
            # 범주형 변수와 종속 변수 간의 관계 분석 (ANOVA 또는 카이제곱)
            if df[dep_var].nunique() > 5:  # 연속형에 가까운 경우 ANOVA 사용
                grouped_data = [group[dep_var].values for _, group in df.groupby(cat_col)]
                f_stat, p_value = stats.f_oneway(*grouped_data)
                method = 'ANOVA'
            else:  # 범주형 종속 변수에는 카이제곱 검정 사용
                contingency_table = pd.crosstab(df[cat_col], df[dep_var])
                chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
                f_stat = chi2
                method = 'Chi-Square'

            # 결과 저장
            results.append({'Dependent Variable': dep_var,
                            'Categorical Variable': cat_col,
                            'Test': method,
                            'Statistic': f_stat,
                            'P-value': p_value})
            print(f"Variable: {cat_col}, Method: {method}, Statistic: {f_stat:.2f}, P-value: {p_value:.4f}")

    return pd.DataFrame(results)


# 연관성 분석 수행
results_df = analyze_categorical_association(df, categorical_columns, dependent_variables)
