import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import ttk

# 데이터 로드
file_path = "D:\\Data\\20241130_2_데이터전처리.csv"
data = pd.read_csv(file_path)

# 필요한 컬럼만 선택
data = data.loc[:, [
    'GENDER',
    'AGE_GRP',
    'REVISIT_YN', 'DGSTFN', 'REVISIT_INTENTION', 'VISIT_AREA_TYPE_CD',
    'STORE_NM', 'ROAD_NM_ADDR', 'LOTNO_ADDR', 'PAYMENT_DT',
    'STATION_ID', 'STATION_NAME', 'TEMPERATURE', 'PRECIPITATION', 'SUNSHINE_HOURS'
]]

# 결측치 제거 후 복사본 생성
data_cleaned = data.dropna().copy()

# STATION_NAME과 STATION_ID 매핑
station_map = {'성산': 188, '고산': 185, '서귀포': 189, '제주': 184}
data_cleaned = data_cleaned[data_cleaned['STATION_ID'].isin(station_map.values())]

# 데이터에서 고유한 STORE_NM 값을 불러와 매핑 생성
unique_store_names = data_cleaned['STORE_NM'].unique()
store_names_map = {idx: name for idx, name in enumerate(unique_store_names)}

# 라벨 컬럼 추가 (STORE_NM을 숫자 라벨로 변환)
data_cleaned['STORE_NM_LABEL'] = data_cleaned['STORE_NM'].map({v: k for k, v in store_names_map.items()})

# 간단한 전처리 (GENDER, AGE_GRP 인코딩)
data_cleaned['GENDER'] = data_cleaned['GENDER'].map({'남': 1, '여': 0})  # 성별을 숫자로 변환
data_cleaned['AGE_GRP'] = data_cleaned['AGE_GRP'].astype(int)

# VISIT_AREA_TYPE_CD를 숫자로 인코딩
visit_area_type_map = {cd: idx for idx, cd in enumerate(data_cleaned['VISIT_AREA_TYPE_CD'].unique())}
data_cleaned['VISIT_AREA_TYPE_CD'] = data_cleaned['VISIT_AREA_TYPE_CD'].map(visit_area_type_map)

# 피쳐와 라벨 분리
X = data_cleaned[['GENDER', 'AGE_GRP', 'TEMPERATURE', 'PRECIPITATION', 'SUNSHINE_HOURS', 'VISIT_AREA_TYPE_CD']]
y = data_cleaned['STORE_NM_LABEL']

# 학습용, 테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# RandomForest 모델 학습
print("RandomForest 모델 학습을 시작합니다...")
model = RandomForestClassifier(
    n_estimators=500,  # 트리 개수
    max_depth=20,  # 최대 깊이
    random_state=42,
    class_weight='balanced'  # 클래스 불균형 보정
)
model.fit(X_train, y_train)
print("RandomForest 모델 학습이 완료되었습니다.")

# GUI 생성
root = tk.Tk()
root.title("Store Recommendation System")

# Gender mapping for conversion to numeric
gender_map = {'여': 0, '남': 1}

# 추천 함수
def recommend_place():
    # 입력 값 가져오기
    gender = gender_var.get()
    age_group = int(age_group_var.get())
    temperature = float(temperature_var.get())
    precipitation = float(precipitation_var.get())
    sunshine = float(sunshine_var.get())
    visit_area_type = visit_area_type_var.get()
    station_name = station_var.get()

    # Convert gender to numeric value
    gender_numeric = gender_map.get(gender)
    if gender_numeric is None:
        return

    # Convert visit_area_type to numeric
    visit_area_numeric = visit_area_type_map.get(visit_area_type)
    if visit_area_numeric is None:
        return

    # Get selected station_id
    station_id = station_map.get(station_name)
    if station_id is None:
        return

    # Prepare input data
    input_data = pd.DataFrame([[gender_numeric, age_group, temperature, precipitation, sunshine, visit_area_numeric]],
                              columns=['GENDER', 'AGE_GRP', 'TEMPERATURE', 'PRECIPITATION', 'SUNSHINE_HOURS', 'VISIT_AREA_TYPE_CD'])

    # Get the probability distribution for each class
    probas = model.predict_proba(input_data)

    # Convert probabilities into a DataFrame for easier display
    probas_df = pd.DataFrame(probas, columns=model.classes_)
    probas_df_sorted = probas_df.T.sort_values(by=0, ascending=False).reset_index()
    probas_df_sorted.columns = ['Store Label', 'Recommendation Score']
    probas_df_sorted['Store Name'] = probas_df_sorted['Store Label'].map(store_names_map)

    # Scale recommendation scores for better visibility
    probas_df_sorted['Recommendation Score'] *= 100

    # Filter by STATION_ID and score > 0
    probas_df_filtered = probas_df_sorted[probas_df_sorted['Recommendation Score'] > 0]
    probas_df_filtered = probas_df_filtered[probas_df_filtered['Store Label'].map(
        lambda x: data_cleaned[data_cleaned['STORE_NM_LABEL'] == x]['STATION_ID'].iloc[0] == station_id)]

    # Clear tree view
    for row in tree.get_children():
        tree.delete(row)

    # Add results to tree view
    for idx, row in probas_df_filtered.iterrows():
        store_name = row['Store Name']
        road_addr = data_cleaned[data_cleaned['STORE_NM_LABEL'] == row['Store Label']]['ROAD_NM_ADDR'].iloc[0]
        avg_satisfaction = data_cleaned[data_cleaned['STORE_NM_LABEL'] == row['Store Label']]['DGSTFN'].mean()
        avg_satisfaction = round(avg_satisfaction, 2)
        tree.insert("", "end", values=(store_name, station_name, road_addr, avg_satisfaction))

# 입력 필드 생성
tk.Label(root, text="성별 (남/여):").grid(row=0, column=0)
gender_var = tk.StringVar(value="남")
tk.Entry(root, textvariable=gender_var).grid(row=0, column=1)

tk.Label(root, text="나이대 (10~70):").grid(row=1, column=0)
age_group_var = tk.StringVar(value="30")
tk.Entry(root, textvariable=age_group_var).grid(row=1, column=1)

tk.Label(root, text="기온 (-10~40):").grid(row=2, column=0)
temperature_var = tk.StringVar(value="20")
tk.Entry(root, textvariable=temperature_var).grid(row=2, column=1)

tk.Label(root, text="강수량 (0~200):").grid(row=3, column=0)
precipitation_var = tk.StringVar(value="0")
tk.Entry(root, textvariable=precipitation_var).grid(row=3, column=1)

tk.Label(root, text="일조시간 (0~1):").grid(row=4, column=0)
sunshine_var = tk.StringVar(value="0.5")
tk.Entry(root, textvariable=sunshine_var).grid(row=4, column=1)

tk.Label(root, text="활동유형:").grid(row=5, column=0)
visit_area_type_var = tk.StringVar(value=list(visit_area_type_map.keys())[0])
tk.OptionMenu(root, visit_area_type_var, *visit_area_type_map.keys()).grid(row=5, column=1)

tk.Label(root, text="위치:").grid(row=6, column=0)
station_var = tk.StringVar(value=list(station_map.keys())[0])
tk.OptionMenu(root, station_var, *station_map.keys()).grid(row=6, column=1)

# TreeView 생성
columns = ('상점명', '지역', '도로명 주소', '만족도')
tree = ttk.Treeview(root, columns=columns, show='headings')
for col in columns:
    tree.heading(col, text=col)
tree.grid(row=8, column=0, columnspan=2)

# 추천 버튼
tk.Button(root, text="추천 받기", command=recommend_place).grid(row=7, column=0, columnspan=2)

# GUI 실행
root.mainloop()
