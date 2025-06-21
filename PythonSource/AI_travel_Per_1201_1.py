# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 16:57:22 2024

@author: roads
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # 결측치 처리
    weather_columns = ['TEMPERATURE', 'PRECIPITATION', 'SUNSHINE_HOURS', 'SNOW_DEPTH', 'CLOUD_COVER']
    df[weather_columns] = df[weather_columns].fillna(0)
    
    address_columns = ['VISIT_AREA_ID', 'ROAD_NM_ADDR']
    df[address_columns] = df[address_columns].fillna('')
    
    return df

def normalize_weather_scores(input_value, data_values):
    normalized_values = np.abs(data_values - input_value)
    max_value = np.max(normalized_values)
    if max_value == 0:
        return np.ones(len(normalized_values))
    return 1 - (normalized_values / max_value)

def get_visit_count(df):
    return df['LOTNO_CD'].value_counts()

def recommend_places(input_temperature, input_precipitation, input_sunshine, 
                     input_gender, input_age_group, input_location, 
                     temp_weight, precip_weight, sunshine_weight, df):
    
    # 1. 위치 기반 필터링
    location_filter = df[df['ROAD_NM_ADDR'].str.contains(input_location, na=False)]
    if location_filter.empty:
        return []
    
    road_codes = location_filter['ROAD_NM_CD'].astype(str).str[:6]
    lot_codes = location_filter['LOTNO_CD'].astype(str).str[:6]
    selected_road_code = road_codes.mode().iloc[0]
    selected_lot_code = lot_codes.mode().iloc[0]
    
    filtered_df = df[(df['ROAD_NM_CD'].astype(str).str[:6] == selected_road_code) & 
                     (df['LOTNO_CD'].astype(str).str[:6] == selected_lot_code)]
    
    # 2. 성별 및 나이대 필터링
    filtered_df = filtered_df[(filtered_df['GENDER'] == input_gender) & 
                              (filtered_df['AGE_GRP'] == input_age_group)]
    
    # 날씨 점수 계산 및 정규화
    filtered_df['temperature_score'] = normalize_weather_scores(input_temperature, filtered_df['TEMPERATURE'])
    filtered_df['precipitation_score'] = normalize_weather_scores(input_precipitation, filtered_df['PRECIPITATION'])
    filtered_df['sunshine_score'] = normalize_weather_scores(input_sunshine, filtered_df['SUNSHINE_HOURS'])
    
    # 가중치 적용
    filtered_df['weather_score'] = (
        filtered_df['temperature_score'] * temp_weight +
        filtered_df['precipitation_score'] * precip_weight +
        filtered_df['sunshine_score'] * sunshine_weight
    )
    
    # 몇번 방문했는지 카운트
    visit_counts = get_visit_count(filtered_df)
    filtered_df['visit_count'] = filtered_df['LOTNO_CD'].map(visit_counts)
    
    
    # 최종 점수 계산
    filtered_df['final_score'] = filtered_df['DGSTFN'] * filtered_df['weather_score']
    
    # 중복되는 추천지는 제거
    filtered_df = filtered_df.drop_duplicates(subset=['STORE_NM'])
    

    # 상위 5개 장소 추천
    recommendations = filtered_df.nlargest(5, 'final_score')[['STORE_NM', 'ROAD_NM_ADDR', 'ROAD_NM_CD', 'LOTNO_CD', 'visit_count', 'final_score']]
    return recommendations

# 메인 프로그램
if __name__ == "__main__":
    file_path = r"D:\Data\20241130_2_데이터전처리.csv"
    df = load_and_preprocess_data(file_path)
    
    # 사용자 입력
    input_temperature = float(input("현재 온도를 입력하세요: "))
    input_precipitation = float(input("현재 강수량을 입력하세요: "))
    input_sunshine = float(input("현재 일조시간을 입력하세요: "))
    input_gender = input("성별을 입력하세요 (남/여): ")
    input_age_group = int(input("나이대를 입력하세요: "))
    input_location = input("현재 위치를 입력하세요: ")
    
    temp_weight = float(input("온도의 가중치를 입력하세요 (0-1): "))
    precip_weight = float(input("강수량의 가중치를 입력하세요 (0-1): "))
    sunshine_weight = float(input("일조시간의 가중치를 입력하세요 (0-1): "))
    
    recommendations = recommend_places(
        input_temperature, input_precipitation, input_sunshine,
        input_gender, input_age_group, input_location,
        temp_weight, precip_weight, sunshine_weight, df
    )
    
    print("\n추천 장소:")
    for idx, (name, address, raod_nm_code, lotno_code, visit_count, score) in enumerate(recommendations.values, 1):
        print(f"{idx}. {name}")
        print(f"   주소: {address}")
        print(f"   도로명코드: {raod_nm_code}")
        print(f"   지번코드: {lotno_code}")
        print(f"   방문 횟수: {visit_count}회")
        print(f"   추천 점수: {score:.2f}\n")
