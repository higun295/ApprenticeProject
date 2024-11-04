import chardet

# 읽을 파일 경로 설정
file_path = "D:/Data/Training/TL_cs/tn_activity_consume_his_활동소비내역_H.csv"

# 인코딩 감지
with open(file_path, 'rb') as f:
    result = chardet.detect(f.read())
    detected_encoding = result['encoding']
    print(f"감지된 인코딩: {detected_encoding}")

# 감지된 인코딩으로 파일 내용 출력
try:
    with open(file_path, 'r', encoding=detected_encoding) as f:
        content = f.read()
        print("파일 내용 미리보기:")
        print(content[:500])  # 처음 500자를 미리 보기로 출력
except Exception as e:
    print(f"오류 발생: {e}")
