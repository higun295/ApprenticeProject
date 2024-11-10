import os
import csv

# 파일 경로 지정
input_folder = r"D:\Data\Training\TL_gps_data"
output_file = r"D:\Data\Training\combined_data.csv"

# 모든 CSV 파일 경로를 리스트로 가져오기
csv_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')]
total_files = len(csv_files)

# 첫 번째 파일의 헤더를 읽고 파일 생성
with open(output_file, mode='w', newline='', encoding='utf-8') as fout:
    writer = None
    for i, file in enumerate(csv_files):
        with open(file, mode='r', newline='', encoding='utf-8') as fin:
            reader = csv.reader(fin)
            if i == 0:  # 첫 번째 파일은 헤더를 포함하여 작성
                writer = csv.writer(fout)
                writer.writerow(next(reader))  # 헤더 작성
            else:  # 두 번째 파일부터는 헤더를 건너뜀
                next(reader)

            # 데이터 작성
            writer.writerows(reader)

        print(f"총 {total_files}개의 파일 중 {i + 1}번째 파일을 처리 완료.")

print(f"모든 파일이 {output_file}에 성공적으로 합쳐졌습니다.")
