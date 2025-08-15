import os
import re
from pathlib import Path

def fix_date_order():
    """JSON 파일명의 날짜 순서를 수정"""
    
    labels_dir = Path(r"D:\8.간접업무자동화\1. PO시트입력자동화\YMFK_OCR\data\processed\labels")
    
    if not labels_dir.exists():
        print(f"폴더가 존재하지 않습니다: {labels_dir}")
        return
    
    json_files = list(labels_dir.glob("*.json"))
    
    if not json_files:
        print("JSON 파일이 없습니다.")
        return
    
    renamed_count = 0
    skipped_count = 0
    error_count = 0
    
    print(f"총 {len(json_files)}개의 JSON 파일 검사")
    print("=" * 60)
    
    # 패턴: 첫번째_날짜_두번째_날짜-나머지
    # 잘못된 형식: 20250808_110604_20240909165721-0001_page001_label.json
    # 올바른 형식: 20250807_175828_20240909162139-0001_page001_label.json
    # 즉, 두 날짜 부분을 서로 바꿔야 함
    
    pattern = r'^(\d{8}_\d{6})_(\d{14})-(.+)\.json$'
    
    for json_file in json_files:
        filename = json_file.name
        
        # 패턴 매칭
        match = re.match(pattern, filename)
        
        if match:
            first_date = match.group(1)   # 20250808_110604
            second_date = match.group(2)  # 20240909165721
            rest = match.group(3)          # 0001_page001_label
            
            # 첫번째 날짜가 2025로 시작하면 잘못된 순서
            if first_date.startswith('2025'):
                # 날짜 순서 바꾸기
                # 20250808_110604를 20250808110604로 변환
                first_date_clean = first_date.replace('_', '')
                # 20240909165721는 그대로
                
                # 새로운 파일명: 두번째날짜_첫번째날짜-나머지
                new_name = f"{second_date[:8]}_{second_date[8:]}_{first_date_clean}-{rest}.json"
                new_path = json_file.parent / new_name
                
                try:
                    if not new_path.exists():
                        json_file.rename(new_path)
                        print(f"변경 완료: {filename}")
                        print(f"       → {new_name}")
                        renamed_count += 1
                    else:
                        print(f"대상 파일이 이미 존재: {new_name}")
                        # 기존 파일 삭제
                        json_file.unlink()
                        print(f"  중복 파일 삭제: {filename}")
                        error_count += 1
                except Exception as e:
                    print(f"오류 발생: {filename} - {e}")
                    error_count += 1
            else:
                # 이미 올바른 순서
                skipped_count += 1
        else:
            # 예상하지 못한 패턴
            print(f"패턴 불일치: {filename}")
            skipped_count += 1
    
    print("=" * 60)
    print(f"작업 완료!")
    print(f"  변경된 파일: {renamed_count}개")
    print(f"  건너뛴 파일: {skipped_count}개")
    print(f"  오류/중복: {error_count}개")

if __name__ == "__main__":
    fix_date_order()