import os
from pathlib import Path

def rename_json_files():
    """JSON 파일명에 _label 추가"""
    
    annotation_dir = Path(r"D:\8.간접업무자동화\1. PO시트입력자동화\YMFK_OCR\data\processed\labels")
    
    if not annotation_dir.exists():
        print(f"폴더가 존재하지 않습니다: {annotation_dir}")
        return
    
    json_files = list(annotation_dir.glob("*.json"))
    
    if not json_files:
        print("JSON 파일이 없습니다.")
        return
    
    renamed_count = 0
    skipped_count = 0
    
    print(f"총 {len(json_files)}개의 JSON 파일 발견")
    print("=" * 60)
    
    for json_file in json_files:
        # 파일명에서 확장자 제거
        name_without_ext = json_file.stem
        
        # 이미 _label이 있는지 확인
        if name_without_ext.endswith("_label"):
            print(f"건너뛰기: {json_file.name} (이미 _label 있음)")
            skipped_count += 1
            continue
        
        # 새 파일명 생성
        new_name = f"{name_without_ext}_label.json"
        new_path = json_file.parent / new_name
        
        # 파일명 변경
        try:
            json_file.rename(new_path)
            print(f"변경 완료: {json_file.name} → {new_name}")
            renamed_count += 1
        except Exception as e:
            print(f"오류 발생: {json_file.name} - {e}")
    
    print("=" * 60)
    print(f"작업 완료!")
    print(f"  변경된 파일: {renamed_count}개")
    print(f"  건너뛴 파일: {skipped_count}개")

if __name__ == "__main__":
    rename_json_files()