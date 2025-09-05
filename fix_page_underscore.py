import os
from pathlib import Path

def fix_page_underscore():
    """JSON 파일명에서 page_XXX를 pageXXX로 변경 (중복 파일 처리 포함)"""
    
    labels_dir = Path(r"D:\8.간접업무자동화\1. PO시트입력자동화\YMFK_OCR\data\processed\labels")
    
    if not labels_dir.exists():
        print(f"폴더가 존재하지 않습니다: {labels_dir}")
        return
    
    json_files = list(labels_dir.glob("*.json"))
    
    if not json_files:
        print("JSON 파일이 없습니다.")
        return
    
    renamed_count = 0
    deleted_count = 0
    skipped_count = 0
    
    print(f"총 {len(json_files)}개의 JSON 파일 검사")
    print("=" * 60)
    
    for json_file in json_files:
        filename = json_file.name
        
        # page_XXX 패턴이 있는지 확인
        if "page_" in filename:
            # page_XXX를 pageXXX로 변경
            new_name = filename.replace("page_", "page")
            new_path = json_file.parent / new_name
            
            # 이미 존재하는 경우 page_ 버전 삭제
            if new_path.exists():
                try:
                    json_file.unlink()
                    print(f"중복 파일 삭제: {filename} (이미 {new_name} 존재)")
                    deleted_count += 1
                except Exception as e:
                    print(f"삭제 오류: {filename} - {e}")
            else:
                # 파일명 변경
                try:
                    json_file.rename(new_path)
                    print(f"변경 완료: {filename} → {new_name}")
                    renamed_count += 1
                except Exception as e:
                    print(f"변경 오류: {filename} - {e}")
        else:
            skipped_count += 1
    
    print("=" * 60)
    print(f"작업 완료!")
    print(f"  변경된 파일: {renamed_count}개")
    print(f"  삭제된 중복 파일: {deleted_count}개")
    print(f"  건너뛴 파일: {skipped_count}개")

if __name__ == "__main__":
    fix_page_underscore()