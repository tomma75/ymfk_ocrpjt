import oracledb
import json
import os
from pathlib import Path

class DBConnection:
    def __init__(self):
        self.connection = None
        self.config_path = Path(__file__).parent.parent / 'config' / 'db_config.json'
        self.instantclient_path = Path(__file__).parent.parent / 'instantclient_21_7'
        
    def load_config(self):
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def connect(self):
        if self.connection:
            return self.connection
            
        config = self.load_config()
        
        # Oracle Instant Client 경로 설정
        oracledb.init_oracle_client(lib_dir=str(self.instantclient_path))
        
        # DSN 생성
        dsn = oracledb.makedsn(
            config['NEURON_HOST'],
            config['NEURON_PORT'],
            service_name=config['NEURON_SERVICE_NAME']
        )
        
        # 연결
        self.connection = oracledb.connect(
            user=config['NEURON_USER'],
            password=config['NEURON_PASSWORD'],
            dsn=dsn
        )
        
        return self.connection
    
    def verify_user(self, user_id, user_password):
        """
        사용자 인증 확인
        user_id: 7-8자리 숫자
        user_password: 숫자만 포함된 패스워드
        """
        # SQL Injection 방지를 위한 입력값 검증
        if not user_id.isdigit() or len(user_id) < 7 or len(user_id) > 8:
            return None
        
        if not user_password.isdigit():
            return None
            
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            # 바인딩 변수 사용으로 SQL Injection 방지
            # USER_ID로 직접 조회하고, USER_SYSTEM_ID, USER_MENU_GROUP 확인
            query = """
                SELECT USER_ID, USER_NAME, USER_PASSWORD, USER_SYSTEM_ID, USER_MENU_GROUP
                FROM SAP.ADM0040
                WHERE USER_SYSTEM_ID = 'TLSM' 
                AND USER_ID = :user_id
            """
            
            cursor.execute(query, user_id=user_id)
            result = cursor.fetchone()
            
            if result:
                db_user_id, db_user_name, db_password, db_system_id, db_menu_group = result
                
                # TLSM 시스템 사용자만 허용
                if db_system_id != 'TLSM':
                    return None
                
                # 패스워드 확인
                if db_password == user_password:
                    # MGR 그룹 여부 확인 (관리자 권한)
                    is_admin = (db_menu_group == 'MGR')
                    
                    return {
                        'user_id': db_user_id,
                        'user_name': db_user_name,
                        'authenticated': True,
                        'is_admin': is_admin,
                        'menu_group': db_menu_group
                    }
                else:
                    return {
                        'user_id': db_user_id,
                        'user_name': db_user_name,
                        'authenticated': False,
                        'is_admin': False,
                        'menu_group': None
                    }
            
            return None
            
        except Exception as e:
            print(f"Database error: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def get_user_name(self, user_id):
        """
        사용자 이름만 가져오기 (아이디 입력 시 이름 표시용)
        """
        if not user_id.isdigit() or len(user_id) < 7 or len(user_id) > 8:
            return None
            
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            # NLSM 시스템에서 해당 USER_ID 직접 조회
            query = """
                SELECT USER_NAME
                FROM SAP.ADM0040
                WHERE USER_SYSTEM_ID = 'TLSM'
                AND USER_ID = :user_id
            """
            
            cursor.execute(query, user_id=user_id)
            result = cursor.fetchone()
            
            if result:
                return result[0]
            return None
            
        except Exception as e:
            print(f"Database error: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def check_if_manager(self, user_id):
        """
        사용자가 MGR 권한을 가지고 있는지 확인
        """
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            # MGR 시스템에도 등록되어 있는지 확인
            query = """
                SELECT COUNT(*)
                FROM SAP.ADM0040
                WHERE USER_ID = :user_id
                AND USER_SYSTEM_ID = 'MGR'
            """
            
            cursor.execute(query, user_id=user_id)
            result = cursor.fetchone()
            
            if result and result[0] > 0:
                return True
            return False
            
        except Exception as e:
            print(f"Database error checking manager status: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
    
    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None