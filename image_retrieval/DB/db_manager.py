import os
import sqlite3
import json

class DBManager:
    def __init__(self, db_path="furniture_database.sqlite3"):
        """
        데이터베이스 관리자를 초기화합니다.
        :param db_path: 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self):
        """ 테이블이 존재하지 않으면 생성합니다. """
        conn = None # conn 변수를 미리 선언
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # category 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS category (
                    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category_name TEXT NOT NULL
                )
            """)
            
            # furniture 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS furniture (
                    furniture_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    productname TEXT NOT NULL,
                    category_id INTEGER NOT NULL,
                    style TEXT,
                    price REAL,
                    filename TEXT,
                    FOREIGN KEY (category_id) REFERENCES category(category_id)
                )
            """)
            
            # feature 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feature (
                    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    furniture_id INTEGER NOT NULL,
                    feature TEXT NOT NULL,
                    FOREIGN KEY (furniture_id) REFERENCES furniture(furniture_id)
                )
            """)
            
            # metadata 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    metadata_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    furniture_id INTEGER NOT NULL,
                    feature_id INTEGER NOT NULL,
                    FOREIGN KEY (furniture_id) REFERENCES furniture(furniture_id),
                    FOREIGN KEY (feature_id) REFERENCES feature(feature_id)
                )
            """)
            
            conn.commit()
        #except sqlite3.Error as e:
        #    print(f"테이블 생성 오류: {e}")
        finally:
            if conn:
                conn.close()
                
    def insert_category(self, category_name):
        """
        카테고리를 데이터베이스에 삽입합니다.
        :param category_name: 카테고리 이름 (예: "Sofa", "Chair")
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 이미 존재하는 카테고리가 있는지 확인
            cursor.execute("SELECT category_id FROM category WHERE category_name = ?", (category_name,))
            existing_category = cursor.fetchone()
            
            if existing_category:
                print(f"카테고리 '{category_name}' 는 이미 존재합니다.")
            else:
                # 카테고리 없으면 새로 삽입
                cursor.execute("""
                    INSERT INTO category (category_name) 
                    VALUES (?)
                """, (category_name,))
                conn.commit()
                print(f"카테고리 삽입 성공: {category_name}")
        except sqlite3.Error as e:
            print(f"카테고리 삽입 오류: {e}")
        finally:
            if conn:
                conn.close()

    def insert_furniture(self, productname, category_id, style, price, filename):
        """
        가구를 furniture 테이블에 삽입합니다.
        :param productname: 제품명
        :param category_id: 카테고리 ID (category 테이블의 ID)
        :param style: 스타일 (예: Modern)
        :param price: 가격
        :param filename: 이미지 파일명
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO furniture (productname, category_id, style, price, filename)
                VALUES (?, ?, ?, ?, ?)
            """, (productname, category_id, style, price, filename))

            conn.commit()
            print(f"가구 삽입 성공: {productname}")
        except sqlite3.Error as e:
            print(f"가구 삽입 오류: {e}")
        finally:
            if conn:
                conn.close()

    def insert_feature(self, furniture_id, feature):
        """
        가구의 특성을 feature 테이블에 삽입합니다.
        :param furniture_id: 가구 ID (furniture 테이블의 ID)
        :param feature: 특성 (예: "Wooden", "Red", 등)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO feature (furniture_id, feature)
                VALUES (?, ?)
            """, (furniture_id, feature))
            
            conn.commit()
            print(f"feature 삽입 성공")
        except sqlite3.Error as e:
            print(f"feature 삽입 오류: {e}")
        finally:
            if conn:
                conn.close()
                
    def insert_metadata(self, furniture_id, feature_id):
        """
        가구의 메타데이터를 metadata 테이블에 삽입합니다.
        :param furniture_id: 가구 ID (furniture 테이블의 ID)
        :param feature_id: 특성 ID (feature 테이블의 ID)
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO metadata (furniture_id, feature_id)
                VALUES (?, ?)
            """, (furniture_id, feature_id))

            conn.commit()
            print(f"메타데이터 삽입 성공")
        #except sqlite3.Error as e:
        #    print(f"메타데이터 삽입 오류: {e}")
        finally:
            if conn:
                conn.close()
    def get_category_id(self, category_name):
        """
        카테고리 이름을 통해 카테고리 ID를 조회합니다.
        :param category_name: 카테고리 이름
        :return: category_id
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT category_id FROM category WHERE category_name = ?", (category_name,))
            category_id = cursor.fetchone()
            return category_id[0] if category_id else None
        except sqlite3.Error as e:
            print(f"카테고리 ID 조회 오류: {e}")
        finally:
            if conn:
                conn.close()
        return None

    def get_furniture_id(self, filename):
        """
        파일 이름을 통해 가구 ID를 조회합니다.
        :param filename: 이미지 파일 이름
        :return: furniture_id
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT furniture_id FROM furniture WHERE filename = ?", (filename,))
            furniture_id = cursor.fetchone()
            return furniture_id[0] if furniture_id else None
        except sqlite3.Error as e:
            print(f"가구 ID 조회 오류: {e}")
        finally:
            if conn:
                conn.close()
        return None
    
    def get_feature_id(self, furniture_id):
        """
        furniture_id 을 통해 feature ID를 조회합니다.
        :param filename: 이미지 파일 이름
        :return: feature_id
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT feature_id FROM feature WHERE furniture_id = ?", (furniture_id,))
            result = cursor.fetchone()
            return result[0] if result else None
        except sqlite3.Error as e:
            print(f"feature ID 조회 오류: {e}")
        finally:
            if conn:
                conn.close()
        return None
