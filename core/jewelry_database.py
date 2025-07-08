#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLite 기반 주얼리 용어 데이터베이스 - 솔로몬드 AI 시스템 확장

주얼리 업계 전문 용어 관리 시스템 (JSON에서 SQLite로 업그레이드)
- 다국어 용어 저장 및 검색
- 카테고리별 분류
- 사용 빈도 추적
- STT 정확도 향상을 위한 용어 매칭

Author: 전근혁 (솔로몬드 대표)
Date: 2025.07.08
"""

import logging
import sqlite3
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

class JewelryTerminologyDB:
    """주얼리 전문 용어 SQLite 데이터베이스 클래스"""
    
    def __init__(self, db_path: str = "data/jewelry_terms.db"):
        """
        데이터베이스 초기화
        
        Args:
            db_path: SQLite 데이터베이스 파일 경로
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        
        # 데이터베이스 초기화
        self._init_database()
        self._populate_initial_data()
        
    def _init_database(self):
        """데이터베이스 테이블 생성"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 용어 테이블 생성
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS jewelry_terms (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            term_key TEXT NOT NULL UNIQUE,
                            category TEXT NOT NULL,
                            subcategory TEXT,
                            korean TEXT,
                            english TEXT,
                            chinese TEXT,
                            japanese TEXT,
                            thai TEXT,
                            description_ko TEXT,
                            description_en TEXT,
                            phonetic_ko TEXT,
                            phonetic_en TEXT,
                            frequency INTEGER DEFAULT 0,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    # 인덱스 생성
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON jewelry_terms(category)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_korean ON jewelry_terms(korean)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_english ON jewelry_terms(english)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chinese ON jewelry_terms(chinese)')
                    
                    # 용어 수정 이력 테이블
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS term_corrections (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            original_text TEXT NOT NULL,
                            corrected_text TEXT NOT NULL,
                            language TEXT NOT NULL,
                            frequency INTEGER DEFAULT 1,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    # 사용 통계 테이블
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS usage_stats (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            term_key TEXT NOT NULL,
                            usage_count INTEGER DEFAULT 1,
                            last_used DATETIME DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (term_key) REFERENCES jewelry_terms (term_key)
                        )
                    ''')
                    
                    conn.commit()
                    logger.info("데이터베이스 테이블 초기화 완료")
                    
            except Exception as e:
                logger.error(f"데이터베이스 초기화 오류: {str(e)}")
                raise
    
    def _populate_initial_data(self):
        """초기 주얼리 용어 데이터 입력 (기존 JSON 데이터 호환)"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 기존 데이터 확인
                    cursor.execute('SELECT COUNT(*) FROM jewelry_terms')
                    count = cursor.fetchone()[0]
                    
                    if count > 0:
                        logger.info(f"기존 용어 데이터 {count}개 확인됨")
                        return
                    
                    # 기존 JSON 파일에서 데이터 로딩 시도
                    json_path = Path("data/jewelry_terms.json")
                    if json_path.exists():
                        logger.info("기존 JSON 데이터를 SQLite로 마이그레이션 중...")
                        self._migrate_from_json(json_path)
                        return
                    
                    # 초기 용어 데이터 (JSON이 없을 경우)
                    initial_terms = [
                        # 보석류
                        ('diamond', '보석류', 'precious', '다이아몬드', 'diamond', '钻石', 'ダイヤモンド', 'เพชร', 
                         '가장 단단한 천연 보석', 'Hardest natural gemstone', 'dia-mon-deu', 'DAI-uh-muhnd'),
                        ('ruby', '보석류', 'precious', '루비', 'ruby', '红宝石', 'ルビー', 'ทับทิม',
                         '빨간색 강옥', 'Red corundum', 'lu-bi', 'ROO-bee'),
                        ('sapphire', '보석류', 'precious', '사파이어', 'sapphire', '蓝宝石', 'サファイア', 'ไพลิน',
                         '루비 외의 모든 색상 강옥', 'All non-red corundum', 'sa-pa-i-eo', 'SAF-ahyuhr'),
                        ('emerald', '보석류', 'precious', '에메랄드', 'emerald', '祖母绿', 'エメラルド', 'มรกต',
                         '녹색 베릴', 'Green beryl', 'e-me-ral-deu', 'EM-er-uhld'),
                        ('pearl', '보석류', 'organic', '진주', 'pearl', '珍珠', 'パール', 'ไข่มุก',
                         '조개에서 생성되는 유기보석', 'Organic gem from mollusks', 'jin-ju', 'purl'),
                        
                        # 등급 평가
                        ('carat', '등급', 'measurement', '캐럿', 'carat', '克拉', 'カラット', 'กะรัต',
                         '다이아몬드 중량 단위 (0.2g)', 'Diamond weight unit (0.2g)', 'kae-reot', 'KAR-uht'),
                        ('cut', '등급', 'quality', '컷', 'cut', '切工', 'カット', 'การตัด',
                         '다이아몬드 연마 품질', 'Diamond cutting quality', 'keot', 'kuht'),
                        ('color', '등급', 'quality', '컬러', 'color', '颜色', 'カラー', 'สี',
                         '다이아몬드 색상 등급', 'Diamond color grade', 'keol-leo', 'KUH-ler'),
                        ('clarity', '등급', 'quality', '클래리티', 'clarity', '净度', 'クラリティ', 'ความใส',
                         '다이아몬드 투명도', 'Diamond clarity grade', 'keul-lae-ri-ti', 'KLAR-i-tee'),
                        ('gia', '등급', 'institute', 'GIA', 'GIA', '美国宝石学院', 'GIA', 'GIA',
                         '미국 보석학회', 'Gemological Institute of America', 'ji-ai-ei', 'jee-eye-AY'),
                        
                        # 금속
                        ('gold', '금속', 'precious', '금', 'gold', '黄金', 'ゴールド', 'ทอง',
                         '귀금속의 대표격', 'Representative precious metal', 'geum', 'gohld'),
                        ('white_gold', '금속', 'alloy', '화이트골드', 'white gold', '白金', 'ホワイトゴールド', 'ทองขาว',
                         '금에 팔라듐이나 니켈을 합금한 금속', 'Gold alloyed with palladium or nickel', 'hwa-i-teu-gol-deu', 'wahyt gohld'),
                        ('rose_gold', '금속', 'alloy', '로즈골드', 'rose gold', '玫瑰金', 'ローズゴールド', 'ทองกุหลาบ',
                         '금에 구리를 합금해 핑크빛을 낸 금속', 'Gold alloyed with copper for pink color', 'lo-jeu-gol-deu', 'rohz gohld'),
                        ('platinum', '금속', 'precious', '플래티넘', 'platinum', '铂金', 'プラチナ', 'แพลทินัม',
                         '희귀 귀금속, 순도 높음', 'Rare precious metal, high purity', 'peul-lae-ti-neom', 'PLAT-n-uhm'),
                        ('silver', '금속', 'precious', '은', 'silver', '银', 'シルバー', 'เงิน',
                         '백색 귀금속', 'White precious metal', 'eun', 'SIL-ver'),
                        
                        # 주얼리 종류
                        ('ring', '주얼리종류', 'finger', '반지', 'ring', '戒指', 'リング', 'แหวน',
                         '손가락에 착용하는 장신구', 'Finger jewelry', 'ban-ji', 'ring'),
                        ('necklace', '주얼리종류', 'neck', '목걸이', 'necklace', '项链', 'ネックレス', 'สร้อยคอ',
                         '목에 착용하는 장신구', 'Neck jewelry', 'mok-geol-i', 'NEK-lis'),
                        ('earring', '주얼리종류', 'ear', '귀걸이', 'earring', '耳环', 'イヤリング', 'ต่างหู',
                         '귀에 착용하는 장신구', 'Ear jewelry', 'gwi-geol-i', 'EER-ring'),
                        ('bracelet', '주얼리종류', 'wrist', '팔찌', 'bracelet', '手镯', 'ブレスレット', 'สร้อยข้อมือ',
                         '손목에 착용하는 장신구', 'Wrist jewelry', 'pal-jji', 'BRAYS-lit'),
                        ('solitaire', '주얼리종류', 'setting', '솔리테어', 'solitaire', '单钻戒指', 'ソリテール', 'โซลิแทร์',
                         '중앙에 큰 다이아몬드 하나만 있는 반지', 'Ring with single center diamond', 'sol-li-te-eo', 'SOL-i-ter'),
                        
                        # 세팅 기법
                        ('prong_setting', '기술', 'setting', '프롱 세팅', 'prong setting', '爪镶', 'プロングセッティング', 'การติดตั้งแบบเล็บ',
                         '작은 발톱으로 보석을 고정하는 방식', 'Setting with small prongs', 'peu-long-se-ting', 'prong SET-ing'),
                        ('bezel_setting', '기술', 'setting', '베젤 세팅', 'bezel setting', '包镶', 'ベゼルセッティング', 'การติดตั้งแบบล้อม',
                         '금속으로 보석 둘레를 완전히 감싸는 방식', 'Setting that surrounds the gem', 'be-jel-se-ting', 'BEZ-uhl SET-ing'),
                        ('pave_setting', '기술', 'setting', '파베 세팅', 'pave setting', '密镶', 'パヴェセッティング', 'การติดตั้งแบบปูพื้น',
                         '작은 보석들을 빽빽하게 박아넣는 방식', 'Setting with closely set small gems', 'pa-be-se-ting', 'pah-VAY SET-ing'),
                        
                        # 비즈니스 용어
                        ('wholesale', '비즈니스', 'trading', '도매', 'wholesale', '批发', '卸売り', 'ขายส่ง',
                         '대량 거래', 'Bulk trading', 'do-mae', 'HOHL-sayl'),
                        ('retail', '비즈니스', 'trading', '소매', 'retail', '零售', '小売り', 'ขายปลีก',
                         '개별 고객 판매', 'Individual customer sales', 'so-mae', 'REE-tayl'),
                        ('certificate', '비즈니스', 'document', '감정서', 'certificate', '证书', '鑑定書', 'ใบรับรอง',
                         '보석 품질 인증서', 'Gem quality certificate', 'gam-jeong-seo', 'ser-TIF-i-kit'),
                        ('appraisal', '비즈니스', 'evaluation', '감정', 'appraisal', '评估', '鑑定', 'การประเมิน',
                         '가치 평가', 'Value assessment', 'gam-jeong', 'uh-PRAYZ-uhl'),
                        
                        # 시장분석
                        ('market_trend', '시장분석', 'trend', '시장 동향', 'market trend', '市场趋势', '市場トレンド', 'แนวโน้มตลาด',
                         '시장의 변화 추세', 'Market change trends', 'si-jang-dong-hyang', 'MAHR-kit trend'),
                        ('price_analysis', '시장분석', 'pricing', '가격 분석', 'price analysis', '价格分析', '価格分析', 'การวิเคราะห์ราคา',
                         '가격 변동 분석', 'Price fluctuation analysis', 'ga-gyeok-bun-seok', 'prahys uh-NAL-uh-sis'),
                        
                        # 교육
                        ('gemology', '교육', 'study', '보석학', 'gemology', '宝石学', '宝石学', 'อัญมณีวิทยา',
                         '보석에 대한 학문', 'Study of gemstones', 'bo-seok-hak', 'jem-AHL-uh-jee'),
                        ('certification', '교육', 'credential', '자격증', 'certification', '认证', '認定', 'การรับรอง',
                         '전문 자격 인증', 'Professional qualification', 'ja-gyeok-jeung', 'sur-tuh-fi-KAY-shuhn')
                    ]
                    
                    # 데이터 삽입
                    cursor.executemany('''
                        INSERT OR IGNORE INTO jewelry_terms 
                        (term_key, category, subcategory, korean, english, chinese, japanese, thai, 
                         description_ko, description_en, phonetic_ko, phonetic_en)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', initial_terms)
                    
                    conn.commit()
                    logger.info(f"초기 용어 데이터 {len(initial_terms)}개 입력 완료")
                    
            except Exception as e:
                logger.error(f"초기 데이터 입력 오류: {str(e)}")
    
    def _migrate_from_json(self, json_path: Path):
        """기존 JSON 데이터를 SQLite로 마이그레이션"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # JSON 구조에 맞게 데이터 변환
            terms_db = json_data.get('jewelry_terms_db', {})
            
            terms_to_insert = []
            
            for category, lang_data in terms_db.items():
                if isinstance(lang_data, dict):
                    korean_terms = lang_data.get('korean', [])
                    english_terms = lang_data.get('english', [])
                    chinese_terms = lang_data.get('chinese', [])
                    
                    # 각 언어의 용어를 매칭하여 삽입
                    max_len = max(len(korean_terms), len(english_terms), len(chinese_terms))
                    
                    for i in range(max_len):
                        korean = korean_terms[i] if i < len(korean_terms) else ''
                        english = english_terms[i] if i < len(english_terms) else ''
                        chinese = chinese_terms[i] if i < len(chinese_terms) else ''
                        
                        if korean or english or chinese:
                            term_key = english.lower().replace(' ', '_') if english else korean
                            terms_to_insert.append((
                                term_key, category, '', korean, english, chinese, '', '',
                                '', '', '', ''
                            ))
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.executemany('''
                    INSERT OR IGNORE INTO jewelry_terms 
                    (term_key, category, subcategory, korean, english, chinese, japanese, thai, 
                     description_ko, description_en, phonetic_ko, phonetic_en)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', terms_to_insert)
                conn.commit()
            
            logger.info(f"JSON에서 {len(terms_to_insert)}개 용어 마이그레이션 완료")
            
        except Exception as e:
            logger.error(f"JSON 마이그레이션 오류: {str(e)}")
    
    def search_terms(self, query: str, language: str = 'ko', limit: int = 10) -> List[Dict[str, Any]]:
        """용어 검색"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    # 언어별 검색 컬럼 매핑
                    lang_columns = {
                        'ko': 'korean',
                        'en': 'english',
                        'zh': 'chinese',
                        'ja': 'japanese',
                        'th': 'thai'
                    }
                    
                    search_column = lang_columns.get(language, 'korean')
                    
                    # LIKE 검색
                    cursor.execute(f'''
                        SELECT * FROM jewelry_terms 
                        WHERE {search_column} LIKE ? OR term_key LIKE ?
                        ORDER BY frequency DESC, korean ASC
                        LIMIT ?
                    ''', (f'%{query}%', f'%{query}%', limit))
                    
                    results = []
                    for row in cursor.fetchall():
                        results.append({
                            'term_key': row['term_key'],
                            'category': row['category'],
                            'subcategory': row['subcategory'],
                            'translations': {
                                'ko': row['korean'],
                                'en': row['english'],
                                'zh': row['chinese'],
                                'ja': row['japanese'],
                                'th': row['thai']
                            },
                            'descriptions': {
                                'ko': row['description_ko'],
                                'en': row['description_en']
                            },
                            'phonetics': {
                                'ko': row['phonetic_ko'],
                                'en': row['phonetic_en']
                            },
                            'frequency': row['frequency']
                        })
                    
                    return results
                    
            except Exception as e:
                logger.error(f"용어 검색 오류: {str(e)}")
                return []
    
    def get_categories(self) -> List[Dict[str, Any]]:
        """카테고리 목록 반환"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        SELECT category, COUNT(*) as count
                        FROM jewelry_terms 
                        GROUP BY category
                        ORDER BY count DESC
                    ''')
                    
                    categories = []
                    for row in cursor.fetchall():
                        categories.append({
                            'category': row[0],
                            'count': row[1]
                        })
                    
                    return categories
                    
            except Exception as e:
                logger.error(f"카테고리 조회 오류: {str(e)}")
                return []
    
    def add_correction(self, original: str, corrected: str, language: str):
        """용어 수정 이력 추가"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 기존 수정 이력 확인
                    cursor.execute('''
                        SELECT id, frequency FROM term_corrections 
                        WHERE original_text = ? AND corrected_text = ? AND language = ?
                    ''', (original, corrected, language))
                    
                    existing = cursor.fetchone()
                    
                    if existing:
                        # 빈도 업데이트
                        cursor.execute('''
                            UPDATE term_corrections 
                            SET frequency = frequency + 1 
                            WHERE id = ?
                        ''', (existing[0],))
                    else:
                        # 새로운 수정 이력 추가
                        cursor.execute('''
                            INSERT INTO term_corrections 
                            (original_text, corrected_text, language)
                            VALUES (?, ?, ?)
                        ''', (original, corrected, language))
                    
                    conn.commit()
                    
            except Exception as e:
                logger.error(f"수정 이력 추가 오류: {str(e)}")
    
    def update_usage_stats(self, term_key: str):
        """용어 사용 통계 업데이트"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 기존 통계 확인
                    cursor.execute('''
                        SELECT id, usage_count FROM usage_stats 
                        WHERE term_key = ?
                    ''', (term_key,))
                    
                    existing = cursor.fetchone()
                    
                    if existing:
                        # 사용 횟수 증가
                        cursor.execute('''
                            UPDATE usage_stats 
                            SET usage_count = usage_count + 1, last_used = CURRENT_TIMESTAMP
                            WHERE id = ?
                        ''', (existing[0],))
                    else:
                        # 새로운 통계 추가
                        cursor.execute('''
                            INSERT INTO usage_stats (term_key)
                            VALUES (?)
                        ''', (term_key,))
                    
                    # 메인 테이블의 frequency도 업데이트
                    cursor.execute('''
                        UPDATE jewelry_terms 
                        SET frequency = frequency + 1, updated_at = CURRENT_TIMESTAMP
                        WHERE term_key = ?
                    ''', (term_key,))
                    
                    conn.commit()
                    
            except Exception as e:
                logger.error(f"사용 통계 업데이트 오류: {str(e)}")
    
    def get_frequent_corrections(self, language: str = 'ko', limit: int = 50) -> List[Tuple[str, str]]:
        """자주 수정되는 용어 목록 반환"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        SELECT original_text, corrected_text, frequency
                        FROM term_corrections 
                        WHERE language = ?
                        ORDER BY frequency DESC
                        LIMIT ?
                    ''', (language, limit))
                    
                    return [(row[0], row[1]) for row in cursor.fetchall()]
                    
            except Exception as e:
                logger.error(f"자주 수정되는 용어 조회 오류: {str(e)}")
                return []
    
    def is_ready(self) -> bool:
        """데이터베이스 준비 상태 확인"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM jewelry_terms')
                count = cursor.fetchone()[0]
                return count > 0
        except:
            return False
    
    def get_stats(self) -> Dict[str, int]:
        """데이터베이스 통계 정보"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 전체 용어 수
                    cursor.execute('SELECT COUNT(*) FROM jewelry_terms')
                    total_terms = cursor.fetchone()[0]
                    
                    # 카테고리 수
                    cursor.execute('SELECT COUNT(DISTINCT category) FROM jewelry_terms')
                    categories = cursor.fetchone()[0]
                    
                    # 수정 이력 수
                    cursor.execute('SELECT COUNT(*) FROM term_corrections')
                    corrections = cursor.fetchone()[0]
                    
                    # 총 사용 횟수
                    cursor.execute('SELECT SUM(usage_count) FROM usage_stats')
                    total_usage = cursor.fetchone()[0] or 0
                    
                    return {
                        'total_terms': total_terms,
                        'categories': categories,
                        'corrections': corrections,
                        'total_usage': total_usage
                    }
                    
            except Exception as e:
                logger.error(f"통계 조회 오류: {str(e)}")
                return {
                    'total_terms': 0,
                    'categories': 0,
                    'corrections': 0,
                    'total_usage': 0
                }
    
    def backup_database(self, backup_path: str = None) -> str:
        """데이터베이스 백업"""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"data/jewelry_terms_backup_{timestamp}.db"
        
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"데이터베이스 백업 완료: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"데이터베이스 백업 오류: {str(e)}")
            raise
