# -*- coding: utf-8 -*-
"""
자동 번역 시스템 - 영어-한국어 번역
다양한 번역 엔진 지원 (Google Translate, DeepL, 로컬 모델)
"""

import os
import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
from utils.logger import get_logger

class AutoTranslationSystem:
    """자동 번역 시스템"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.translation_engines = {}
        self.default_engine = None
        
        # 사용 가능한 번역 엔진 초기화
        self._initialize_translation_engines()
        
        # 언어 감지 패턴
        self.language_patterns = {
            'english': re.compile(r'[a-zA-Z]{3,}'),
            'korean': re.compile(r'[가-힣]{2,}'),
            'mixed': re.compile(r'[a-zA-Z가-힣]')
        }
    
    def _initialize_translation_engines(self):
        """번역 엔진 초기화"""
        
        self.logger.info("🌍 번역 엔진 초기화 중...")
        
        # 1. Google Translate (googletrans 라이브러리)
        try:
            from googletrans import Translator
            self.translation_engines['google'] = {
                'translator': Translator(),
                'available': True,
                'name': 'Google Translate'
            }
            self.logger.info("  ✅ Google Translate 사용 가능")
            if not self.default_engine:
                self.default_engine = 'google'
        except ImportError:
            self.logger.info("  ⚠️ Google Translate 불가능 (googletrans 미설치)")
            self.translation_engines['google'] = {'available': False}
        
        # 2. OpenAI Translation (GPT 기반)
        try:
            import openai
            api_key = os.environ.get('OPENAI_API_KEY')
            if api_key:
                self.translation_engines['openai'] = {
                    'client': openai.OpenAI(api_key=api_key),
                    'available': True,
                    'name': 'OpenAI GPT Translation'
                }
                self.logger.info("  ✅ OpenAI Translation 사용 가능")
                if not self.default_engine:
                    self.default_engine = 'openai'
            else:
                self.logger.info("  ⚠️ OpenAI Translation 불가능 (API 키 없음)")
                self.translation_engines['openai'] = {'available': False}
        except ImportError:
            self.logger.info("  ⚠️ OpenAI Translation 불가능 (openai 미설치)")
            self.translation_engines['openai'] = {'available': False}
        
        # 3. 로컬 Transformers 모델 (현재 비활성화)
        self.translation_engines['transformers'] = {'available': False}
        self.logger.info("  ⚠️ Transformers 번역 비활성화 (호환성 문제)")
        
        # 4. 기본 사전 기반 번역 (fallback)
        self.translation_engines['dictionary'] = {
            'available': True,
            'name': 'Basic Dictionary Translation'
        }
        if not self.default_engine:
            self.default_engine = 'dictionary'
        
        self.logger.info(f"🎯 기본 번역 엔진: {self.default_engine}")
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """텍스트 언어 감지"""
        
        if not text or len(text.strip()) < 3:
            return {'language': 'unknown', 'confidence': 0.0, 'needs_translation': False}
        
        # 영어 패턴 매칭
        english_matches = len(self.language_patterns['english'].findall(text))
        korean_matches = len(self.language_patterns['korean'].findall(text))
        total_words = len(text.split())
        
        # 비율 계산
        english_ratio = english_matches / max(total_words, 1)
        korean_ratio = korean_matches / max(total_words, 1)
        
        # 언어 판정
        if english_ratio > 0.6:
            language = 'english'
            confidence = min(english_ratio, 1.0)
            needs_translation = True
        elif korean_ratio > 0.6:
            language = 'korean' 
            confidence = min(korean_ratio, 1.0)
            needs_translation = False
        elif english_ratio > 0.3 and korean_ratio > 0.3:
            language = 'mixed'
            confidence = 0.7
            needs_translation = True  # 영어 부분만 번역
        else:
            language = 'unknown'
            confidence = 0.5
            needs_translation = False
        
        return {
            'language': language,
            'confidence': confidence,
            'needs_translation': needs_translation,
            'english_ratio': english_ratio,
            'korean_ratio': korean_ratio,
            'stats': {
                'english_words': english_matches,
                'korean_words': korean_matches,
                'total_words': total_words
            }
        }
    
    def translate_text(self, text: str, engine: Optional[str] = None) -> Dict[str, Any]:
        """텍스트 번역 (영어 → 한국어)"""
        
        if not text or len(text.strip()) < 2:
            return {
                'original_text': text,
                'translated_text': text,
                'language_detected': 'unknown',
                'translation_needed': False,
                'success': True,
                'engine_used': 'none'
            }
        
        # 언어 감지
        lang_info = self.detect_language(text)
        
        if not lang_info['needs_translation']:
            return {
                'original_text': text,
                'translated_text': text,
                'language_detected': lang_info['language'],
                'language_confidence': lang_info['confidence'],
                'translation_needed': False,
                'success': True,
                'engine_used': 'none'
            }
        
        # 번역 엔진 선택 - 실제 작동하는 엔진만 사용
        engine = engine or self.default_engine
        
        if engine not in self.translation_engines or not self.translation_engines[engine]['available']:
            # 폴백 엔진 선택 - dictionary만 확실히 작동함
            engine = 'dictionary'
        
        self.logger.info(f"🌍 번역 시작: {engine} 엔진 사용")
        
        try:
            # 엔진별 번역 실행
            if engine == 'google':
                translated = self._translate_with_google(text)
            elif engine == 'openai':
                translated = self._translate_with_openai(text)
            elif engine == 'transformers':
                translated = self._translate_with_transformers(text)
            elif engine == 'dictionary':
                translated = self._translate_with_dictionary(text)
            else:
                translated = text  # 번역 실패시 원문 반환
            
            return {
                'original_text': text,
                'translated_text': translated,
                'language_detected': lang_info['language'],
                'language_confidence': lang_info['confidence'],
                'translation_needed': True,
                'success': True,
                'engine_used': engine,
                'engine_name': self.translation_engines[engine]['name']
            }
            
        except Exception as e:
            self.logger.error(f"❌ 번역 실패 ({engine}): {e}")
            
            return {
                'original_text': text,
                'translated_text': text,  # 실패시 원문 반환
                'language_detected': lang_info['language'],
                'language_confidence': lang_info['confidence'],
                'translation_needed': True,
                'success': False,
                'error': str(e),
                'engine_used': engine
            }
    
    def _translate_with_google(self, text: str) -> str:
        """Google Translate로 번역"""
        
        translator = self.translation_engines['google']['translator']
        result = translator.translate(text, src='en', dest='ko')
        return result.text
    
    def _translate_with_openai(self, text: str) -> str:
        """OpenAI GPT로 번역"""
        
        client = self.translation_engines['openai']['client']
        
        prompt = f"""다음 영어 텍스트를 자연스러운 한국어로 번역해주세요. 
번역할 때 다음 사항을 고려해주세요:
1. 문맥과 의미를 정확히 전달
2. 자연스러운 한국어 표현 사용
3. 전문 용어는 적절한 한국어 용어로 변환

영어 텍스트: {text}

한국어 번역:"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 전문 번역가입니다. 영어를 한국어로 정확하고 자연스럽게 번역하세요."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    
    def _translate_with_transformers(self, text: str) -> str:
        """Transformers 모델로 번역"""
        
        engine_info = self.translation_engines['transformers']
        
        # 지연 로딩
        if engine_info['pipeline'] is None:
            from transformers import pipeline
            
            try:
                engine_info['pipeline'] = pipeline(
                    "translation",
                    model=engine_info['model_name'],
                    tokenizer=engine_info['model_name']
                )
                self.logger.info("  ✅ Transformers 모델 로드 완료")
            except Exception as e:
                self.logger.warning(f"  ⚠️ Transformers 모델 로드 실패: {e}")
                raise e
        
        # 번역 실행
        translator = engine_info['pipeline']
        
        # 텍스트가 너무 길면 분할
        max_length = 500
        if len(text) > max_length:
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            translated_chunks = []
            
            for chunk in chunks:
                result = translator(chunk, max_length=512, num_beams=4)
                translated_chunks.append(result[0]['translation_text'])
            
            return ' '.join(translated_chunks)
        else:
            result = translator(text, max_length=512, num_beams=4)
            return result[0]['translation_text']
    
    def _translate_with_dictionary(self, text: str) -> str:
        """기본 사전 기반 번역 (fallback)"""
        
        # 확장된 단어 치환 사전
        basic_dictionary = {
            # 기본 인사
            'hello': '안녕하세요',
            'hi': '안녕',
            'goodbye': '안녕히 가세요',
            'bye': '안녕',
            'thank you': '감사합니다',
            'thanks': '고마워요',
            'please': '부탁합니다',
            'sorry': '죄송합니다',
            'excuse me': '실례합니다',
            
            # 기본 답변
            'yes': '네',
            'no': '아니오',
            'okay': '좋아요',
            'ok': '좋아요',
            'sure': '물론',
            'maybe': '아마도',
            
            # 형용사
            'good': '좋은',
            'bad': '나쁜',
            'great': '훌륭한',
            'nice': '좋은',
            'wonderful': '멋진',
            'beautiful': '아름다운',
            'big': '큰',
            'small': '작은',
            'large': '큰',
            'little': '작은',
            'new': '새로운',
            'old': '오래된',
            'young': '젊은',
            'happy': '행복한',
            'sad': '슬픈',
            'important': '중요한',
            'interesting': '흥미로운',
            
            # 접속사/전치사
            'the': '',
            'and': '그리고',
            'or': '또는',
            'but': '하지만',
            'with': '함께',
            'without': '없이',
            'for': '위한',
            'to': '에게',
            'from': '에서',
            'in': '안에',
            'on': '위에',
            'at': '에서',
            'by': '으로',
            'about': '대해',
            'after': '후에',
            'before': '전에',
            'during': '동안',
            'between': '사이에',
            
            # 명사
            'work': '일',
            'job': '직업',
            'time': '시간',
            'day': '날',
            'week': '주',
            'month': '월',
            'year': '년',
            'people': '사람들',
            'person': '사람',
            'man': '남자',
            'woman': '여자',
            'child': '아이',
            'family': '가족',
            'friend': '친구',
            'home': '집',
            'house': '집',
            'place': '장소',
            'way': '방법',
            'money': '돈',
            'business': '사업',
            'company': '회사',
            'product': '제품',
            'service': '서비스',
            'customer': '고객',
            'meeting': '회의',
            'project': '프로젝트',
            'problem': '문제',
            'solution': '해결책',
            'question': '질문',
            'answer': '답변',
            'information': '정보',
            'data': '데이터',
            'system': '시스템',
            'computer': '컴퓨터',
            'phone': '전화',
            'email': '이메일',
            'internet': '인터넷',
            'website': '웹사이트',
            'book': '책',
            'music': '음악',
            'movie': '영화',
            'food': '음식',
            'water': '물',
            'coffee': '커피',
            'tea': '차',
            'travel': '여행',
            'hotel': '호텔',
            'airport': '공항',
            'car': '자동차',
            'train': '기차',
            'bus': '버스',
            'school': '학교',
            'university': '대학교',
            'hospital': '병원',
            'doctor': '의사',
            'teacher': '선생님',
            'student': '학생',
            
            # 동사
            'can': '할 수 있다',
            'could': '할 수 있었다',
            'will': '할 것이다',
            'would': '할 것이다',
            'should': '해야 한다',
            'must': '해야 한다',
            'have': '가지다',
            'has': '가지고 있다',
            'had': '가졌다',
            'be': '이다',
            'is': '이다',
            'are': '이다',
            'was': '였다',
            'were': '였다',
            'do': '하다',
            'does': '한다',
            'did': '했다',
            'make': '만들다',
            'get': '얻다',
            'give': '주다',
            'take': '가져가다',
            'bring': '가져오다',
            'put': '놓다',
            'keep': '유지하다',
            'let': '하게 하다',
            'help': '돕다',
            'try': '시도하다',
            'start': '시작하다',
            'stop': '멈추다',
            'finish': '끝내다',
            'continue': '계속하다',
            'come': '오다',
            'go': '가다',
            'leave': '떠나다',
            'stay': '머물다',
            'live': '살다',
            'work': '일하다',
            'play': '놀다',
            'study': '공부하다',
            'learn': '배우다',
            'teach': '가르치다',
            'read': '읽다',
            'write': '쓰다',
            'speak': '말하다',
            'talk': '이야기하다',
            'listen': '듣다',
            'hear': '듣다',
            'see': '보다',
            'look': '보다',
            'watch': '보다',
            'know': '알다',
            'understand': '이해하다',
            'think': '생각하다',
            'believe': '믿다',
            'remember': '기억하다',
            'forget': '잊다',
            'feel': '느끼다',
            'love': '사랑하다',
            'like': '좋아하다',
            'want': '원하다',
            'need': '필요하다',
            'buy': '사다',
            'sell': '팔다',
            'pay': '지불하다',
            'cost': '비용이 들다',
            'eat': '먹다',
            'drink': '마시다',
            'sleep': '자다',
            'wake': '깨다',
            'open': '열다',
            'close': '닫다',
            'turn': '돌리다',
            'move': '움직이다',
            'walk': '걷다',
            'run': '뛰다',
            'drive': '운전하다',
            'fly': '날다',
            'swim': '수영하다',
            'dance': '춤추다',
            'sing': '노래하다',
            'laugh': '웃다',
            'cry': '울다',
            'smile': '미소짓다',
            'wait': '기다리다',
            'hope': '희망하다',
            'expect': '기대하다',
            'plan': '계획하다',
            'decide': '결정하다',
            'choose': '선택하다',
            'change': '바꾸다',
            'improve': '개선하다',
            'increase': '증가하다',
            'decrease': '감소하다',
            'create': '만들다',
            'build': '짓다',
            'design': '디자인하다',
            'develop': '개발하다',
            'manage': '관리하다',
            'control': '통제하다',
            'organize': '조직하다',
            'prepare': '준비하다',
            'check': '확인하다',
            'test': '테스트하다',
            'analyze': '분석하다',
            'compare': '비교하다',
            'measure': '측정하다',
            'count': '세다',
            'calculate': '계산하다',
            'solve': '해결하다',
            'fix': '고치다',
            'repair': '수리하다',
            'clean': '청소하다',
            'wash': '씻다',
            'cook': '요리하다',
            'cut': '자르다',
            'break': '부수다',
            'connect': '연결하다',
            'join': '참가하다',
            'meet': '만나다',
            'visit': '방문하다',
            'call': '전화하다',
            'send': '보내다',
            'receive': '받다',
            'share': '공유하다',
            'use': '사용하다',
            'wear': '입다',
            'carry': '나르다',
            'hold': '잡다',
            'touch': '만지다',
            'push': '밀다',
            'pull': '당기다',
            'throw': '던지다',
            'catch': '잡다',
            'hit': '치다',
            'kick': '차다',
            'jump': '뛰다',
            'climb': '오르다',
            'fall': '떨어지다',
            'win': '이기다',
            'lose': '지다',
            'fight': '싸우다',
            'compete': '경쟁하다',
            'support': '지원하다',
            'protect': '보호하다',
            'save': '저장하다',
            'spend': '쓰다',
            'waste': '낭비하다',
            'enjoy': '즐기다',
            'relax': '휴식하다',
            'rest': '쉬다',
            'worry': '걱정하다',
            'care': '돌보다',
            'notice': '알아차리다',
            'ignore': '무시하다',
            'avoid': '피하다',
            'prevent': '방지하다',
            'allow': '허용하다',
            'accept': '받아들이다',
            'refuse': '거절하다',
            'agree': '동의하다',
            'disagree': '동의하지 않다',
            'discuss': '논의하다',
            'argue': '논쟁하다',
            'explain': '설명하다',
            'describe': '묘사하다',
            'suggest': '제안하다',
            'recommend': '추천하다',
            'advise': '조언하다',
            'warn': '경고하다',
            'promise': '약속하다',
            'invite': '초대하다',
            'welcome': '환영하다',
            'celebrate': '축하하다',
            'congratulate': '축하하다',
            'thank': '감사하다',
            'apologize': '사과하다',
            'forgive': '용서하다',
            'miss': '그리워하다',
            'suppose': '생각하다',
            'imagine': '상상하다',
            'dream': '꿈꾸다',
            'wish': '바라다',
            'wonder': '궁금하다',
            'guess': '추측하다',
            'doubt': '의심하다',
            'trust': '신뢰하다',
            'depend': '의존하다',
            'rely': '의지하다',
            'belong': '속하다',
            'contain': '포함하다',
            'include': '포함시키다',
            'exclude': '제외하다',
            'relate': '관련되다',
            'connect': '연결하다',
            'attach': '붙이다',
            'separate': '분리하다',
            'divide': '나누다',
            'combine': '결합하다',
            'mix': '섞다',
            'add': '더하다',
            'subtract': '빼다',
            'multiply': '곱하다',
            'divide': '나누다',
            'equal': '같다',
            'match': '맞다',
            'fit': '맞다',
            'suit': '어울리다',
            'follow': '따르다',
            'lead': '이끌다',
            'guide': '안내하다',
            'direct': '지시하다',
            'order': '주문하다',
            'command': '명령하다',
            'request': '요청하다',
            'ask': '묻다',
            'answer': '답하다',
            'reply': '답장하다',
            'respond': '응답하다',
            'react': '반응하다',
            'act': '행동하다',
            'behave': '행동하다',
            'perform': '수행하다',
            'execute': '실행하다',
            'operate': '운영하다',
            'function': '기능하다',
            'serve': '서비스하다',
            'provide': '제공하다',
            'supply': '공급하다',
            'deliver': '배달하다',
            'transport': '운송하다',
            'carry': '운반하다',
            'load': '적재하다',
            'unload': '하역하다',
            'pack': '포장하다',
            'unpack': '포장을 풀다',
            'wrap': '포장하다',
            'unwrap': '포장을 풀다',
            'cover': '덮다',
            'uncover': '덮개를 벗기다',
            'hide': '숨기다',
            'show': '보여주다',
            'reveal': '드러내다',
            'display': '표시하다',
            'present': '발표하다',
            'introduce': '소개하다',
            'represent': '대표하다',
            'replace': '교체하다',
            'substitute': '대체하다',
            'exchange': '교환하다',
            'trade': '거래하다',
            'negotiate': '협상하다',
            'bargain': '흥정하다',
            'compete': '경쟁하다',
            'challenge': '도전하다',
            'attempt': '시도하다',
            'succeed': '성공하다',
            'fail': '실패하다',
            'achieve': '달성하다',
            'accomplish': '성취하다',
            'complete': '완성하다',
            'finish': '끝내다',
            'end': '끝나다',
            'begin': '시작하다',
            'start': '시작하다',
            'continue': '계속하다',
            'proceed': '진행하다',
            'advance': '진보하다',
            'progress': '진행하다',
            'develop': '발전하다',
            'grow': '성장하다',
            'expand': '확장하다',
            'extend': '연장하다',
            'stretch': '늘이다',
            'reach': '도달하다',
            'arrive': '도착하다',
            'depart': '출발하다',
            'return': '돌아오다',
            'enter': '들어가다',
            'exit': '나가다',
            'approach': '접근하다',
            'escape': '탈출하다',
            'discover': '발견하다',
            'find': '찾다',
            'search': '검색하다',
            'seek': '찾다',
            'explore': '탐험하다',
            'investigate': '조사하다',
            'examine': '조사하다',
            'observe': '관찰하다',
            'monitor': '모니터하다',
            'track': '추적하다',
            'record': '기록하다',
            'document': '문서화하다',
            'report': '보고하다',
            'announce': '발표하다',
            'declare': '선언하다',
            'state': '말하다',
            'claim': '주장하다',
            'assert': '단언하다',
            'confirm': '확인하다',
            'verify': '검증하다',
            'prove': '증명하다',
            'demonstrate': '보여주다',
            'illustrate': '설명하다',
            'clarify': '명확히하다',
            'specify': '명시하다',
            'define': '정의하다',
            'identify': '식별하다',
            'recognize': '인식하다',
            'distinguish': '구별하다',
            'classify': '분류하다',
            'categorize': '분류하다',
            'organize': '정리하다',
            'arrange': '배열하다',
            'sort': '정렬하다',
            'order': '주문하다',
            'rank': '순위를 매기다',
            'rate': '평가하다',
            'evaluate': '평가하다',
            'judge': '판단하다',
            'criticize': '비판하다',
            'praise': '칭찬하다',
            'compliment': '칭찬하다',
            'blame': '비난하다',
            'accuse': '고발하다',
            'charge': '요금을 부과하다',
            'fine': '벌금을 부과하다',
            'punish': '처벌하다',
            'reward': '보상하다',
            'award': '수여하다',
            'grant': '부여하다',
            'permit': '허가하다',
            'authorize': '인가하다',
            'approve': '승인하다',
            'disapprove': '반대하다',
            'reject': '거절하다',
            'deny': '거부하다',
            'admit': '인정하다',
            'confess': '고백하다',
            'reveal': '밝히다',
            'conceal': '숨기다',
            'expose': '폭로하다',
            'publish': '발간하다',
            'broadcast': '방송하다',
            'transmit': '전송하다',
            'communicate': '소통하다',
            'inform': '알리다',
            'notify': '통지하다',
            'alert': '경보하다',
            'remind': '알리다',
            'memorize': '암기하다',
            'recall': '회상하다',
            'recollect': '기억하다',
            'review': '검토하다',
            'revise': '수정하다',
            'edit': '편집하다',
            'correct': '수정하다',
            'fix': '고치다',
            'adjust': '조정하다',
            'adapt': '적응하다',
            'modify': '수정하다',
            'alter': '바꾸다',
            'transform': '변환하다',
            'convert': '전환하다',
            'translate': '번역하다',
            'interpret': '해석하다',
            'switch': '바꾸다',
            'shift': '이동하다',
            'transfer': '이전하다',
            'move': '이동하다',
            'relocate': '재배치하다',
            'migrate': '이주하다',
            'travel': '여행하다',
            'journey': '여행하다',
            'tour': '관광하다',
            'visit': '방문하다',
            'stay': '머물다',
            'remain': '남다',
            'maintain': '유지하다',
            'preserve': '보존하다',
            'conserve': '보존하다',
            'protect': '보호하다',
            'defend': '방어하다',
            'attack': '공격하다',
            'assault': '공격하다',
            'invade': '침입하다',
            'conquer': '정복하다',
            'defeat': '물리치다',
            'overcome': '극복하다',
            'handle': '다루다',
            'manage': '관리하다',
            'deal': '거래하다',
            'cope': '대처하다',
            'struggle': '투쟁하다',
            'effort': '노력하다',
            'attempt': '시도하다',
            'strive': '노력하다',
            'aim': '목표로 삼다',
            'target': '표적으로 삼다',
            'focus': '집중하다',
            'concentrate': '집중하다',
            'dedicate': '헌신하다',
            'devote': '바치다',
            'commit': '약속하다',
            'engage': '참여하다',
            'participate': '참가하다',
            'involve': '포함시키다',
            'contribute': '기여하다',
            'donate': '기부하다',
            'volunteer': '자원하다',
            'offer': '제공하다',
            'propose': '제안하다',
            
            # 특별 구문
            'how are you': '어떻게 지내세요',
            'nice to meet you': '만나서 반가워요',
            'see you later': '나중에 봐요',
            'good morning': '좋은 아침',
            'good afternoon': '좋은 오후',
            'good evening': '좋은 저녁',
            'good night': '잘 자요',
            'have a good day': '좋은 하루 되세요',
            'take care': '조심하세요',
            'you are welcome': '천만에요',
            'no problem': '문제없어요',
            'of course': '물론이죠',
            'i think': '제 생각에는',
            'i believe': '저는 믿어요',
            'in my opinion': '제 의견으로는',
            'for example': '예를 들어',
            'such as': '같은',
            'as well': '또한',
            'in addition': '게다가',
            'however': '하지만',
            'therefore': '따라서',
            'because': '왜냐하면',
            'although': '비록',
            'unless': '하지 않는 한',
            'whether': '인지 아닌지',
            'either': '둘 중 하나',
            'neither': '둘 다 아닌',
            'both': '둘 다',
            'each': '각각',
            'every': '모든',
            'all': '모든',
            'some': '일부',
            'many': '많은',
            'few': '적은',
            'several': '몇 개의',
            'enough': '충분한',
            'too much': '너무 많은',
            'too many': '너무 많은',
            'a lot of': '많은',
            'a little': '조금'
        }
        
        # 단어별로 치환 (단순한 방식)
        words = text.lower().split()
        translated_words = []
        
        for word in words:
            # 구두점 제거
            clean_word = re.sub(r'[^\w]', '', word)
            
            if clean_word in basic_dictionary:
                translated = basic_dictionary[clean_word]
                if translated:  # 빈 문자열이 아닌 경우만
                    translated_words.append(translated)
            else:
                # 번역되지 않은 단어는 그대로 유지
                translated_words.append(word)
        
        result = ' '.join(translated_words)
        
        # 기본 번역 표시
        return f"[기본번역] {result}"
    
    def translate_mixed_text(self, text: str, engine: Optional[str] = None) -> Dict[str, Any]:
        """혼합 언어 텍스트 번역 (영어 부분만 번역)"""
        
        # 영어 구문 추출
        sentences = re.split(r'[.!?]+', text)
        translated_sentences = []
        translation_info = {
            'total_sentences': len(sentences),
            'translated_sentences': 0,
            'original_text': text
        }
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            lang_info = self.detect_language(sentence)
            
            if lang_info['needs_translation']:
                # 영어 문장 번역
                result = self.translate_text(sentence, engine)
                if result['success']:
                    translated_sentences.append(result['translated_text'])
                    translation_info['translated_sentences'] += 1
                else:
                    translated_sentences.append(sentence)  # 번역 실패시 원문
            else:
                # 한국어 문장은 그대로 유지
                translated_sentences.append(sentence)
        
        translation_info['translated_text'] = '. '.join(translated_sentences)
        translation_info['success'] = True
        
        return translation_info
    
    def batch_translate(self, texts: List[str], engine: Optional[str] = None) -> List[Dict[str, Any]]:
        """배치 번역"""
        
        results = []
        
        for i, text in enumerate(texts):
            self.logger.info(f"🌍 배치 번역 ({i+1}/{len(texts)})")
            result = self.translate_text(text, engine)
            results.append(result)
            
            # API 호출 제한 고려 (잠시 대기)
            if engine in ['google', 'openai'] and i < len(texts) - 1:
                time.sleep(0.5)
        
        return results
    
    def get_translation_stats(self) -> Dict[str, Any]:
        """번역 시스템 통계"""
        
        available_engines = [
            name for name, info in self.translation_engines.items() 
            if info.get('available', False)
        ]
        
        return {
            'available_engines': available_engines,
            'total_engines': len(self.translation_engines),
            'default_engine': self.default_engine,
            'engine_details': {
                name: {
                    'available': info.get('available', False),
                    'name': info.get('name', name.title())
                }
                for name, info in self.translation_engines.items()
            }
        }

# 전역 번역 시스템 인스턴스
_global_translation_system = None

def get_translation_system():
    """전역 번역 시스템 인스턴스 반환"""
    global _global_translation_system
    if _global_translation_system is None:
        _global_translation_system = AutoTranslationSystem()
    return _global_translation_system

def translate_text(text: str, engine: Optional[str] = None) -> Dict[str, Any]:
    """편의 함수: 텍스트 번역"""
    return get_translation_system().translate_text(text, engine)

def detect_language(text: str) -> Dict[str, Any]:
    """편의 함수: 언어 감지"""
    return get_translation_system().detect_language(text)