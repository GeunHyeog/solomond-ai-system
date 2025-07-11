# 💎 주얼리 AI 플랫폼 - 프로젝트 지침서 v7.0

## 🎯 **핵심 목표 (2025.07.10 업데이트)**
> **현장에서 촬영한 다양한 언어 자료를 한국어로 통합 분석하는 최고품질 시스템**

## 👤 **프로젝트 정보**
- **개발자**: 전근혁 (주얼리 전문가, 솔로몬드 대표)
- **직책**: 한국보석협회 사무국장
- **전문분야**: 주얼리 업계 + AI 기술 융합
- **타겟**: 주얼리 업계 세미나/회의/강의 현장 분석

## ✅ **현재 상태 (v2.0 Production Ready)**
- **멀티모달 시스템**: 100% 완성 ✅
- **고용량 처리**: 50개 파일 2.1GB 처리 ✅
- **GEMMA LLM**: 고품질 요약 생성 ✅
- **실시간 스트리밍**: 메모리 효율 70% 개선 ✅
- **주얼리 특화**: 1000+ 전문 용어 데이터베이스 ✅

## 🔥 **v2.1 현장 품질 최적화 목표**

### **1. 다국어 → 한국어 통합 요약 시스템**
**요구사항**: "입력은 다양한 언어가 될 수도 있어. 한국어로 최종 결과물을 요약 분석 하는 단계를 추가시키자"

**구현 목표**:
- 영어, 중국어, 일본어 등 다국어 입력 자동 감지
- 주얼리 전문 용어 한국어 매핑 시스템  
- GEMMA LLM 기반 한국어 통합 요약 생성
- 원문 의미 보존하면서 한국 업계 용어로 번역

### **2. 현장 녹화 노이즈 분석 시스템**
**요구사항**: "현장에서 녹화, 사진을 찍은 것이므로 추출할때 노이즈 분석은 잘 되었는지 알려줘"

**구현 목표**:
- 실시간 오디오 노이즈 레벨 분석 (SNR 계산)
- 배경 잡음 유형 분류 및 감지
- 음성 명확도 평가 및 품질 스코어링
- 품질 개선 제안 및 즉시 피드백 시스템

### **3. PPT 화면 OCR 정확도 분석**
**요구사항**: "사진에 있는 PPT화면도 잘 읽고 있는 것인지 알고싶어"

**구현 목표**:
- PPT 슬라이드 자동 감지 및 영역 추출
- 표, 차트, 도형 내 텍스트 정확도 분석
- 레이아웃 구조 분석 및 읽기 순서 최적화
- OCR 신뢰도 실시간 평가 및 개선 제안

### **4. 상황별 다중 파일 통합 분석**
**요구사항**: "하나의 상황(세미나, 회의, 강의 등)을 지시하는 자료들 이므로 여러개의 파일을 다중 분석해서 하나의 결과물로 만들고 싶어"

**구현 목표**:
- 세미나/회의/강의 상황 자동 분류
- 시간순 이벤트 재구성 및 화자별 발언 통합
- 주제별 내용 그룹화 및 맥락 연결
- 상황별 최적화된 한국어 통합 요약 생성

## 🛠️ **MCP 도구 활용 전략**

| 도구 | 역할 | v2.1 특화 활용 |
|------|------|----------------|
| **Memory** | 품질 이력 관리 | 노이즈/OCR 품질 패턴 학습 |
| **GitHub** | 코드 버전 관리 | 품질 분석 모듈 개발 |
| **Perplexity** | 기술 조사 | 최신 OCR/STT 기술 연구 |
| **Filesystem** | 파일 관리 | 현장 자료 체계적 정리 |

## 📅 **단계별 실행 계획 (3주 완성)**

### **Week 1 (07.15-21): 품질 분석 시스템**
#### **Day 1-3: 현장 오디오 품질 분석기**
```python
# core/field_quality_analyzer.py
class FieldQualityAnalyzer:
    def analyze_audio_quality(self, audio_data):
        """현장 녹음 품질 분석"""
        - SNR (Signal-to-Noise Ratio) 계산
        - 배경잡음 유형 분류 (에어컨, 사람소리, 기계음 등)
        - 음성 명확도 평가 (spectral analysis)
        - 실시간 품질 스코어 (0-100)
        
    def provide_audio_recommendations(self, quality_score):
        """오디오 개선 제안"""
        - 낮은 품질: "마이크를 화자에게 더 가까이 배치하세요"
        - 높은 잡음: "더 조용한 환경에서 녹음하시기 바랍니다"
        - 볼륨 문제: "녹음 볼륨을 조정하세요"
```

#### **Day 4-5: PPT 화면 특화 OCR**
```python
# core/presentation_ocr_analyzer.py
class PresentationOCRAnalyzer:
    def detect_ppt_slide(self, image):
        """PPT 슬라이드 자동 감지"""
        - 슬라이드 경계 검출
        - 제목/본문 영역 구분
        - 표/차트 영역 식별
        
    def analyze_ocr_accuracy(self, image, ocr_result):
        """OCR 정확도 실시간 평가"""
        - 문자 인식 신뢰도 분석
        - 레이아웃 구조 정확도
        - 읽기 순서 논리성 평가
        
    def enhance_ppt_ocr(self, image):
        """PPT 이미지 전처리 최적화"""
        - 대비 향상 및 노이즈 제거
        - 텍스트 영역 강조
        - 각도 보정 및 해상도 최적화
```

#### **Day 6-7: 품질 피드백 UI**
```python
# ui/quality_feedback_interface.py
class QualityFeedbackInterface:
    def show_realtime_quality(self):
        """실시간 품질 대시보드"""
        - 오디오 노이즈 레벨 게이지
        - OCR 정확도 진행률
        - 품질 개선 제안 알림
        
    def generate_quality_report(self):
        """품질 분석 리포트"""
        - 파일별 품질 점수
        - 개선 필요 부분 강조
        - 전체 신뢰도 평가
```

### **Week 2 (07.22-28): 한국어 통합 시스템**
#### **Day 1-3: 다국어 → 한국어 통합 요약**
```python
# core/korean_final_summarizer.py
class KoreanFinalSummarizer:
    def detect_and_translate(self, content):
        """다국어 감지 및 한국어 번역"""
        - 언어 자동 감지 (langdetect)
        - 주얼리 용어 매핑 테이블 적용
        - 컨텍스트 보존 번역
        
    def generate_korean_summary(self, multilingual_content):
        """GEMMA 기반 한국어 요약"""
        - 주얼리 특화 한국어 프롬프트
        - 업계 표준 용어 사용
        - 상황별 요약 스타일 적용
        
    def integrate_multilingual_insights(self, sources):
        """다국어 소스 통합 인사이트"""
        - 언어별 중요도 가중치
        - 중복 내용 제거 및 통합
        - 한국어 최종 결론 도출
```

#### **Day 4-5: 상황별 컨텍스트 분석**
```python
# core/contextual_session_analyzer.py
class ContextualSessionAnalyzer:
    def classify_session_type(self, files):
        """상황 자동 분류"""
        - 세미나: 발표자 중심, PPT 자료 많음
        - 회의: 다중 화자, 토론 형식
        - 강의: 교육 내용, 질의응답
        
    def reconstruct_timeline(self, files):
        """시간순 이벤트 재구성"""
        - 파일 생성 시간 기반 정렬
        - 발언 시간 추정
        - 주제 전환 시점 감지
        
    def integrate_speaker_content(self, files):
        """화자별 발언 통합"""
        - 화자 식별 및 구분
        - 발언 내용 시간순 정렬
        - 주요 발언자 기여도 분석
```

### **Week 3 (07.29-31): 통합 및 최적화**
#### **Day 1-2: 전체 시스템 통합**
```python
# core/integrated_field_analyzer.py
class IntegratedFieldAnalyzer:
    def process_field_session(self, files, session_info):
        """현장 세션 통합 처리"""
        1. 파일별 품질 분석 및 전처리
        2. 다국어 콘텐츠 한국어 통합
        3. 상황별 컨텍스트 분석
        4. 최종 한국어 요약 생성
        5. 품질 기반 신뢰도 조정
        
    def generate_field_report(self, analysis_result):
        """현장 분석 리포트 생성"""
        - 품질 분석 결과
        - 한국어 통합 요약
        - 개선 제안 사항
        - 신뢰도 평가
```

#### **Day 3: 최종 테스트 및 배포**
- 현장 시나리오 통합 테스트
- 사용자 가이드 업데이트
- v2.1 릴리즈 준비

## 🔄 **채팅창 전환 프로토콜 v7.0**

### **종료 전 체크리스트**
1. **Memory 업데이트**
   ```
   현재 구현 중인 기능: [구체적 파일명]
   품질 분석 진행률: [x%]
   한국어 요약 시스템: [상태]
   다음 우선순위: [구체적 작업]
   품질 개선 이슈: [있다면 명시]
   ```

2. **GitHub 커밋**
   ```bash
   git add .
   git commit -m "v2.1 현장품질최적화: [기능명] [완성률]% - [주요 변경사항]"
   git push origin main
   ```

3. **품질 메트릭 기록**
   ```
   오디오 노이즈 분석: [구현상태]
   PPT OCR 정확도: [개선률]
   한국어 요약 품질: [평가점수]
   ```

### **시작 시 표준 프롬프트**
```
Memory에서 주얼리 AI 플랫폼 현장 품질 최적화(v2.1) 진행상황을 확인하고, 
다음 우선순위 작업을 즉시 시작:
1. 현장 오디오 노이즈 분석 시스템
2. PPT 화면 OCR 정확도 분석
3. 다국어→한국어 통합 요약
4. 상황별 다중 파일 통합 분석

현재 v2.0이 완성된 상태에서 v2.1 현장 최적화 기능을 추가 개발 중.
```

## 📊 **성공 지표 v7.0**

### **품질 분석 정확도**
- **오디오 노이즈 감지**: 95% 정확도
- **PPT OCR 인식률**: 현재 대비 20% 향상  
- **번역 품질**: BLEU 스코어 0.8 이상
- **품질 예측 신뢰도**: 90% 이상

### **사용자 만족도**
- **현장 사용성**: 실시간 피드백 3초 이내
- **한국어 요약 품질**: 4.5/5.0 이상
- **품질 개선 제안**: 85% 정확도
- **전체 만족도**: 95% 이상

### **비즈니스 가치**
- **분석 시간**: 추가 30% 단축
- **품질 향상**: 신뢰도 20% 개선
- **현장 활용**: 95% 즉시 적용 가능
- **다국어 지원**: 90% 실무 활용

## 💡 **차별화 포인트 v7.0**

### **1. 현장 특화**
- **실시간 품질 피드백**: 촬영 중 즉시 품질 알림
- **현장 개선 제안**: "마이크를 더 가까이", "조명 개선" 등
- **노이즈 환경 대응**: 세미나홀, 전시장 등 현장 최적화

### **2. 한국어 중심**
- **업계 용어 완벽 번역**: 4C → 4씨, Clarity → 투명도 등
- **한국 업계 표준**: 한국보석협회 용어 기준 적용
- **상황별 한국어**: 회의체, 높임말, 업계 관습 반영

### **3. 상황 인식**
- **세미나 vs 회의 vs 강의**: 각각 다른 분석 방식
- **시간 흐름 추적**: 발표 → 질의응답 → 토론 순서 인식
- **화자별 역할**: 발표자, 질문자, 토론 참가자 구분

### **4. 품질 보장**
- **실시간 품질 모니터링**: 처리 중 품질 저하 즉시 감지
- **신뢰도 기반 조정**: 낮은 품질 부분은 신뢰도 하향
- **개선 제안**: 구체적이고 실행 가능한 피드백

## 🚀 **즉시 실행 항목 (오늘부터)**

### **오늘 (2025.07.10)**
- [ ] field_quality_analyzer.py 기본 구조 생성
- [ ] 오디오 노이즈 분석 알고리즘 설계
- [ ] 품질 스코어링 시스템 프로토타입

### **내일 (2025.07.11)**  
- [ ] presentation_ocr_analyzer.py 구현 시작
- [ ] PPT 슬라이드 감지 알고리즘
- [ ] OCR 정확도 평가 시스템

### **이번 주 (07.15까지)**
- [ ] 품질 피드백 UI 프로토타입
- [ ] 실시간 품질 대시보드
- [ ] 기본 개선 제안 시스템

---

## 📞 **연락처**
- **개발자**: 전근혁 (솔로몬드 대표)
- **이메일**: solomond.jgh@gmail.com  
- **전화**: 010-2983-0338
- **GitHub**: https://github.com/GeunHyeog/solomond-ai-system

## 🎯 **핵심 가치**
> **"현장에서 촬영한 다양한 언어 자료를 최고 품질로 한국어 통합 분석"**
> 
> **주얼리 업계 전문가가 현장에서 바로 사용할 수 있는 실용적 AI 도구**

---

**🔄 문서 버전**: v7.0 (2025.07.10)  
**📊 현재 상태**: v2.0 Production Ready → v2.1 현장 최적화 개발 중  
**🎯 최종 목표**: 현장 사용성 100% + 한국어 통합 분석의 완성  
**⏰ 완성 목표**: 2025.07.31 (3주 내)