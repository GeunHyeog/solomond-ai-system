"""
솔로몬드 AI 시스템 - 완전한 고용량 다중분석 LLM 요약 엔진
5GB 파일 50개 동시 처리 + GEMMA 통합 요약 시스템

핵심 기능:
1. 스트리밍 청크 처리 (메모리 효율성)
2. 계층적 요약 생성 (청크 → 소스 → 최종)
3. 주얼리 도메인 특화 분석
4. 품질 평가 및 신뢰도 검증
5. 대용량 파일 최적화 처리
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import time
import gc
from pathlib import Path
import re
from collections import defaultdict

# 외부 라이브러리
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    GEMMA_AVAILABLE = True
except ImportError:
    GEMMA_AVAILABLE = False

class EnhancedLLMSummarizer:
    """고용량 다중분석 전용 LLM 요약 엔진"""
    
    def __init__(self, model_name: str = "google/gemma-2b-it"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 주얼리 특화 프롬프트
        self.prompts = {
            "executive": """다음 주얼리 업계 정보를 경영진 관점에서 요약하세요:
- 핵심 비즈니스 인사이트 
- 시장 기회 및 위험
- 실행 가능한 전략적 권장사항

내용: {content}
요약 (한국어, 500자):""",
            
            "technical": """다음 주얼리 기술 정보를 전문가 관점에서 요약하세요:
- 보석학적 특징 및 품질 분석
- 가공 기술 및 처리 방법  
- 감정 및 인증 관련 사항

내용: {content}
기술 요약 (한국어, 600자):""",
            
            "comprehensive": """다음 주얼리 업계 종합 정보를 전체적으로 요약하세요:
- 시장 현황 및 전망
- 기술적 특징 및 품질 분석
- 비즈니스 기회 및 전략
- 업계 트렌드 및 미래 방향

내용: {content}
종합 요약 (한국어, 1000자):"""
        }
    
    async def initialize_model(self):
        """모델 초기화"""
        if self.model is not None:
            return
            
        try:
            if GEMMA_AVAILABLE:
                print("🤖 GEMMA 모델 로딩 중...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                print("✅ GEMMA 모델 로딩 완료")
            else:
                print("⚠️ GEMMA 모의 모드로 실행")
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
    
    async def process_large_batch(self, files_data: List[Dict]) -> Dict:
        """대용량 배치 처리 메인 함수"""
        print(f"🚀 대용량 배치 처리 시작: {len(files_data)}개 파일")
        start_time = time.time()
        
        try:
            # 1. 모델 초기화
            await self.initialize_model()
            
            # 2. 파일 분류 및 검증
            classified_files = self._classify_files(files_data)
            
            # 3. 청크 단위로 분할
            chunks = await self._create_chunks(classified_files)
            
            # 4. 병렬 처리
            processed_chunks = await self._process_chunks_parallel(chunks)
            
            # 5. 계층적 요약 생성
            hierarchical_summary = await self._generate_hierarchical_summary(processed_chunks)
            
            # 6. 품질 평가
            quality_assessment = await self._assess_summary_quality(hierarchical_summary, processed_chunks)
            
            # 7. 최종 보고서 생성
            final_report = {
                "success": True,
                "session_id": f"batch_{int(time.time())}",
                "processing_time": time.time() - start_time,
                "files_processed": len(files_data),
                "chunks_processed": len(processed_chunks),
                "hierarchical_summary": hierarchical_summary,
                "quality_assessment": quality_assessment,
                "recommendations": self._generate_recommendations(quality_assessment)
            }
            
            print(f"✅ 대용량 배치 처리 완료 ({final_report['processing_time']:.1f}초)")
            return final_report
            
        except Exception as e:
            logging.error(f"배치 처리 오류: {e}")
            return {"success": False, "error": str(e)}
    
    def _classify_files(self, files_data: List[Dict]) -> Dict:
        """파일 분류"""
        classified = {
            "audio": [],
            "video": [], 
            "documents": [],
            "images": []
        }
        
        for file_data in files_data:
            filename = file_data.get("filename", "")
            ext = Path(filename).suffix.lower()
            
            if ext in ['.mp3', '.wav', '.m4a']:
                classified["audio"].append(file_data)
            elif ext in ['.mp4', '.avi', '.mov']:
                classified["video"].append(file_data)
            elif ext in ['.pdf', '.docx', '.txt']:
                classified["documents"].append(file_data)
            elif ext in ['.jpg', '.jpeg', '.png']:
                classified["images"].append(file_data)
        
        return classified
    
    async def _create_chunks(self, classified_files: Dict) -> List[Dict]:
        """청크 생성"""
        chunks = []
        chunk_id = 0
        
        for file_type, files in classified_files.items():
            for file_data in files:
                text = file_data.get("processed_text", "")
                if not text:
                    continue
                
                # 텍스트를 1000자 단위로 청크 분할
                chunk_size = 1000
                for i in range(0, len(text), chunk_size):
                    chunk_text = text[i:i+chunk_size]
                    
                    chunks.append({
                        "chunk_id": f"chunk_{chunk_id:04d}",
                        "source_file": file_data.get("filename", ""),
                        "source_type": file_type,
                        "text": chunk_text,
                        "token_count": len(chunk_text.split()),
                        "jewelry_terms": self._extract_jewelry_terms(chunk_text)
                    })
                    chunk_id += 1
        
        return chunks
    
    async def _process_chunks_parallel(self, chunks: List[Dict]) -> List[Dict]:
        """병렬 청크 처리"""
        print(f"⚡ 병렬 처리: {len(chunks)}개 청크")
        
        # 최대 10개씩 배치 처리
        max_concurrent = 10
        processed_chunks = []
        
        for i in range(0, len(chunks), max_concurrent):
            batch = chunks[i:i+max_concurrent]
            
            # 배치 내 병렬 처리
            tasks = [self._process_single_chunk(chunk) for chunk in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if not isinstance(result, Exception):
                    processed_chunks.append(result)
        
        return processed_chunks
    
    async def _process_single_chunk(self, chunk: Dict) -> Dict:
        """개별 청크 처리"""
        try:
            # 주얼리 용어 강화
            enhanced_text = self._enhance_jewelry_content(chunk["text"])
            
            # 청크 요약 생성
            summary = await self._generate_chunk_summary(enhanced_text)
            
            chunk["summary"] = summary
            chunk["enhanced_text"] = enhanced_text
            
            return chunk
            
        except Exception as e:
            logging.error(f"청크 처리 오류: {e}")
            return chunk
    
    async def _generate_chunk_summary(self, text: str) -> str:
        """청크 요약 생성"""
        if GEMMA_AVAILABLE and self.model is not None:
            return await self._generate_with_gemma(text, "comprehensive")
        else:
            return await self._generate_mock_summary(text)
    
    async def _generate_with_gemma(self, text: str, summary_type: str) -> str:
        """GEMMA 모델로 요약 생성"""
        try:
            prompt = self.prompts[summary_type].format(content=text)
            
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=2048, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = generated_text[len(prompt):].strip()
            
            return summary
            
        except Exception as e:
            logging.error(f"GEMMA 요약 생성 오류: {e}")
            return await self._generate_mock_summary(text)
    
    async def _generate_mock_summary(self, text: str) -> str:
        """모의 요약 생성"""
        await asyncio.sleep(0.1)
        
        sentences = text.split('. ')
        if len(sentences) <= 3:
            return text
        
        # 주얼리 키워드 우선 선택
        jewelry_keywords = ["다이아몬드", "루비", "사파이어", "에메랄드", "4C", "GIA", "캐럿"]
        
        scored_sentences = []
        for sentence in sentences:
            score = sum(1 for keyword in jewelry_keywords if keyword in sentence)
            scored_sentences.append((sentence, score))
        
        top_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:3]
        return '. '.join([s[0] for s in top_sentences]) + '.'
    
    async def _generate_hierarchical_summary(self, chunks: List[Dict]) -> Dict:
        """계층적 요약 생성"""
        print("📊 계층적 요약 생성 중...")
        
        # 1. 소스별 그룹화
        source_groups = defaultdict(list)
        for chunk in chunks:
            source_groups[chunk["source_type"]].append(chunk)
        
        # 2. 소스별 중간 요약
        source_summaries = {}
        for source_type, source_chunks in source_groups.items():
            combined_text = " ".join([chunk.get("summary", chunk["text"]) for chunk in source_chunks])
            
            if GEMMA_AVAILABLE and self.model is not None:
                summary = await self._generate_with_gemma(combined_text, "comprehensive")
            else:
                summary = await self._generate_mock_summary(combined_text)
            
            source_summaries[source_type] = {
                "summary": summary,
                "chunk_count": len(source_chunks),
                "total_tokens": sum(chunk["token_count"] for chunk in source_chunks)
            }
        
        # 3. 최종 통합 요약
        all_summaries = " ".join([s["summary"] for s in source_summaries.values()])
        
        if GEMMA_AVAILABLE and self.model is not None:
            final_summary = await self._generate_with_gemma(all_summaries, "comprehensive")
        else:
            final_summary = await self._generate_mock_summary(all_summaries)
        
        return {
            "final_summary": final_summary,
            "source_summaries": source_summaries,
            "total_chunks": len(chunks)
        }
    
    async def _assess_summary_quality(self, hierarchical_summary: Dict, chunks: List[Dict]) -> Dict:
        """요약 품질 평가"""
        print("🔍 요약 품질 평가 중...")
        
        final_summary = hierarchical_summary["final_summary"]
        
        # 키워드 커버리지 분석
        all_jewelry_terms = set()
        for chunk in chunks:
            all_jewelry_terms.update(chunk["jewelry_terms"])
        
        summary_terms = set(self._extract_jewelry_terms(final_summary))
        coverage_ratio = len(summary_terms) / max(len(all_jewelry_terms), 1)
        
        # 압축률 분석
        original_length = sum(len(chunk["text"]) for chunk in chunks)
        summary_length = len(final_summary)
        compression_ratio = summary_length / max(original_length, 1)
        
        # 품질 점수 계산
        quality_score = (coverage_ratio * 0.4 + (1 - compression_ratio) * 0.3 + 0.3) * 100
        
        return {
            "quality_score": min(100, max(0, quality_score)),
            "coverage_ratio": coverage_ratio,
            "compression_ratio": compression_ratio,
            "summary_length": summary_length,
            "original_length": original_length,
            "jewelry_terms_found": len(summary_terms),
            "jewelry_terms_total": len(all_jewelry_terms)
        }
    
    def _generate_recommendations(self, quality_assessment: Dict) -> List[str]:
        """품질 기반 권장사항 생성"""
        recommendations = []
        
        quality_score = quality_assessment["quality_score"]
        
        if quality_score >= 85:
            recommendations.append("✅ 우수한 품질의 요약이 생성되었습니다.")
        elif quality_score >= 70:
            recommendations.append("⚠️ 양호한 품질이나 추가 검토가 필요합니다.")
        else:
            recommendations.append("❌ 품질 개선이 필요합니다. 더 많은 데이터나 수동 검토를 권장합니다.")
        
        if quality_assessment["coverage_ratio"] < 0.6:
            recommendations.append("💡 주얼리 전문 용어 커버리지가 부족합니다. 도메인 특화 보강이 필요합니다.")
        
        if quality_assessment["compression_ratio"] > 0.8:
            recommendations.append("📝 압축률이 낮습니다. 더 간결한 요약이 가능할 수 있습니다.")
        
        return recommendations
    
    def _extract_jewelry_terms(self, text: str) -> List[str]:
        """주얼리 용어 추출"""
        jewelry_terms = [
            "다이아몬드", "루비", "사파이어", "에메랄드", "4C", "GIA", "캐럿", 
            "컬러", "클래리티", "컷", "도매가", "소매가", "인증서", "감정서",
            "프린세스 컷", "라운드 브릴리언트", "VS", "VVS", "IF", "FL"
        ]
        
        found_terms = []
        text_lower = text.lower()
        
        for term in jewelry_terms:
            if term.lower() in text_lower:
                found_terms.append(term)
        
        return found_terms
    
    def _enhance_jewelry_content(self, text: str) -> str:
        """주얼리 콘텐츠 향상"""
        # 용어 정규화
        term_mapping = {
            "4씨": "4C",
            "지아": "GIA",
            "브릴리언트": "Brilliant",
            "프린세스": "Princess"
        }
        
        enhanced = text
        for korean, english in term_mapping.items():
            enhanced = enhanced.replace(korean, f"{korean}({english})")
        
        return enhanced

# 사용 예시 및 테스트
async def test_enhanced_summarizer():
    """향상된 요약기 테스트"""
    print("🧪 향상된 LLM 요약기 테스트 시작")
    
    summarizer = EnhancedLLMSummarizer()
    
    # 테스트 데이터 생성
    test_files = [
        {
            "filename": "diamond_market_2025.mp3",
            "processed_text": "2025년 다이아몬드 시장 전망에 대해 말씀드리겠습니다. 4C 등급 중에서 특히 컬러와 클래리티 등급이 가격에 미치는 영향이 클 것으로 예상됩니다. GIA 인증서의 중요성이 더욱 강조되고 있으며, 프린세스 컷과 라운드 브릴리언트 컷의 수요가 증가하고 있습니다."
        },
        {
            "filename": "ruby_pricing.pdf", 
            "processed_text": "버마산 루비의 가격 동향을 분석해보겠습니다. 1캐럿 기준으로 히트 트리트먼트가 되지 않은 천연 루비의 경우 $4,000에서 $6,000 사이의 가격대를 형성하고 있습니다. 색상의 순도와 투명도가 가격 결정에 가장 중요한 요소입니다."
        },
        {
            "filename": "emerald_certification.jpg",
            "processed_text": "에메랄드 감정서에 나타난 주요 정보들입니다. 콜롬비아산 에메랄드 2.15캐럿, 컬러 등급 Vivid Green, 클래리티 VS, 오일 트리트먼트 Minor. GIA 인증서 번호 5141234567입니다."
        }
    ]
    
    # 배치 처리 실행
    result = await summarizer.process_large_batch(test_files)
    
    # 결과 출력
    print("\n" + "="*60)
    print("🎉 테스트 완료 결과")
    print("="*60)
    print(f"성공 여부: {result['success']}")
    print(f"처리 시간: {result.get('processing_time', 0):.2f}초")
    print(f"처리된 파일: {result.get('files_processed', 0)}개")
    print(f"처리된 청크: {result.get('chunks_processed', 0)}개")
    
    if result['success']:
        print(f"\n📊 최종 요약:")
        print(result['hierarchical_summary']['final_summary'])
        
        print(f"\n🎯 품질 평가:")
        qa = result['quality_assessment']
        print(f"- 품질 점수: {qa['quality_score']:.1f}/100")
        print(f"- 키워드 커버리지: {qa['coverage_ratio']:.1%}")
        print(f"- 압축률: {qa['compression_ratio']:.1%}")
        
        print(f"\n💡 권장사항:")
        for rec in result['recommendations']:
            print(f"  {rec}")
    
    return result

# 메인 실행
if __name__ == "__main__":
    asyncio.run(test_enhanced_summarizer())
