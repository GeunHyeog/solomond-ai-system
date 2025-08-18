#!/usr/bin/env python3
"""
🔄 크로스모달 융합 레이어 - SOLOMOND AI 진정한 멀티모달리티 구현
Cross-Modal Fusion Layer: Inter-Modal Correlation Learning & Fused Representation

🎯 주요 기능:
1. 멀티헤드 어텐션 - 모달간 복잡한 관계 학습
2. 상관관계 분석 - 높은 관련성 쌍 탐지
3. 융합 표현 생성 - 모든 모달리티 통합 벡터
4. 시맨틱 정렬 - 의미적 일관성 보장
5. 적응적 가중치 - 모달 중요도 자동 조정
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import time
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

# 로컬 임포트
from .multimodal_encoder import EncodedResult, MultimodalEncoder

# 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FusionResult:
    """크로스모달 융합 결과"""
    fused_embedding: np.ndarray  # 최종 융합된 768차원 벡터
    modal_weights: Dict[str, float]  # 모달별 기여도
    cross_modal_correlations: List[Dict]  # 모달간 상관관계
    dominant_modality: str  # 가장 영향력 있는 모달
    fusion_confidence: float  # 융합 신뢰도
    semantic_coherence: float  # 의미적 일관성 점수
    metadata: Dict[str, Any]

class MultiHeadAttention(nn.Module):
    """멀티헤드 어텐션 모듈"""
    
    def __init__(self, d_model=768, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply final linear layer
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(attended)
        
        return output, attention_weights

class CrossModalFusionLayer:
    """크로스모달 융합 레이어 핵심 엔진"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        
        # PyTorch 모델들
        self.attention_layer = MultiHeadAttention(
            d_model=self.config['embedding_dim'],
            num_heads=self.config['num_attention_heads']
        )
        self.fusion_network = self._build_fusion_network()
        
        # 성능 메트릭
        self.stats = {
            'fusion_operations': 0,
            'high_correlation_pairs': 0,
            'average_coherence': 0.0,
            'modality_contributions': {'image': [], 'audio': [], 'text': []}
        }
        
    def _get_default_config(self) -> Dict:
        """기본 설정"""
        return {
            'embedding_dim': 768,
            'num_attention_heads': 8,
            'correlation_threshold': 0.6,
            'fusion_layers': [768, 512, 768],
            'dropout_rate': 0.1,
            'temperature': 0.1,  # 소프트맥스 온도
            'min_modalities': 2,  # 최소 모달 수
            'coherence_weight': 0.3
        }
    
    def _build_fusion_network(self) -> nn.Module:
        """융합 네트워크 구축"""
        layers = []
        layer_sizes = self.config['fusion_layers']
        
        for i in range(len(layer_sizes) - 1):
            layers.extend([
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                nn.ReLU(),
                nn.Dropout(self.config['dropout_rate'])
            ])
            
        # 마지막 레이어는 활성화 함수 없음
        layers = layers[:-2]  # 마지막 ReLU, Dropout 제거
        layers.append(nn.Tanh())  # 최종 정규화를 위한 Tanh
        
        return nn.Sequential(*layers)
    
    def fuse_multimodal_encodings(self, encoded_results: List[EncodedResult]) -> FusionResult:
        """멀티모달 인코딩 결과 융합"""
        if len(encoded_results) < self.config['min_modalities']:
            logger.warning(f"융합에 필요한 최소 모달 수({self.config['min_modalities']}) 미달")
            return self._create_single_modal_result(encoded_results[0] if encoded_results else None)
            
        logger.info(f"🔄 크로스모달 융합 시작: {len(encoded_results)}개 모달")
        start_time = time.time()
        
        # 1. 모달간 상관관계 분석
        correlations = self._analyze_cross_modal_correlations(encoded_results)
        
        # 2. 어텐션 기반 모달 가중치 계산
        modal_weights = self._calculate_modal_weights(encoded_results, correlations)
        
        # 3. 융합된 표현 생성
        fused_embedding = self._generate_fused_representation(encoded_results, modal_weights)
        
        # 4. 의미적 일관성 평가
        coherence_score = self._evaluate_semantic_coherence(encoded_results, fused_embedding)
        
        # 5. 지배적 모달리티 결정
        dominant_modality = max(modal_weights.items(), key=lambda x: x[1])[0]
        
        # 6. 융합 신뢰도 계산
        fusion_confidence = self._calculate_fusion_confidence(encoded_results, correlations, coherence_score)
        
        processing_time = time.time() - start_time
        
        # 통계 업데이트
        self._update_stats(modal_weights, correlations, coherence_score)
        
        result = FusionResult(
            fused_embedding=fused_embedding,
            modal_weights=modal_weights,
            cross_modal_correlations=correlations,
            dominant_modality=dominant_modality,
            fusion_confidence=fusion_confidence,
            semantic_coherence=coherence_score,
            metadata={
                'num_modalities': len(encoded_results),
                'processing_time': processing_time,
                'high_correlation_pairs': len([c for c in correlations if c['correlation'] > self.config['correlation_threshold']]),
                'modality_types': [r.modality for r in encoded_results]
            }
        )
        
        logger.info(f"✅ 크로스모달 융합 완료: 신뢰도 {fusion_confidence:.3f}, 일관성 {coherence_score:.3f}")
        return result
    
    def _analyze_cross_modal_correlations(self, encoded_results: List[EncodedResult]) -> List[Dict]:
        """모달간 상관관계 분석"""
        correlations = []
        
        # 모든 쌍에 대해 상관관계 계산
        for i, result_a in enumerate(encoded_results):
            for j, result_b in enumerate(encoded_results[i + 1:], i + 1):
                
                # 코사인 유사도 계산
                similarity = cosine_similarity(
                    result_a.encoding.reshape(1, -1),
                    result_b.encoding.reshape(1, -1)
                )[0][0]
                
                # 상관관계 정보 저장
                correlation_info = {
                    'modality_pair': f"{result_a.modality}-{result_b.modality}",
                    'file_pair': (Path(result_a.file_path).name, Path(result_b.file_path).name),
                    'correlation': float(similarity),
                    'confidence_product': result_a.confidence * result_b.confidence,
                    'semantic_distance': 1.0 - similarity,
                    'is_high_correlation': similarity > self.config['correlation_threshold']
                }
                
                correlations.append(correlation_info)
                
        # 상관관계 순으로 정렬
        correlations.sort(key=lambda x: x['correlation'], reverse=True)
        
        return correlations
    
    def _calculate_modal_weights(self, encoded_results: List[EncodedResult], correlations: List[Dict]) -> Dict[str, float]:
        """어텐션 기반 모달 가중치 계산"""
        
        # 각 모달의 기본 가중치 (신뢰도 기반)
        base_weights = {}
        for result in encoded_results:
            base_weights[result.modality] = result.confidence
            
        # 상관관계 기반 가중치 조정
        correlation_bonus = {modality: 0.0 for modality in base_weights.keys()}
        
        for corr in correlations:
            if corr['is_high_correlation']:
                modalities = corr['modality_pair'].split('-')
                bonus = corr['correlation'] * 0.2  # 20% 보너스
                
                for modality in modalities:
                    if modality in correlation_bonus:
                        correlation_bonus[modality] += bonus
                        
        # 최종 가중치 계산 및 정규화
        final_weights = {}
        total_weight = 0.0
        
        for modality in base_weights.keys():
            weight = base_weights[modality] + correlation_bonus[modality]
            final_weights[modality] = weight
            total_weight += weight
            
        # 정규화 (총합 1.0)
        if total_weight > 0:
            for modality in final_weights.keys():
                final_weights[modality] /= total_weight
        else:
            # 균등 가중치로 폴백
            uniform_weight = 1.0 / len(final_weights)
            final_weights = {modality: uniform_weight for modality in final_weights.keys()}
            
        return final_weights
    
    def _generate_fused_representation(self, encoded_results: List[EncodedResult], modal_weights: Dict[str, float]) -> np.ndarray:
        """융합된 표현 생성"""
        
        # 가중 평균으로 기본 융합
        fused_embedding = np.zeros(self.config['embedding_dim'], dtype=np.float32)
        
        for result in encoded_results:
            weight = modal_weights.get(result.modality, 0.0)
            fused_embedding += weight * result.encoding
            
        # 어텐션 메커니즘 적용 (PyTorch)
        try:
            # 임베딩들을 텐서로 변환
            embeddings_tensor = torch.FloatTensor([result.encoding for result in encoded_results])
            embeddings_tensor = embeddings_tensor.unsqueeze(0)  # 배치 차원 추가
            
            # 셀프 어텐션 적용
            with torch.no_grad():
                attended_embeddings, attention_weights = self.attention_layer(
                    embeddings_tensor, embeddings_tensor, embeddings_tensor
                )
                
                # 가중 평균으로 최종 융합
                weights_tensor = torch.FloatTensor(list(modal_weights.values())).unsqueeze(0).unsqueeze(-1)
                final_fused = (attended_embeddings * weights_tensor).mean(dim=1).squeeze()
                
                fused_embedding = final_fused.numpy()
                
        except Exception as e:
            logger.warning(f"어텐션 적용 실패, 기본 융합 사용: {e}")
            
        # L2 정규화
        norm = np.linalg.norm(fused_embedding)
        if norm > 0:
            fused_embedding = fused_embedding / norm
            
        return fused_embedding
    
    def _evaluate_semantic_coherence(self, encoded_results: List[EncodedResult], fused_embedding: np.ndarray) -> float:
        """의미적 일관성 평가"""
        coherence_scores = []
        
        # 각 모달이 융합 결과와 얼마나 일치하는지 측정
        for result in encoded_results:
            similarity = cosine_similarity(
                result.encoding.reshape(1, -1),
                fused_embedding.reshape(1, -1)
            )[0][0]
            coherence_scores.append(similarity * result.confidence)
            
        # 모달간 일관성도 고려
        if len(encoded_results) > 1:
            embeddings = np.array([result.encoding for result in encoded_results])
            pairwise_similarities = cosine_similarity(embeddings)
            
            # 대각선 제외한 평균 유사도
            mask = np.ones_like(pairwise_similarities, dtype=bool)
            np.fill_diagonal(mask, False)
            inter_modal_coherence = pairwise_similarities[mask].mean()
            coherence_scores.append(inter_modal_coherence)
            
        return float(np.mean(coherence_scores))
    
    def _calculate_fusion_confidence(self, encoded_results: List[EncodedResult], correlations: List[Dict], coherence: float) -> float:
        """융합 신뢰도 계산"""
        
        # 개별 모달 신뢰도 평균
        individual_confidence = np.mean([result.confidence for result in encoded_results])
        
        # 상관관계 신뢰도
        if correlations:
            correlation_confidence = np.mean([corr['correlation'] for corr in correlations])
        else:
            correlation_confidence = 0.5
            
        # 모달 수에 따른 보너스 (더 많은 모달 = 더 높은 신뢰도)
        modality_bonus = min(0.2, (len(encoded_results) - 1) * 0.05)
        
        # 최종 신뢰도 계산
        fusion_confidence = (
            individual_confidence * 0.4 +
            correlation_confidence * 0.3 +
            coherence * 0.3 +
            modality_bonus
        )
        
        return min(1.0, fusion_confidence)
    
    def _create_single_modal_result(self, encoded_result: Optional[EncodedResult]) -> FusionResult:
        """단일 모달 결과 생성 (폴백)"""
        if not encoded_result:
            # 빈 결과
            return FusionResult(
                fused_embedding=np.zeros(self.config['embedding_dim'], dtype=np.float32),
                modal_weights={},
                cross_modal_correlations=[],
                dominant_modality="none",
                fusion_confidence=0.0,
                semantic_coherence=0.0,
                metadata={'single_modal_fallback': True}
            )
            
        return FusionResult(
            fused_embedding=encoded_result.encoding,
            modal_weights={encoded_result.modality: 1.0},
            cross_modal_correlations=[],
            dominant_modality=encoded_result.modality,
            fusion_confidence=encoded_result.confidence,
            semantic_coherence=1.0,  # 단일 모달이므로 완전 일관성
            metadata={
                'single_modal_fallback': True,
                'original_modality': encoded_result.modality
            }
        )
    
    def _update_stats(self, modal_weights: Dict[str, float], correlations: List[Dict], coherence: float):
        """통계 업데이트"""
        self.stats['fusion_operations'] += 1
        self.stats['high_correlation_pairs'] += sum(1 for c in correlations if c['is_high_correlation'])
        
        # 이동 평균으로 일관성 업데이트
        alpha = 0.1
        self.stats['average_coherence'] = (1 - alpha) * self.stats['average_coherence'] + alpha * coherence
        
        # 모달별 기여도 기록
        for modality, weight in modal_weights.items():
            if modality in self.stats['modality_contributions']:
                self.stats['modality_contributions'][modality].append(weight)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        avg_contributions = {}
        for modality, contributions in self.stats['modality_contributions'].items():
            avg_contributions[modality] = np.mean(contributions) if contributions else 0.0
            
        return {
            'total_fusions': self.stats['fusion_operations'],
            'high_correlation_pairs': self.stats['high_correlation_pairs'],
            'average_coherence': round(self.stats['average_coherence'], 3),
            'average_modal_contributions': avg_contributions,
            'correlation_rate': (
                self.stats['high_correlation_pairs'] / max(1, self.stats['fusion_operations'])
            )
        }
    
    def analyze_fusion_patterns(self, fusion_results: List[FusionResult]) -> Dict[str, Any]:
        """융합 패턴 분석"""
        if not fusion_results:
            return {'error': '분석할 융합 결과가 없습니다'}
            
        # 지배적 모달리티 패턴
        dominant_modalities = [result.dominant_modality for result in fusion_results]
        modality_counts = {mod: dominant_modalities.count(mod) for mod in set(dominant_modalities)}
        
        # 평균 신뢰도 및 일관성
        avg_confidence = np.mean([result.fusion_confidence for result in fusion_results])
        avg_coherence = np.mean([result.semantic_coherence for result in fusion_results])
        
        # 높은 상관관계 패턴
        all_correlations = []
        for result in fusion_results:
            all_correlations.extend(result.cross_modal_correlations)
            
        high_corr_pairs = [corr for corr in all_correlations if corr['is_high_correlation']]
        
        return {
            'total_fusions': len(fusion_results),
            'dominant_modality_distribution': modality_counts,
            'average_fusion_confidence': round(avg_confidence, 3),
            'average_semantic_coherence': round(avg_coherence, 3),
            'high_correlation_patterns': len(high_corr_pairs),
            'most_correlated_pair': max(all_correlations, key=lambda x: x['correlation']) if all_correlations else None,
            'fusion_quality': 'excellent' if avg_confidence > 0.8 else 'good' if avg_confidence > 0.6 else 'needs_improvement'
        }

# 사용 예제
def main():
    """사용 예제"""
    
    # 멀티모달 인코더 초기화
    encoder = MultimodalEncoder()
    fusion_layer = CrossModalFusionLayer()
    
    # 테스트 파일들
    test_files = [
        Path("test_image.jpg"),
        Path("test_audio.wav"),
        Path("test_document.txt")
    ]
    
    existing_files = [f for f in test_files if f.exists()]
    
    if len(existing_files) >= 2:
        print("🔄 크로스모달 융합 레이어 테스트")
        print("=" * 50)
        
        # 1. 멀티모달 인코딩
        encoded_results = encoder.encode_batch(existing_files)
        print(f"📊 인코딩 완료: {len(encoded_results)}개 모달")
        
        # 2. 크로스모달 융합
        fusion_result = fusion_layer.fuse_multimodal_encodings(encoded_results)
        
        print(f"\n🎯 융합 결과:")
        print(f"   융합 임베딩 차원: {fusion_result.fused_embedding.shape}")
        print(f"   지배적 모달리티: {fusion_result.dominant_modality}")
        print(f"   융합 신뢰도: {fusion_result.fusion_confidence:.3f}")
        print(f"   의미적 일관성: {fusion_result.semantic_coherence:.3f}")
        
        print(f"\n⚖️ 모달 가중치:")
        for modality, weight in fusion_result.modal_weights.items():
            print(f"   {modality}: {weight:.3f}")
            
        print(f"\n🔗 크로스모달 상관관계:")
        for corr in fusion_result.cross_modal_correlations[:3]:  # 상위 3개만
            print(f"   {corr['modality_pair']}: {corr['correlation']:.3f} {'✓' if corr['is_high_correlation'] else ''}")
            
        # 3. 성능 통계
        stats = fusion_layer.get_performance_stats()
        print(f"\n📈 성능 통계:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
            
    else:
        print("❌ 테스트를 위해 최소 2개의 서로 다른 모달리티 파일이 필요합니다.")

if __name__ == "__main__":
    main()