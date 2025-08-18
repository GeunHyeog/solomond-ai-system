# -*- coding: utf-8 -*-
"""
간단한 향상된 이미지 분석기 - Windows 안전 버전
OCR + 기본 이미지 분석 통합
"""

import os
import sys
import time
from pathlib import Path

# 프로젝트 경로 추가  
sys.path.append(str(Path(__file__).parent))

def analyze_image_enhanced(image_path):
    """이미지 향상 분석 - OCR + 기본 분석"""
    
    print(f"이미지 분석: {os.path.basename(image_path)}")
    
    result = {
        'filename': os.path.basename(image_path),
        'ocr_text': '',
        'image_info': '',
        'combined_analysis': ''
    }
    
    # 1. OCR 텍스트 추출
    try:
        import easyocr
        reader = easyocr.Reader(['ko', 'en'])
        ocr_results = reader.readtext(image_path)
        
        if ocr_results:
            texts = [item[1] for item in ocr_results if item[2] > 0.5]
            result['ocr_text'] = ' '.join(texts)
            print(f"  OCR 추출: {len(result['ocr_text'])}글자")
        else:
            print("  OCR: 텍스트 없음")
            
    except Exception as e:
        print(f"  OCR 오류: {str(e)}")
    
    # 2. 기본 이미지 정보 분석
    try:
        from PIL import Image
        import numpy as np
        
        image = Image.open(image_path)
        width, height = image.size
        
        info_parts = []
        info_parts.append(f"크기: {width}x{height}")
        
        # 화면비 분석
        aspect_ratio = width / height
        if aspect_ratio > 1.5:
            info_parts.append("가로형(풍경/스크린샷)")
        elif aspect_ratio < 0.7:
            info_parts.append("세로형(모바일)")
        else:
            info_parts.append("정방형")
        
        # 색상 분석
        if image.mode == 'RGB':
            np_image = np.array(image)
            avg_color = np.mean(np_image, axis=(0, 1))
            
            if avg_color[0] > 200 and avg_color[1] > 200 and avg_color[2] > 200:
                info_parts.append("밝은 색조")
            elif avg_color[0] < 100 and avg_color[1] < 100 and avg_color[2] < 100:
                info_parts.append("어두운 색조")
        
        result['image_info'] = ' | '.join(info_parts)
        print(f"  이미지 정보: {result['image_info']}")
        
    except Exception as e:
        print(f"  이미지 분석 오류: {str(e)}")
    
    # 3. 결합 분석
    combined_insights = []
    
    if result['ocr_text'] and result['image_info']:
        combined_insights.append("텍스트와 시각 정보 모두 포함")
    elif result['ocr_text']:
        combined_insights.append("주로 텍스트 기반 이미지")
    elif result['image_info']:
        combined_insights.append("주로 시각적 이미지")
    
    # 내용 기반 키워드 분석
    content = result['ocr_text'].lower()
    
    if '2025' in content or 'thu' in content or 'pm' in content:
        combined_insights.append("날짜/시간 정보")
    
    if 'rise' in content or 'eco' in content:
        combined_insights.append("성장/환경 관련")
    
    if 'global' in content or 'cultura' in content:
        combined_insights.append("국제/문화 관련")
    
    if len(result['ocr_text']) > 500:
        combined_insights.append("풍부한 텍스트 정보")
    
    result['combined_analysis'] = ' | '.join(combined_insights) if combined_insights else "기본 이미지 분석 완료"
    print(f"  종합 분석: {result['combined_analysis']}")
    
    return result

def analyze_user_images():
    """사용자 이미지 폴더 분석"""
    
    image_folder = "user_files/images"
    
    if not os.path.exists(image_folder):
        print("user_files/images 폴더가 없습니다.")
        return []
    
    image_files = []
    for file in os.listdir(image_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(image_folder, file))
    
    if not image_files:
        print("이미지 파일이 없습니다.")
        return []
    
    print(f"발견된 이미지: {len(image_files)}개")
    
    results = []
    for i, image_path in enumerate(image_files[:5], 1):  # 최대 5개만
        print(f"\n[{i}] 분석 중...")
        result = analyze_image_enhanced(image_path)
        results.append(result)
    
    return results

def main():
    """메인 실행"""
    
    print("=== 향상된 이미지 분석기 ===")
    print("OCR + 기본 이미지 분석을 수행합니다.")
    
    try:
        results = analyze_user_images()
        
        if results:
            print("\n" + "="*50)
            print("종합 분석 결과")
            print("="*50)
            
            total_ocr = 0
            
            for result in results:
                print(f"\n파일: {result['filename']}")
                
                if result['ocr_text']:
                    ocr_len = len(result['ocr_text'])
                    total_ocr += ocr_len
                    print(f"  OCR 텍스트: {ocr_len}글자")
                    
                    # 50글자까지만 미리보기
                    preview = result['ocr_text'][:50] + "..." if len(result['ocr_text']) > 50 else result['ocr_text']
                    print(f"  내용: {preview}")
                
                if result['image_info']:
                    print(f"  이미지 정보: {result['image_info']}")
                
                if result['combined_analysis']:
                    print(f"  종합 분석: {result['combined_analysis']}")
            
            print(f"\n전체 통계:")
            print(f"  분석 이미지: {len(results)}개")
            print(f"  총 OCR 텍스트: {total_ocr}글자")
            
            if total_ocr > 0:
                print(f"  평균 텍스트량: {total_ocr//len(results)}글자/이미지")
            
        else:
            print("\n분석할 이미지가 없습니다.")
            print("user_files/images/ 폴더에 이미지를 넣어주세요.")
    
    except KeyboardInterrupt:
        print("\n분석이 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")

if __name__ == "__main__":
    main()