# -*- coding: utf-8 -*-
"""
통합 스토리 빌더 - 모든 파일의 내용을 하나의 이야기로 결합
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# 프로젝트 경로 추가  
sys.path.append(str(Path(__file__).parent))

class IntegratedStoryBuilder:
    """모든 파일의 내용을 통합하여 하나의 스토리 생성"""
    
    def __init__(self):
        self.extracted_contents = {
            'audio': [],
            'image': [],
            'video': []
        }
        
    def find_and_analyze_files(self):
        """사용자 파일들을 찾아서 내용 추출"""
        
        print("=== 모든 파일에서 내용 추출 중 ===")
        
        # 오디오 파일들 분석
        audio_folder = "user_files/audio"
        if os.path.exists(audio_folder):
            for file in os.listdir(audio_folder):
                if file.lower().endswith(('.wav', '.m4a', '.mp3')):
                    file_path = os.path.join(audio_folder, file)
                    print(f"\n🎤 오디오 분석: {file}")
                    content = self.extract_audio_content(file_path)
                    if content:
                        self.extracted_contents['audio'].append({
                            'filename': file,
                            'content': content,
                            'type': 'audio'
                        })
        
        # 이미지 파일들 분석
        image_folder = "user_files/images"
        if os.path.exists(image_folder):
            for file in os.listdir(image_folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(image_folder, file)
                    print(f"\n🖼️ 이미지 분석: {file}")
                    content = self.extract_image_content(file_path)
                    if content:
                        self.extracted_contents['image'].append({
                            'filename': file,
                            'content': content,
                            'type': 'image'
                        })
        
        # 비디오 파일들 분석 (시간이 오래 걸리므로 선택적)
        video_folder = "user_files/videos"
        if os.path.exists(video_folder):
            video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mov', '.mp4', '.avi'))]
            if video_files:
                print(f"\n🎬 비디오 파일 {len(video_files)}개 발견")
                answer = input("비디오도 분석하시겠습니까? (y/n): ").strip().lower()
                if answer == 'y':
                    for file in video_files[:2]:  # 최대 2개만
                        file_path = os.path.join(video_folder, file)
                        print(f"\n🎬 비디오 분석: {file}")
                        content = self.extract_video_content(file_path)
                        if content:
                            self.extracted_contents['video'].append({
                                'filename': file,
                                'content': content,
                                'type': 'video'
                            })
    
    def extract_audio_content(self, file_path):
        """오디오에서 텍스트 추출"""
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(file_path, language="ko")
            text = result["text"].strip()
            
            if text:
                print(f"  추출: {len(text)}글자")
                print(f"  내용: {text[:100]}...")
                return text
            else:
                print("  실패: 텍스트 없음")
                return None
                
        except Exception as e:
            print(f"  오류: {str(e)}")
            return None
    
    def extract_image_content(self, file_path):
        """이미지에서 텍스트 추출"""
        try:
            import easyocr
            reader = easyocr.Reader(['ko', 'en'])
            results = reader.readtext(file_path)
            
            if results:
                text = ' '.join([item[1] for item in results])
                print(f"  추출: {len(text)}글자")
                print(f"  내용: {text[:100]}...")
                return text
            else:
                print("  실패: 텍스트 없음")
                return None
                
        except Exception as e:
            print(f"  오류: {str(e)}")
            return None
    
    def extract_video_content(self, file_path):
        """비디오에서 오디오 추출 후 텍스트 변환"""
        try:
            # 임시로 오디오만 추출하는 간단한 방법
            import subprocess
            import tempfile
            
            # 임시 오디오 파일 생성
            temp_audio = tempfile.mktemp(suffix='.wav')
            
            # FFmpeg로 오디오 추출
            cmd = [
                'ffmpeg', '-i', file_path, 
                '-vn', '-acodec', 'pcm_s16le', 
                '-ar', '16000', '-ac', '1',
                temp_audio
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Whisper로 텍스트 변환
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(temp_audio, language="ko")
            text = result["text"].strip()
            
            # 임시 파일 삭제
            if os.path.exists(temp_audio):
                os.unlink(temp_audio)
            
            if text:
                print(f"  추출: {len(text)}글자")
                print(f"  내용: {text[:100]}...")
                return text
            else:
                print("  실패: 텍스트 없음")
                return None
                
        except Exception as e:
            print(f"  오류: {str(e)}")
            return None
    
    def build_integrated_story(self):
        """모든 추출된 내용을 하나의 이야기로 통합"""
        
        print("\n" + "="*60)
        print("🎭 통합 스토리 생성 중...")
        print("="*60)
        
        # 모든 내용 수집
        all_contents = []
        
        for audio_item in self.extracted_contents['audio']:
            all_contents.append({
                'source': f"🎤 {audio_item['filename']}",
                'content': audio_item['content'],
                'type': 'audio'
            })
        
        for image_item in self.extracted_contents['image']:
            all_contents.append({
                'source': f"🖼️ {image_item['filename']}",
                'content': image_item['content'],
                'type': 'image'
            })
        
        for video_item in self.extracted_contents['video']:
            all_contents.append({
                'source': f"🎬 {video_item['filename']}",
                'content': video_item['content'],
                'type': 'video'
            })
        
        if not all_contents:
            print("❌ 추출된 내용이 없습니다.")
            return None
        
        # 스토리 구성
        story = self.create_coherent_story(all_contents)
        
        return story
    
    def create_coherent_story(self, contents):
        """추출된 내용들을 논리적으로 연결하여 스토리 생성"""
        
        print(f"📝 {len(contents)}개 소스에서 통합 스토리 생성...")
        
        # 기본 스토리 구조
        story = {
            'title': '통합 분석 결과',
            'summary': '',
            'detailed_story': '',
            'sources': contents,
            'insights': [],
            'timeline': []
        }
        
        # 모든 텍스트 결합
        combined_text = ""
        source_info = []
        
        for item in contents:
            combined_text += f"\n[{item['source']}에서 추출]\n{item['content']}\n"
            source_info.append(f"- {item['source']}: {len(item['content'])}글자")
        
        # 간단한 요약 생성
        story['summary'] = self.generate_simple_summary(combined_text)
        
        # 상세 스토리 구성
        story['detailed_story'] = self.construct_detailed_story(contents)
        
        # 인사이트 추출
        story['insights'] = self.extract_insights(combined_text)
        
        # 소스 정보
        story['source_summary'] = source_info
        
        return story
    
    def generate_simple_summary(self, text):
        """간단한 요약 생성"""
        
        # 길이 기반 요약
        if len(text) > 500:
            # 첫 200자 + 중간 부분 + 마지막 200자
            summary = text[:200] + " ... " + text[-200:]
        else:
            summary = text
        
        return summary.strip()
    
    def construct_detailed_story(self, contents):
        """상세 스토리 구성"""
        
        story_parts = []
        story_parts.append("📋 통합 분석 결과\n")
        story_parts.append("=" * 50)
        
        # 각 소스별로 정리
        for i, item in enumerate(contents, 1):
            story_parts.append(f"\n{i}. {item['source']}")
            story_parts.append("-" * 30)
            
            # 내용을 적절히 줄여서 표시
            content = item['content']
            if len(content) > 300:
                content = content[:300] + "..."
            
            story_parts.append(content)
            story_parts.append("")  # 빈 줄
        
        return "\n".join(story_parts)
    
    def extract_insights(self, text):
        """간단한 인사이트 추출"""
        
        insights = []
        
        # 키워드 기반 인사이트
        if "Global" in text or "cultura" in text:
            insights.append("🌍 국제적 또는 문화적 내용이 포함되어 있습니다")
        
        if "2025" in text:
            insights.append("📅 2025년과 관련된 내용입니다")
        
        if "Rise" in text or "Eco" in text:
            insights.append("📈 성장 또는 친환경과 관련된 주제입니다")
        
        if len(text) > 1000:
            insights.append("📊 상당한 양의 정보가 포함되어 있습니다")
        
        if not insights:
            insights.append("💭 다양한 형태의 멀티미디어 콘텐츠가 분석되었습니다")
        
        return insights
    
    def save_story(self, story):
        """스토리를 파일로 저장"""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"integrated_story_{timestamp}.md"
        
        content = f"""# {story['title']}
생성일시: {time.strftime("%Y-%m-%d %H:%M:%S")}

## 📋 요약
{story['summary']}

## 🔍 주요 인사이트
"""
        
        for insight in story['insights']:
            content += f"- {insight}\n"
        
        content += f"""
## 📊 소스 정보
"""
        
        for source in story['source_summary']:
            content += f"{source}\n"
        
        content += f"""
## 📖 상세 내용
{story['detailed_story']}

---
*솔로몬드 AI 통합 스토리 빌더로 생성됨*
"""
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"\n💾 스토리 저장: {filename}")
        except Exception as e:
            print(f"\n❌ 저장 실패: {str(e)}")
        
        return filename

def main():
    """메인 실행"""
    
    print("=== 솔로몬드 AI 통합 스토리 빌더 ===")
    print("모든 파일의 내용을 하나의 이야기로 만듭니다.")
    
    try:
        builder = IntegratedStoryBuilder()
        
        # 1. 모든 파일 분석
        builder.find_and_analyze_files()
        
        # 2. 통합 스토리 생성
        story = builder.build_integrated_story()
        
        if story:
            # 3. 결과 표시
            print("\n" + "="*60)
            print("🎭 생성된 통합 스토리")
            print("="*60)
            
            print(f"\n📋 제목: {story['title']}")
            print(f"\n📝 요약:\n{story['summary']}")
            
            print(f"\n🔍 주요 인사이트:")
            for insight in story['insights']:
                print(f"  {insight}")
            
            print(f"\n📊 분석된 소스:")
            for source in story['source_summary']:
                print(f"  {source}")
            
            # 4. 파일 저장
            filename = builder.save_story(story)
            
            print(f"\n✅ 통합 스토리 생성 완료!")
            print(f"📁 저장된 파일: {filename}")
            
        else:
            print("\n❌ 스토리 생성에 실패했습니다.")
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자가 중단했습니다.")
    except Exception as e:
        print(f"\n💥 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()