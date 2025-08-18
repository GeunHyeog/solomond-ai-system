#!/usr/bin/env python3
"""
🎯 SOLOMOND AI - 사용자 친화적 GUI 애플리케이션
포트 문제 완전 우회한 데스크톱 앱
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import json
import os
import time
from pathlib import Path
from datetime import datetime
import sys

class SolomondAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 SOLOMOND AI - 지능형 분석 시스템")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # 변수 초기화
        self.files_to_analyze = []
        self.analysis_results = []
        self.is_analyzing = False
        
        self.setup_gui()
        
    def setup_gui(self):
        """GUI 구성"""
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 타이틀
        title_label = tk.Label(main_frame, text="🤖 SOLOMOND AI", 
                             font=("Arial", 24, "bold"), 
                             fg="#2563eb", bg="#f0f0f0")
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        subtitle_label = tk.Label(main_frame, text="지능형 다각도 분석 시스템", 
                                font=("Arial", 12), 
                                fg="#6b7280", bg="#f0f0f0")
        subtitle_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # 좌측 패널 - 파일 관리
        left_frame = ttk.LabelFrame(main_frame, text="📁 파일 관리", padding="15")
        left_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # 파일 추가 버튼들
        ttk.Button(left_frame, text="📂 파일 선택", 
                  command=self.select_files, 
                  style="Accent.TButton").grid(row=0, column=0, sticky=tk.W+tk.E, pady=2)
        
        ttk.Button(left_frame, text="📁 폴더 선택", 
                  command=self.select_folder, 
                  style="Accent.TButton").grid(row=1, column=0, sticky=tk.W+tk.E, pady=2)
        
        ttk.Button(left_frame, text="🗂️ user_files 스캔", 
                  command=self.scan_user_files, 
                  style="Accent.TButton").grid(row=2, column=0, sticky=tk.W+tk.E, pady=2)
        
        # 파일 목록
        ttk.Label(left_frame, text="선택된 파일:").grid(row=3, column=0, sticky=tk.W, pady=(10, 5))
        
        self.file_listbox = tk.Listbox(left_frame, height=10, width=40)
        self.file_listbox.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=2)
        
        # 파일 목록 스크롤바
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        scrollbar.grid(row=4, column=1, sticky=(tk.N, tk.S))
        self.file_listbox.configure(yscrollcommand=scrollbar.set)
        
        # 파일 제거 버튼
        ttk.Button(left_frame, text="❌ 선택 제거", 
                  command=self.remove_selected_file).grid(row=5, column=0, sticky=tk.W+tk.E, pady=5)
        
        # 우측 패널 - 분석 및 결과
        right_frame = ttk.LabelFrame(main_frame, text="🤖 AI 분석", padding="15")
        right_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 분석 옵션
        options_frame = ttk.LabelFrame(right_frame, text="분석 옵션", padding="10")
        options_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.image_ocr_var = tk.BooleanVar(value=True)
        self.audio_stt_var = tk.BooleanVar(value=True)
        self.video_analysis_var = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(options_frame, text="🖼️ 이미지 텍스트 추출 (OCR)", 
                       variable=self.image_ocr_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(options_frame, text="🎵 음성 텍스트 변환 (STT)", 
                       variable=self.audio_stt_var).grid(row=1, column=0, sticky=tk.W)
        ttk.Checkbutton(options_frame, text="🎬 비디오 분석 (기본정보)", 
                       variable=self.video_analysis_var).grid(row=2, column=0, sticky=tk.W)
        
        # 분석 시작 버튼
        self.analyze_button = tk.Button(right_frame, text="🚀 AI 분석 시작", 
                                       font=("Arial", 14, "bold"),
                                       bg="#10b981", fg="white",
                                       command=self.start_analysis,
                                       height=2)
        self.analyze_button.grid(row=1, column=0, sticky=tk.W+tk.E, pady=10)
        
        # 진행률 표시
        ttk.Label(right_frame, text="분석 진행률:").grid(row=2, column=0, sticky=tk.W)
        self.progress = ttk.Progressbar(right_frame, length=300, mode='determinate')
        self.progress.grid(row=3, column=0, sticky=tk.W+tk.E, pady=2)
        
        self.status_label = tk.Label(right_frame, text="대기 중...", 
                                    fg="#6b7280", bg="#f0f0f0")
        self.status_label.grid(row=4, column=0, sticky=tk.W, pady=2)
        
        # 결과 표시 영역
        results_frame = ttk.LabelFrame(main_frame, text="📊 분석 결과", padding="15")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(20, 0))
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=80)
        self.results_text.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 결과 관리 버튼들
        buttons_frame = ttk.Frame(results_frame)
        buttons_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(buttons_frame, text="💾 결과 저장", 
                  command=self.save_results).grid(row=0, column=0, padx=5)
        ttk.Button(buttons_frame, text="📋 결과 복사", 
                  command=self.copy_results).grid(row=0, column=1, padx=5)
        ttk.Button(buttons_frame, text="🗑️ 결과 지우기", 
                  command=self.clear_results).grid(row=0, column=2, padx=5)
        
        # 하단 상태바
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.bottom_status = tk.Label(status_frame, 
                                     text="✅ SOLOMOND AI 준비 완료 | 파일을 선택하고 분석을 시작하세요", 
                                     fg="#059669", bg="#f0f0f0")
        self.bottom_status.grid(row=0, column=0, sticky=tk.W)
        
        # 그리드 가중치 설정
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
    
    def select_files(self):
        """파일 선택"""
        files = filedialog.askopenfilenames(
            title="분석할 파일을 선택하세요",
            filetypes=[
                ("이미지 파일", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
                ("오디오 파일", "*.wav *.mp3 *.m4a *.flac *.ogg"),
                ("비디오 파일", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("모든 파일", "*.*")
            ]
        )
        
        for file in files:
            if file not in self.files_to_analyze:
                self.files_to_analyze.append(file)
                self.file_listbox.insert(tk.END, os.path.basename(file))
        
        self.update_status()
    
    def select_folder(self):
        """폴더 선택"""
        folder = filedialog.askdirectory(title="분석할 파일이 있는 폴더를 선택하세요")
        if folder:
            self.scan_folder(folder)
    
    def scan_user_files(self):
        """user_files 폴더 스캔"""
        user_files_path = Path("user_files")
        if user_files_path.exists():
            self.scan_folder(str(user_files_path))
        else:
            messagebox.showwarning("경고", "user_files 폴더를 찾을 수 없습니다.")
    
    def scan_folder(self, folder_path):
        """폴더 내 파일 스캔"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.wav', '.mp3', '.m4a', '.mp4', '.mov', '.avi']
        
        folder = Path(folder_path)
        found_files = []
        
        for ext in extensions:
            found_files.extend(folder.rglob(f"*{ext}"))
            found_files.extend(folder.rglob(f"*{ext.upper()}"))
        
        added_count = 0
        for file in found_files:
            file_str = str(file)
            if file_str not in self.files_to_analyze:
                self.files_to_analyze.append(file_str)
                self.file_listbox.insert(tk.END, file.name)
                added_count += 1
        
        if added_count > 0:
            messagebox.showinfo("파일 추가", f"{added_count}개 파일이 추가되었습니다.")
        else:
            messagebox.showinfo("알림", "새로 추가된 파일이 없습니다.")
        
        self.update_status()
    
    def remove_selected_file(self):
        """선택된 파일 제거"""
        selection = self.file_listbox.curselection()
        if selection:
            index = selection[0]
            self.file_listbox.delete(index)
            del self.files_to_analyze[index]
            self.update_status()
    
    def update_status(self):
        """상태 업데이트"""
        file_count = len(self.files_to_analyze)
        if file_count == 0:
            self.bottom_status.config(text="📁 파일을 선택해주세요")
        else:
            self.bottom_status.config(text=f"📁 {file_count}개 파일 선택됨 | 분석 준비 완료")
    
    def start_analysis(self):
        """분석 시작"""
        if not self.files_to_analyze:
            messagebox.showwarning("경고", "분석할 파일을 먼저 선택해주세요.")
            return
        
        if self.is_analyzing:
            messagebox.showwarning("경고", "분석이 이미 진행 중입니다.")
            return
        
        # UI 상태 변경
        self.is_analyzing = True
        self.analyze_button.config(text="⏳ 분석 중...", state="disabled", bg="#6b7280")
        self.progress['value'] = 0
        self.results_text.delete(1.0, tk.END)
        
        # 별도 스레드에서 분석 실행
        analysis_thread = threading.Thread(target=self.run_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def run_analysis(self):
        """실제 분석 실행"""
        try:
            self.analysis_results = []
            total_files = len(self.files_to_analyze)
            
            self.update_status_safe("🚀 AI 분석 시작...")
            
            for i, file_path in enumerate(self.files_to_analyze):
                # 진행률 업데이트
                progress = (i / total_files) * 100
                self.progress['value'] = progress
                
                file_name = os.path.basename(file_path)
                self.update_status_safe(f"📋 [{i+1}/{total_files}] {file_name} 분석 중...")
                
                # 파일 분석
                result = self.analyze_single_file(file_path)
                self.analysis_results.append(result)
                
                # 결과 실시간 표시
                self.display_single_result(result)
                
                time.sleep(0.1)  # UI 반응성
            
            # 분석 완료
            self.progress['value'] = 100
            self.update_status_safe(f"✅ 분석 완료! ({total_files}개 파일)")
            
            # 최종 보고서 생성
            self.generate_final_report()
            
        except Exception as e:
            self.update_status_safe(f"❌ 분석 중 오류: {str(e)}")
        
        finally:
            # UI 상태 복원
            self.is_analyzing = False
            self.root.after(0, self.reset_analyze_button)
    
    def analyze_single_file(self, file_path):
        """단일 파일 분석"""
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        result = {
            'file': file_path.name,
            'path': str(file_path),
            'type': 'unknown',
            'size_mb': round(file_path.stat().st_size / (1024*1024), 2),
            'status': 'processed'
        }
        
        try:
            # 이미지 분석
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'] and self.image_ocr_var.get():
                result.update(self.analyze_image(file_path))
            
            # 오디오 분석  
            elif file_ext in ['.wav', '.mp3', '.m4a', '.flac', '.ogg'] and self.audio_stt_var.get():
                result.update(self.analyze_audio(file_path))
            
            # 비디오 분석
            elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv'] and self.video_analysis_var.get():
                result.update(self.analyze_video(file_path))
            
            else:
                result.update({
                    'type': 'basic_info',
                    'note': '기본 정보만 수집됨'
                })
                
        except Exception as e:
            result.update({
                'status': 'error',
                'error': str(e)
            })
        
        return result
    
    def analyze_image(self, file_path):
        """이미지 OCR 분석"""
        try:
            import easyocr
            reader = easyocr.Reader(['en', 'ko'], gpu=False)
            
            ocr_results = reader.readtext(str(file_path))
            extracted_text = ' '.join([r[1] for r in ocr_results if r[2] > 0.3])
            
            return {
                'type': 'image_ocr',
                'extracted_text': extracted_text,
                'text_blocks': len(ocr_results),
                'avg_confidence': round(sum(r[2] for r in ocr_results) / len(ocr_results), 3) if ocr_results else 0
            }
        except Exception as e:
            return {
                'type': 'image_basic',
                'error': f'OCR 실패: {str(e)}'
            }
    
    def analyze_audio(self, file_path):
        """오디오 STT 분석"""
        try:
            import whisper
            model = whisper.load_model("tiny")
            result = model.transcribe(str(file_path), language='ko')
            
            return {
                'type': 'audio_stt',
                'transcript': result['text'],
                'language': result.get('language', 'unknown'),
                'segments': len(result.get('segments', []))
            }
        except Exception as e:
            return {
                'type': 'audio_basic',
                'error': f'STT 실패: {str(e)}'
            }
    
    def analyze_video(self, file_path):
        """비디오 기본 분석"""
        return {
            'type': 'video_basic',
            'note': '비디오 파일 감지됨 (기본 정보만)'
        }
    
    def display_single_result(self, result):
        """단일 결과 표시"""
        def update_display():
            self.results_text.insert(tk.END, f"\n📄 {result['file']} ({result['size_mb']} MB)\n")
            
            if result.get('extracted_text'):
                preview = result['extracted_text'][:100] + "..." if len(result['extracted_text']) > 100 else result['extracted_text']
                self.results_text.insert(tk.END, f"   🔍 추출 텍스트: {preview}\n")
            
            if result.get('transcript'):
                preview = result['transcript'][:100] + "..." if len(result['transcript']) > 100 else result['transcript']
                self.results_text.insert(tk.END, f"   🎵 음성 인식: {preview}\n")
            
            if result.get('error'):
                self.results_text.insert(tk.END, f"   ⚠️ 오류: {result['error']}\n")
            
            self.results_text.see(tk.END)
        
        self.root.after(0, update_display)
    
    def generate_final_report(self):
        """최종 보고서 생성"""
        def update_report():
            self.results_text.insert(tk.END, f"\n{'='*50}\n")
            self.results_text.insert(tk.END, f"🎉 분석 완료 보고서\n")
            self.results_text.insert(tk.END, f"{'='*50}\n")
            self.results_text.insert(tk.END, f"📊 총 파일: {len(self.analysis_results)}개\n")
            
            success_count = len([r for r in self.analysis_results if r.get('status') != 'error'])
            self.results_text.insert(tk.END, f"✅ 성공: {success_count}개\n")
            self.results_text.insert(tk.END, f"❌ 실패: {len(self.analysis_results) - success_count}개\n")
            
            # 타입별 통계
            type_counts = {}
            for result in self.analysis_results:
                result_type = result.get('type', 'unknown')
                type_counts[result_type] = type_counts.get(result_type, 0) + 1
            
            self.results_text.insert(tk.END, f"\n📋 분석 타입별 통계:\n")
            for type_name, count in type_counts.items():
                self.results_text.insert(tk.END, f"   {type_name}: {count}개\n")
            
            self.results_text.insert(tk.END, f"\n💡 결과 저장: 💾 결과 저장 버튼 클릭\n")
            self.results_text.see(tk.END)
        
        self.root.after(0, update_report)
    
    def update_status_safe(self, message):
        """스레드 안전 상태 업데이트"""
        def update():
            self.status_label.config(text=message)
        
        self.root.after(0, update)
    
    def reset_analyze_button(self):
        """분석 버튼 상태 복원"""
        self.analyze_button.config(text="🚀 AI 분석 시작", state="normal", bg="#10b981")
    
    def save_results(self):
        """결과 저장"""
        if not self.analysis_results:
            messagebox.showwarning("경고", "저장할 결과가 없습니다.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON 파일", "*.json"), ("텍스트 파일", "*.txt")],
            title="분석 결과 저장"
        )
        
        if file_path:
            try:
                report_data = {
                    'analysis_date': datetime.now().isoformat(),
                    'total_files': len(self.analysis_results),
                    'results': self.analysis_results,
                    'app_version': 'SOLOMOND AI GUI v1.0'
                }
                
                if file_path.endswith('.json'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(report_data, f, indent=2, ensure_ascii=False)
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(self.results_text.get(1.0, tk.END))
                
                messagebox.showinfo("성공", f"결과가 저장되었습니다:\n{file_path}")
            except Exception as e:
                messagebox.showerror("오류", f"저장 실패: {str(e)}")
    
    def copy_results(self):
        """결과 클립보드 복사"""
        try:
            results_text = self.results_text.get(1.0, tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(results_text)
            messagebox.showinfo("성공", "결과가 클립보드에 복사되었습니다.")
        except Exception as e:
            messagebox.showerror("오류", f"복사 실패: {str(e)}")
    
    def clear_results(self):
        """결과 지우기"""
        self.results_text.delete(1.0, tk.END)
        self.analysis_results = []

def main():
    # GUI 애플리케이션 실행
    root = tk.Tk()
    
    # 스타일 설정
    style = ttk.Style()
    style.theme_use('clam')
    
    app = SolomondAIApp(root)
    
    # 애플리케이션 실행
    root.mainloop()

if __name__ == "__main__":
    main()