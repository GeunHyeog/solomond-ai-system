"""
솔로몬드 AI 프레임워크
멀티모달 분석을 위한 재사용 가능한 AI 엔진 모듈
"""

__version__ = "1.0.0"
__author__ = "SolomondAI Team"

from .engines.audio_engine import AudioEngine
from .engines.image_engine import ImageEngine  
from .engines.video_engine import VideoEngine
from .engines.text_engine import TextEngine
from .engines.integration_engine import IntegrationEngine

from .processors.file_processor import FileProcessor
from .processors.batch_processor import BatchProcessor

from .monitoring.performance_monitor import PerformanceMonitor
from .monitoring.progress_tracker import ProgressTracker

from .utils.config_manager import ConfigManager
from .utils.logger import setup_logger

class SolomondAI:
    """솔로몬드 AI 메인 클래스"""
    
    def __init__(self, domain="general", engines=None, ui_layout="four_step", theme="default"):
        self.domain = domain
        self.engines = engines or ["audio", "image", "text"]
        self.ui_layout = ui_layout
        self.theme = theme
        
        # 설정 관리자 초기화
        self.config = ConfigManager()
        
        # 엔진들 초기화
        self._init_engines()
        
        # 성능 모니터링 시작
        self.monitor = PerformanceMonitor()
        
    def _init_engines(self):
        """분석 엔진들 초기화"""
        self.engine_instances = {}
        
        if "audio" in self.engines:
            self.engine_instances["audio"] = AudioEngine()
            
        if "image" in self.engines:
            self.engine_instances["image"] = ImageEngine()
            
        if "video" in self.engines:
            self.engine_instances["video"] = VideoEngine()
            
        if "text" in self.engines:
            self.engine_instances["text"] = TextEngine()
            
        # 통합 엔진 (교차 검증용)
        self.integration_engine = IntegrationEngine(self.engine_instances)
    
    @classmethod
    def from_config(cls, config_path: str):
        """설정 파일로부터 초기화"""
        config = ConfigManager.load_from_file(config_path)
        return cls(
            domain=config.get("project", {}).get("domain", "general"),
            engines=list(config.get("engines", {}).keys()),
            ui_layout=config.get("ui", {}).get("layout", "four_step"),
            theme=config.get("ui", {}).get("theme", "default")
        )
    
    def analyze(self, files):
        """파일들을 분석하여 결과 반환"""
        results = {}
        
        # 각 엔진별로 분석 실행
        for engine_name, engine in self.engine_instances.items():
            results[engine_name] = engine.analyze_files(files)
        
        # 교차 검증 및 통합 분석
        results["integration"] = self.integration_engine.cross_validate(results)
        
        return results
    
    def run(self, port: int = 8503):
        """웹 UI 실행"""
        from .ui.app_factory import create_streamlit_app
        app = create_streamlit_app(self)
        # Streamlit 실행 로직은 별도 구현
        print(f"Starting SolomondAI on port {port}")
        print(f"Domain: {self.domain}, Engines: {self.engines}")

__all__ = [
    "SolomondAI",
    "AudioEngine", 
    "ImageEngine",
    "VideoEngine", 
    "TextEngine",
    "IntegrationEngine",
    "FileProcessor",
    "BatchProcessor", 
    "PerformanceMonitor",
    "ProgressTracker",
    "ConfigManager"
]