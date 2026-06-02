import whisper
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


INSECT_SCIENTIFIC_NAMES = {
    "黄粉虫": "Tenebrio molitor",
    "黑粉虫": "Alphitobius diaperinus",
    "黑水虻": "Hermetia illucens",
    "家蝇": "Musca domestica",
    "果蝇": "Drosophila melanogaster",
    "蚕": "Bombyx mori",
    "蝗虫": "Locusta migratoria",
    "蟋蟀": "Acheta domesticus",
    "蚱蜢": "Schistocerca gregaria",
    "蜜蜂": "Apis mellifera",
}


class WhisperTranscriber:
    """使用Whisper进行语音识别，重点识别昆虫学名"""

    def __init__(self, model_size: str = "medium", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"加载Whisper模型: {model_size}, 设备: {device}")
        self.model = whisper.load_model(model_size, device=device)
        self.insect_names_zh = list(INSECT_SCIENTIFIC_NAMES.keys())
        self.insect_names_latin = list(INSECT_SCIENTIFIC_NAMES.values())

    def transcribe(
        self,
        audio_path: str,
        language: str = "zh",
        task: str = "transcribe"
    ) -> Dict:
        """
        转录音频为文本
        返回包含文本、时间戳等信息的字典
        """
        logger.info(f"开始转录音频: {audio_path}")

        options = {
            "language": language,
            "task": task,
            "verbose": False,
            "initial_prompt": self._build_prompt()
        }

        result = self.model.transcribe(audio_path, **options)

        segments = []
        for segment in result["segments"]:
            segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
                "insect_terms": self._extract_insect_terms(segment["text"])
            })

        full_text = " ".join([seg["text"] for seg in segments])

        logger.info(f"转录完成，共 {len(segments)} 个片段")

        return {
            "text": full_text,
            "language": result["language"],
            "segments": segments,
            "detected_insects": self._extract_all_insects(full_text)
        }

    def _build_prompt(self) -> str:
        """构建提示词，帮助识别昆虫学名"""
        prompt_parts = [
            "这是一场关于昆虫蛋白产业化的专业会议讨论。",
            "可能会提到以下昆虫及其学名："
        ]
        for zh, latin in INSECT_SCIENTIFIC_NAMES.items():
            prompt_parts.append(f"{zh}({latin})")
        prompt_parts.append("请准确识别昆虫的中文名称和拉丁学名。")
        return " ".join(prompt_parts)

    def _extract_insect_terms(self, text: str) -> List[Dict[str, str]]:
        """从文本中提取昆虫相关术语"""
        found_terms = []
        for zh_name, latin_name in INSECT_SCIENTIFIC_NAMES.items():
            if zh_name in text or latin_name.lower() in text.lower():
                found_terms.append({
                    "chinese_name": zh_name,
                    "latin_name": latin_name
                })
        return found_terms

    def _extract_all_insects(self, text: str) -> List[Dict[str, str]]:
        """提取文本中所有提到的昆虫"""
        insects = []
        seen = set()
        for zh_name, latin_name in INSECT_SCIENTIFIC_NAMES.items():
            if zh_name in text or latin_name.lower() in text.lower():
                key = latin_name.lower()
                if key not in seen:
                    seen.add(key)
                    insects.append({
                        "chinese_name": zh_name,
                        "latin_name": latin_name
                    })
        return insects

    def transcribe_with_timestamps(self, audio_path: str) -> List[Dict]:
        """获取带时间戳的转录结果"""
        result = self.transcribe(audio_path)
        return result["segments"]
