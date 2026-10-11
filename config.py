"""
Module: config.py
จัดการการโหลดการตั้งค่าระบบและ API Keys จากไฟล์ .env
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# หาตำแหน่งโฟลเดอร์ Root ของโปรเจกต์
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# โหลดค่าจากไฟล์ .env
load_dotenv(dotenv_path=ENV_PATH, override=True)


class Config:
    """การตั้งค่าระบบส่วนกลาง"""

    # AI for Thai API
    API_KEY = os.getenv("AIFORTHAI_API_KEY", "").strip()
    MASK_DETECTION_URL = os.getenv(
        "MASK_DETECTION_URL", "https://api.aiforthai.in.th/detectfacemask"
    ).strip()
    VAJA_TTS_URL = os.getenv(
        "VAJA_TTS_URL", "https://api.aiforthai.in.th/vaja"
    ).strip()
    VAJA_SPEAKER = os.getenv("VAJA_SPEAKER", "nana").strip()
    VAJA_TEXT = os.getenv("VAJA_TEXT", "กรุณาถอดหน้ากาก").strip()

    # Hardware & Performance
    CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
    FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "1280"))
    FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "720"))
    CHECK_INTERVAL_SECONDS = float(os.getenv("CHECK_INTERVAL_SECONDS", "5.0"))
    ALERT_COOLDOWN_SECONDS = float(os.getenv("ALERT_COOLDOWN_SECONDS", "6.0"))
    ENABLE_AUDIO_CACHE = (
        os.getenv("ENABLE_AUDIO_CACHE", "true").lower() in ("true", "1", "yes")
    )

    # Directories
    CACHE_DIR = BASE_DIR / "cache"

    @classmethod
    def validate_api_key(cls) -> bool:
        """ตรวจสอบว่ามีการระบุ API Key แล้วหรือยัง"""
        if not cls.API_KEY or cls.API_KEY in ("your_actual_api_key_here", "your_aiforthai_api_key_here"):
            return False
        return True


# สร้างโฟลเดอร์ Cache ถ้ายังไม่มี
Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
