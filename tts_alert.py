"""
Module: tts_alert.py
โมดูลสำหรับสังเคราะห์เสียงเตือนด้วย AI for Thai VAJA API
Endpoint: https://api.aiforthai.in.th/vaja
พร้อมระบบดาวน์โหลด เล่นเสียง ป้องกันเสียงเตือนซ้ำ (Cooldown) และห้ามเล่นเสียงซ้อนกัน (Lock)
"""

import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Tuple
import requests

from config import Config

logger = logging.getLogger(__name__)


class VajaAlertService:
    """ระบบแจ้งเตือนด้วยเสียงผ่าน VAJA API ของ AI for Thai"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        speaker: Optional[str] = None,
        text: Optional[str] = None,
        cooldown_seconds: Optional[float] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.api_key = api_key or Config.API_KEY
        self.api_url = api_url or Config.VAJA_TTS_URL
        self.speaker = speaker or Config.VAJA_SPEAKER
        self.text = text or Config.VAJA_TEXT
        self.cooldown_seconds = cooldown_seconds or Config.ALERT_COOLDOWN_SECONDS
        self.cache_dir = cache_dir or Config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._is_playing = False
        self._last_alert_time = 0.0

        # ชื่อไฟล์แคชสำหรับเสียงเตือนนี้โดยเฉพาะ
        self._cached_audio_path = self.cache_dir / f"vaja_{self.speaker}_alert.wav"

        # เตรียม pygame.mixer
        self._init_pygame_mixer()

    def _init_pygame_mixer(self) -> bool:
        """เตรียมความพร้อมระบบเสียง pygame ถ้าติดตั้งไว้"""
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            return True
        except Exception as e:
            logger.warning(f"ไม่สามารถเริ่มการทำงานของ pygame.mixer ได้: {e}")
            return False

    def trigger_alert_async(self) -> bool:
        """
        ส่งคำสั่งเล่นเสียงเตือนใน Background Thread
        คืนค่า True หากรับคำสั่งสำเร็จ, False หากติด Cooldown หรือกำลังเล่นเสียงอยู่
        """
        current_time = time.time()

        # 1. ป้องกันการเล่นเสียงซ้อนกัน
        if self._is_playing:
            return False

        # 2. ป้องกันเสียงเตือนซ้ำเร็วเกินไป (Cooldown)
        if (current_time - self._last_alert_time) < self.cooldown_seconds:
            return False

        # เริ่มเธรดสำหรับเล่นเสียง
        thread = threading.Thread(target=self._play_alert_worker, daemon=True)
        thread.start()
        return True

    def _play_alert_worker(self):
        """Worker ภายใน Background Thread สำหรับดาวน์โหลดและเล่นเสียง"""
        with self._lock:
            if self._is_playing:
                return
            self._is_playing = True
            self._last_alert_time = time.time()

        try:
            audio_path = self._get_or_download_audio()
            if audio_path and os.path.exists(audio_path):
                self._play_audio_file(audio_path)
            else:
                logger.error("ไม่สามารถจัดเตรียมไฟล์เสียงเตือนได้")
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการเล่นเสียงเตือน: {e}")
        finally:
            self._is_playing = False

    def _get_or_download_audio(self) -> Optional[str]:
        """
        ดึงไฟล์เสียง: ถ้ามีไฟล์แคชที่ดาวน์โหลดไว้แล้วจะนำมาใช้ทันที
        หากยังไม่มีจะเรียก VAJA API แล้วดาวน์โหลดจาก audio_url
        """
        # ถ้าเปิดระบบแคชและมีไฟล์เดิมอยู่แล้ว ให้ใช้ไฟล์เดิมทันทีเพื่อความรวดเร็วและประหยัดโควตา
        if Config.ENABLE_AUDIO_CACHE and self._cached_audio_path.exists() and self._cached_audio_path.stat().st_size > 0:
            return str(self._cached_audio_path)

        # ตรวจสอบ API Key
        if not self.api_key or self.api_key in ("your_actual_api_key_here", "your_aiforthai_api_key_here"):
            logger.warning("ยังไม่ได้ตั้งค่า API Key ใน .env สำหรับ VAJA API")
            return None

        headers = {
            "Apikey": self.api_key
        }

        audio_url = None
        try:
            # เรียก VAJA API: AI for Thai กำหนดให้ส่ง JSON payload: {"text": "...", "speaker": "..."}
            payload = {
                "text": self.text,
                "speaker": self.speaker
            }

            response = requests.post(self.api_url, headers=headers, json=payload, timeout=10.0)

            if response.status_code == 200:
                res_json = response.json()
                audio_url = res_json.get("audio_url") or res_json.get("wav_url")
            else:
                logger.error(f"VAJA API error status: {response.status_code}, {response.text}")
                return None

        except Exception as e:
            logger.error(f"เรียก VAJA API ล้มเหลว: {e}")
            return None

        # ถ้าได้ audio_url ให้ดาวน์โหลดไฟล์เสียง
        if audio_url:
            try:
                # ดาวน์โหลดเสียงจาก URL
                dl_headers = {"Apikey": self.api_key}
                dl_resp = requests.get(audio_url, headers=dl_headers, timeout=12.0)
                if dl_resp.status_code != 200:
                    # บาง Endpoint เป็น public cdn ไม่ต้องใช้ Apikey
                    dl_resp = requests.get(audio_url, timeout=12.0)

                if dl_resp.status_code == 200 and len(dl_resp.content) > 0:
                    with open(self._cached_audio_path, "wb") as f:
                        f.write(dl_resp.content)
                    return str(self._cached_audio_path)
                else:
                    logger.error(f"ดาวน์โหลดไฟล์เสียงจาก {audio_url} ล้มเหลว: {dl_resp.status_code}")
            except Exception as e:
                logger.error(f"ดาวน์โหลดไฟล์เสียงล้มเหลว: {e}")

        return None

    def _play_audio_file(self, file_path: str):
        """เล่นไฟล์เสียงบนระบบปฏิบัติการ Windows อย่างปลอดภัยและบล็อกเฉพาะในเธรด"""
        # วิธีที่ 1: ใช้ Pygame
        try:
            import pygame
            if pygame.mixer.get_init():
                sound = pygame.mixer.Sound(file_path)
                channel = sound.play()
                while channel and channel.get_busy():
                    time.sleep(0.05)
                return
        except Exception as e:
            logger.debug(f"Pygame playback error: {e}, attempting Windows fallback...")

        # วิธีที่ 2: ใช้ Windows Native winsound (สำหรับไฟล์ WAV)
        try:
            import winsound
            winsound.PlaySound(file_path, winsound.SND_FILENAME)
            return
        except Exception as e:
            logger.debug(f"winsound playback error: {e}, attempting PowerShell fallback...")

        # วิธีที่ 3: ใช้ Windows PowerShell สั่งเล่นเสียง (รองรับทั้ง wav และ mp3)
        try:
            escaped_path = file_path.replace("'", "''")
            cmd = (
                f"powershell -c \""
                f"$player = New-Object System.Media.SoundPlayer '{escaped_path}';"
                f"$player.PlaySync();\""
            )
            subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
        except Exception as e:
            logger.error(f"ไม่สามารถเล่นเสียงผ่าน PowerShell ได้: {e}")

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def last_alert_time(self) -> float:
        return self._last_alert_time
