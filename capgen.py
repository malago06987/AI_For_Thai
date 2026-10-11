"""
Module: capgen.py
โมดูลสำหรับสร้างคำบรรยายภาพ (Image Captioning)
ข้อกำหนดสำคัญ:
- ห้ามเรียก API จนกว่าจะตรวจสอบ URL ที่ถูกต้องได้
- ห้ามใช้ URL ตรวจจับหน้ากาก (detectfacemask) แทนโดยเด็ดขาด
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import requests
from config import Config

# ตั้งค่า stdout/stderr ให้รองรับ UTF-8 บน Windows Terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger(__name__)


class CapgenClient:
    """
    ไคลเอนต์สำหรับ Image Captioning (สร้างคำบรรยายภาพ)
    ออกแบบให้เป็นโมดูลแยกอิสระ พร้อมระบบป้องกันการเรียกใช้ Endpoint ที่ไม่ถูกต้อง
    """

    # รายการ Endpoint ต้องห้ามที่ห้ามนำมาใช้กับโมดูลนี้
    FORBIDDEN_ENDPOINTS = [
        "https://api.aiforthai.in.th/detectfacemask",
        "detectfacemask",
    ]

    def __init__(self, api_key: Optional[str] = None, caption_api_url: Optional[str] = None):
        self.api_key = api_key or Config.API_KEY
        self.caption_api_url = caption_api_url or ""
        self._is_url_verified = False

    def verify_and_set_endpoint(self, verified_url: str):
        """
        ตรวจสอบความถูกต้องของ URL ก่อนอนุญาตให้เรียกใช้งาน
        ห้ามใช้ URL ของ Face Mask Detection โดยเด็ดขาด
        """
        normalized_url = verified_url.strip().lower()

        # ตรวจสอบว่าไม่ตรงกับ URL ต้องห้าม
        for forbidden in self.FORBIDDEN_ENDPOINTS:
            if forbidden.lower() in normalized_url:
                raise ValueError(
                    f"ข้อผิดพลาด: ห้ามใช้ URL ตรวจจับหน้ากาก ({verified_url}) กับโมดูล Capgen โดยเด็ดขาด!"
                )

        if not normalized_url.startswith("http://") and not normalized_url.startswith("https://"):
            raise ValueError(f"URL ไม่ถูกต้อง: {verified_url}")

        self.caption_api_url = verified_url.strip()
        self._is_url_verified = True
        logger.info(f"ยืนยัน Endpoint สำหรับ Capgen สำเร็จ: {self.caption_api_url}")

    def generate_caption(self, image_input: Any, mock_mode: bool = False) -> Dict[str, Any]:
        """
        สร้างคำบรรยายภาพ
        - หากยังไม่มีการยืนยัน URL จะปฏิเสธการเรียกใช้งาน API ทันที
        - รองรับ mock_mode สำหรับการทดสอบ UI / Logic ก่อนเชื่อมต่อจริง
        """
        if mock_mode:
            return {
                "success": True,
                "caption": "ตัวอย่างคำบรรยายภาพ (Mock Caption): บุคคลกำลังยืนอยู่หน้ากล้อง",
                "is_mock": True,
                "error": None,
            }

        # ตรวจสอบความปลอดภัย: ป้องกันการเรียกใช้โดยที่ URL ยังไม่ได้รับการตรวจสอบ
        if not self._is_url_verified or not self.caption_api_url:
            return {
                "success": False,
                "caption": None,
                "is_mock": False,
                "error": (
                    "ยังไม่สามารถเรียกใช้งาน Capgen ได้ เนื่องจากยังไม่ได้รับการยืนยัน URL ที่ถูกต้อง "
                    "(กรุณาตรวจสอบ URL ที่ถูกต้องจากเอกสาร AI for Thai และเรียก verify_and_set_endpoint ก่อนใช้งาน)"
                ),
            }

        # เมื่อมี URL ที่ได้รับการตรวจสอบแล้ว โครงสร้างการเรียกใช้งานจะทำงานดังนี้:
        try:
            headers = {"Apikey": self.api_key}
            files = {"file": image_input}

            response = requests.post(
                self.caption_api_url,
                headers=headers,
                files=files,
                timeout=10.0
            )
            response.raise_for_status()
            res_json = response.json()

            return {
                "success": True,
                "caption": res_json.get("caption", res_json.get("text", "")),
                "raw_response": res_json,
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "caption": None,
                "error": f"Capgen API Request Failed: {e}",
            }


if __name__ == "__main__":
    print("=" * 60)
    print("ทดสอบโมดูล Capgen (Image Captioning)")
    print("=" * 60)

    client = CapgenClient()

    # 1. ทดสอบการปฏิเสธการเรียกใช้งานเมื่อยังไม่ระบุ URL
    result_unverified = client.generate_caption(None)
    print("\n[การทดสอบที่ 1: เรียกใช้โดยไม่มีการยืนยัน URL]")
    print(f"ผลลัพธ์: {result_unverified['error']}")

    # 2. ทดสอบความปลอดภัย: ป้องกันการใส่ URL ตรวจจับหน้ากาก
    print("\n[การทดสอบที่ 2: พยายามใส่ URL detectfacemask]")
    try:
        client.verify_and_set_endpoint("https://api.aiforthai.in.th/detectfacemask")
    except ValueError as e:
        print(f"สกัดกั้นสำเร็จ (ตรวจพบ URL ต้องห้าม): {e}")

    # 3. ทดสอบโหมดจำลอง (Mock Mode)
    result_mock = client.generate_caption(None, mock_mode=True)
    print("\n[การทดสอบที่ 3: ทดสอบ Mock Mode]")
    print(f"ผลลัพธ์: {result_mock['caption']}")
    print("=" * 60)
