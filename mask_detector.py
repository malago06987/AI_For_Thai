"""
Module: mask_detector.py
โมดูลสำหรับส่งภาพใบหน้าไปยัง AI for Thai Face Mask Detection API
Endpoint: https://api.aiforthai.in.th/detectfacemask
พร้อมระบบตรวจจับและครอบภาพใบหน้า (Face Cropping) เพื่อความแม่นยำสูงสุด
"""

import io
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple, List
import cv2
import numpy as np
import requests

from config import Config

logger = logging.getLogger(__name__)


class MaskDetectionClient:
    """ไคลเอนต์สำหรับตรวจจับการสวมใส่หน้ากากอนามัยผ่าน AI for Thai API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: float = 8.0,
    ):
        self.api_key = api_key or Config.API_KEY
        self.api_url = api_url or Config.MASK_DETECTION_URL
        self.timeout = timeout

        # โหลดโมเดลค้นหาใบหน้าของ OpenCV สำหรับ Crop ใบหน้าก่อนส่ง API
        self.face_cascade = None
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            logger.warning(f"ไม่สามารถโหลด Haar Cascade ได้: {e}")

    def detect_face_bbox(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        ค้นหาตำแหน่งใบหน้าของบุคคลในภาพ โดยเลือกใบหน้าที่ใหญ่ที่สุด (ผู้ใช้อยู่หน้ากล้อง)
        """
        if self.face_cascade is None or frame is None or frame.size == 0:
            return None

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80)
            )
            if len(faces) == 0:
                return None

            # เลือกใบหน้าที่มีพื้นที่ใหญ่ที่สุด (บุคคลหลักหน้ากล้อง)
            largest_face = max(faces, key=lambda b: b[2] * b[3])
            return tuple(largest_face)
        except Exception as e:
            logger.debug(f"Error detecting face bbox: {e}")
            return None

    def crop_face_region(
        self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """
        ครอบตัดเฉพาะบริเวณใบหน้าพร้อมระยะเผื่อ (Padding 30%) เพื่อส่งให้ AI for Thai วิเคราะห์
        (AI for Thai เป็น Face Classifier หากส่งภาพทั้งห้องติดโปสเตอร์/ผนังจะเกิด False Positive ได้)
        """
        if frame is None or frame.size == 0:
            return frame, None

        h, w = frame.shape[:2]

        if bbox is None:
            bbox = self.detect_face_bbox(frame)

        if bbox is not None:
            x, y, fw, fh = bbox
            pad = int(fw * 0.30)
            y1 = max(0, y - pad)
            y2 = min(h, y + fh + pad)
            x1 = max(0, x - pad)
            x2 = min(w, x + fw + pad)
            face_img = frame[y1:y2, x1:x2]
            return face_img, bbox

        # หากตรวจไม่พบใบหน้าด้วย Haar (เช่น สวมหน้ากากปิดมิดชิด)
        # ให้ครอบตัดบริเวณกึ่งกลางค่อนล่างของเฟรม (จุดที่บุคคลมักนั่งหน้ากล้อง)
        cy1 = int(h * 0.15)
        cy2 = int(h * 0.95)
        cx1 = int(w * 0.20)
        cx2 = int(w * 0.80)
        center_crop = frame[cy1:cy2, cx1:cx2]
        return center_crop, None

    def detect_from_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        ตรวจจับหน้ากากอนามัยจาก OpenCV Frame (ทำการครอบตัดใบหน้า แล้วส่งไปยัง AI for Thai)
        """
        if frame is None or frame.size == 0:
            return {
                "success": False,
                "status": "Error",
                "should_alert": False,
                "error_message": "ภาพจากกล้องว่างเปล่า (Empty Frame)",
                "raw_response": None,
                "objects": [],
                "bbox": None,
            }

        # 1. ครอบตัดภาพเฉพาะบริเวณใบหน้า
        cropped_face, bbox = self.crop_face_region(frame)

        # บันทึกภาพที่ส่งล่าสุดไว้ใน cache เพื่อให้หน้าเว็บนำไปแสดงผลได้
        try:
            cv2.imwrite("cache/last_sent_face.jpg", cropped_face)
        except Exception:
            pass

        # 2. แปลงเป็น JPEG ส่งเข้า API
        success, encoded_img = cv2.imencode(".jpg", cropped_face, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            return {
                "success": False,
                "status": "Error",
                "should_alert": False,
                "error_message": "ไม่สามารถแปลงภาพเป็น JPEG ได้",
                "raw_response": None,
                "objects": [],
                "bbox": bbox,
            }

        img_bytes = encoded_img.tobytes()
        res = self._send_request(img_bytes)
        res["bbox"] = bbox
        return res

    def detect_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        ตรวจจับหน้ากากอนามัยจากไฟล์รูปภาพ
        """
        try:
            img = cv2.imread(file_path)
            if img is not None:
                return self.detect_from_frame(img)

            with open(file_path, "rb") as f:
                img_bytes = f.read()
            return self._send_request(img_bytes)
        except Exception as e:
            return {
                "success": False,
                "status": "Error",
                "should_alert": False,
                "error_message": f"อ่านไฟล์ภาพล้มเหลว: {e}",
                "raw_response": None,
                "objects": [],
                "bbox": None,
            }

    def _send_request(self, img_bytes: bytes) -> Dict[str, Any]:
        """ส่ง HTTP POST Request แบบ multipart ไปยัง AI for Thai"""
        if not self.api_key or self.api_key in ("your_actual_api_key_here", "your_aiforthai_api_key_here"):
            return {
                "success": False,
                "status": "Error",
                "should_alert": False,
                "error_message": "กรุณาใส่ API Key ในไฟล์ .env ให้เรียบร้อยก่อนใช้งาน",
                "raw_response": None,
                "objects": [],
            }

        headers = {
            "Apikey": self.api_key
        }

        # ส่งแบบ multipart/form-data โดยใช้ฟิลด์ 'file'
        files = {
            "file": ("image.jpg", io.BytesIO(img_bytes), "image/jpeg")
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                files=files,
                timeout=self.timeout
            )

            if response.status_code == 401 or response.status_code == 403:
                return {
                    "success": False,
                    "status": "Error",
                    "should_alert": False,
                    "error_message": "API Key ไม่ถูกต้องหรือโควตาการใช้งานหมด (401/403)",
                    "raw_response": response.text,
                    "objects": [],
                }

            response.raise_for_status()

            try:
                res_data = response.json()
            except ValueError:
                res_data = response.text

            return self._parse_result(res_data)

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "status": "Error",
                "should_alert": False,
                "error_message": "การเชื่อมต่อ API หมดเวลา (Timeout)",
                "raw_response": None,
                "objects": [],
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "status": "Error",
                "should_alert": False,
                "error_message": "ไม่สามารถติดต่อเซิร์ฟเวอร์ได้ (ตรวจเครือข่าย/อินเทอร์เน็ต)",
                "raw_response": None,
                "objects": [],
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "status": "Error",
                "should_alert": False,
                "error_message": f"API Error: {e}",
                "raw_response": getattr(response, "text", None) if "response" in locals() else None,
                "objects": [],
            }

    def _parse_result(self, res_data: Union[Dict, list, str]) -> Dict[str, Any]:
        """
        วิเคราะห์ผลลัพธ์จาก AI for Thai:
        API ตอบกลับในรูปแบบ:
        {"OK": true, "objects": [{"result": "Mask", "score": "[1. 0.]"}]}
        หรือ
        {"OK": true, "objects": [{"result": "NoMask", "score": "[0. 1.]"}]}
        """
        objects = []
        has_mask = False
        has_no_mask = False
        score_info = None

        if isinstance(res_data, dict):
            if "objects" in res_data and isinstance(res_data["objects"], list):
                objects = res_data["objects"]
            elif "result" in res_data:
                res_val = res_data["result"]
                if isinstance(res_val, list):
                    objects = res_val
                elif isinstance(res_val, str):
                    val_str = res_val.strip().lower()
                    if "nomask" in val_str or "no_mask" in val_str or "no-mask" in val_str:
                        has_no_mask = True
                    elif "mask" in val_str:
                        has_mask = True

        elif isinstance(res_data, list):
            objects = res_data
        elif isinstance(res_data, str):
            val_str = res_data.strip().lower()
            if "nomask" in val_str or "no_mask" in val_str:
                has_no_mask = True
            elif "mask" in val_str:
                has_mask = True

        # วิเคราะห์ผลลัพธ์จาก objects
        if objects:
            for obj in objects:
                label = ""
                if isinstance(obj, dict):
                    # AI for Thai ใช้คีย์ "result"
                    label = str(
                        obj.get("result") or obj.get("class") or obj.get("label") or obj.get("name") or ""
                    ).strip().lower()

                    score_raw = obj.get("score")
                    if score_raw:
                        score_info = str(score_raw)

                elif isinstance(obj, str):
                    label = obj.strip().lower()

                # ตัดสินสถานะ Mask vs NoMask
                if "nomask" in label or "no_mask" in label or "no-mask" in label:
                    has_no_mask = True
                elif "mask" in label:
                    has_mask = True

        # กำหนดสถานะสุดท้าย
        if has_mask:
            final_status = "Mask"
            should_alert = True
        elif has_no_mask:
            final_status = "NoMask"
            should_alert = False
        elif objects:
            final_status = "NoMask"
            should_alert = False
        else:
            final_status = "NoFace"
            should_alert = False

        return {
            "success": True,
            "status": final_status,
            "should_alert": should_alert,
            "error_message": None,
            "raw_response": res_data,
            "objects": objects,
            "score": score_info,
        }
