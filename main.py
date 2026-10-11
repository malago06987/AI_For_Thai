"""
Main Application: Face Mask Detection with AI for Thai
ตรวจจับการใส่หน้ากากอนามัยผ่าน Webcam แบบเรียลไทม์ และแจ้งเตือนด้วยเสียง VAJA
"""

import logging
import sys
import threading
import time
from typing import Optional, Dict, Any
import cv2
import numpy as np

# ตั้งค่า stdout/stderr ให้รองรับ UTF-8 บน Windows Console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import Config
from mask_detector import MaskDetectionClient
from tts_alert import VajaAlertService
from ui_overlay import draw_detection_hud

# กำหนดรูปแบบการแสดงผล Log
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FaceMaskApp")


class FaceMaskDetectionApp:
    """แอปพลิเคชันตรวจจับหน้ากากอนามัยผ่าน Webcam แบบเรียลไทม์"""

    def __init__(self):
        # ตรวจสอบความถูกต้องของการตั้งค่า
        self._check_environment()

        # สร้าง Service ต่างๆ
        self.detector = MaskDetectionClient()
        self.alert_service = VajaAlertService()

        # สถานะการทำงานของระบบ (Shared State ระหว่าง Thread)
        self.current_status = "Waiting"       # "Mask", "NoMask", "NoFace", "Error", "Waiting"
        self.error_message: Optional[str] = None
        self.detected_objects = []
        self.last_check_timestamp: Optional[float] = None
        self.is_api_busy = False

        self._lock = threading.Lock()
        self.is_running = False

    def _check_environment(self):
        """ตรวจสอบและแจ้งเตือนการตั้งค่า API Key"""
        if not Config.validate_api_key():
            logger.warning("=" * 60)
            logger.warning("⚠️  คำเตือน: ยังไม่ได้ตั้งค่า AIFORTHAI_API_KEY ในไฟล์ .env")
            logger.warning("กรุณานำ API Key จาก https://aiforthai.in.th มาใส่ในไฟล์ .env")
            logger.warning("=" * 60)

    def _async_detect_worker(self, frame_snapshot: np.ndarray):
        """Worker ทำงานเบื้องหลังเพื่อส่งภาพไปยัง API โดยไม่ทำให้หน้าจอค้าง"""
        try:
            logger.info("กำลังส่งภาพไปตรวจจับที่ AI for Thai...")
            result = self.detector.detect_from_frame(frame_snapshot)

            with self._lock:
                self.last_check_timestamp = time.time()
                if result.get("success"):
                    self.current_status = result.get("status", "NoFace")
                    self.error_message = None
                    self.detected_objects = result.get("objects", [])

                    logger.info(f"ผลลัพธ์การตรวจจับ: {self.current_status}")

                    # ถ้าผลลัพธ์เป็น Mask ให้แจ้งเตือนด้วยเสียง VAJA
                    if result.get("should_alert"):
                        logger.warning("ตรวจพบหน้ากากอนามัย! สั่งเล่นเสียงแจ้งเตือน: 'ถอดหน้ากากออกด้วยดิวะ'")
                        self.alert_service.trigger_alert_async()
                    else:
                        logger.info("ผลลัพธ์เป็น NoMask หรือไม่พบหน้ากาก -> ไม่ต้องแจ้งเตือน")
                else:
                    self.current_status = "Error"
                    self.error_message = result.get("error_message", "เกิดข้อผิดพลาดในการเรียก API")
                    logger.error(f"ตรวจจับล้มเหลว: {self.error_message}")

        except Exception as e:
            logger.exception(f"เกิดข้อผิดพลาดไม่คาดคิดใน worker: {e}")
            with self._lock:
                self.current_status = "Error"
                self.error_message = f"Worker Exception: {e}"
        finally:
            with self._lock:
                self.is_api_busy = False

    def trigger_check_now(self, frame: np.ndarray) -> bool:
        """สั่งส่งภาพไปตรวจจับทันทีหาก worker ไม่ติดงานเดิมอยู่"""
        with self._lock:
            if self.is_api_busy:
                return False
            self.is_api_busy = True

        # Copy snapshot เพื่อป้องกัน Race condition ระหว่างที่กล้องกำลังอัปเดตเฟรมใหม่
        snapshot = frame.copy()
        worker_thread = threading.Thread(
            target=self._async_detect_worker, args=(snapshot,), daemon=True
        )
        worker_thread.start()
        return True

    def run(self):
        """เปิด Webcam และเริ่มการทำงานแบบเรียลไทม์"""
        logger.info(f"กำลังเปิดกล้อง Webcam หมายเลข {Config.CAMERA_INDEX}...")
        cap = cv2.VideoCapture(Config.CAMERA_INDEX)

        # ปรับความละเอียดกล้องตามการตั้งค่า
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

        if not cap.isOpened():
            logger.error(
                f"❌ ไม่สามารถเปิด Webcam หมายเลข {Config.CAMERA_INDEX} ได้! "
                "กรุณาตรวจสอบว่ามีกล้องเชื่อมต่ออยู่ หรือปิดโปรแกรมอื่นที่กำลังใช้กล้อง"
            )
            return

        window_name = "AI for Thai - Realtime Face Mask Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 600)

        self.is_running = True
        prev_time = time.time()
        fps = 0.0
        last_interval_check = 0.0

        logger.info("=" * 60)
        logger.info("ระบบเริ่มทำงานเรียบร้อยแล้ว!")
        logger.info(f"- ตรวจจับทุกๆ: {Config.CHECK_INTERVAL_SECONDS} วินาที")
        logger.info(f"- Cooldown เสียงเตือน: {Config.ALERT_COOLDOWN_SECONDS} วินาที")
        logger.info("ปุ่มควบคุม: [Q] ออกจากโปรแกรม | [S] ตรวจทันที | [T] ทดสอบเสียง")
        logger.info("=" * 60)

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning("ไม่สามารถอ่านภาพจากกล้องได้ กำลังลองใหม่...")
                    time.sleep(0.05)
                    continue

                current_time = time.time()

                # คำนวณ Frame Rate (FPS)
                delta_time = current_time - prev_time
                if delta_time > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / delta_time)
                prev_time = current_time

                # เรียก API เป็นช่วงเวลา (Throttled) ไม่ส่งทุกเฟรมเพื่อประหยัด Network และโควตา
                if (current_time - last_interval_check) >= Config.CHECK_INTERVAL_SECONDS:
                    if self.trigger_check_now(frame):
                        last_interval_check = current_time

                # คำนวณเวลาที่ผ่านไปจากการตรวจล่าสุด
                with self._lock:
                    status = self.current_status
                    err_msg = self.error_message
                    is_busy = self.is_api_busy
                    last_check = self.last_check_timestamp
                    objs = list(self.detected_objects)

                last_check_ago = (current_time - last_check) if last_check else None

                # วาด HUD และผลลัพธ์บนเฟรมกล้อง
                rendered_frame = draw_detection_hud(
                    frame=frame,
                    status=status,
                    error_message=err_msg,
                    fps=fps,
                    is_checking=is_busy,
                    last_check_ago=last_check_ago,
                    is_audio_playing=self.alert_service.is_playing,
                    objects=objs,
                )

                # แสดงภาพบนหน้าจอ
                cv2.imshow(window_name, rendered_frame)

                # ดักจับการกดปุ่ม (หน่วง 1 ms เพื่อให้หน้าต่างอัปเดต)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 27 คือปุ่ม ESC
                    logger.info("ผู้ใช้กดปุ่มออกจากโปรแกรม...")
                    break
                elif key in (ord("s"), ord("S")):
                    logger.info("ผู้ใช้สั่งตรวจจับทันที (Manual Trigger)...")
                    if self.trigger_check_now(frame):
                        last_interval_check = current_time
                elif key in (ord("t"), ord("T")):
                    logger.info("ผู้ใช้กดทดสอบเล่นเสียงเตือน...")
                    self.alert_service.trigger_alert_async()

        except KeyboardInterrupt:
            logger.info("หยุดการทำงานด้วย KeyboardInterrupt...")
        finally:
            self.is_running = False
            cap.release()
            cv2.destroyAllWindows()
            logger.info("ปิดการทำงานของ Webcam และหน้าต่างโปรแกรมเรียบร้อยแล้ว")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Realtime Face Mask Detection with AI for Thai")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="จำนวนวินาทีที่ต้องการให้โปรแกรมทำงานก่อนปิดอัตโนมัติ (สำหรับทดสอบ)",
    )
    args = parser.parse_args()

    app = FaceMaskDetectionApp()

    # หากมีการกำหนดระยะเวลาสำหรับทดสอบ
    if args.duration:
        def auto_stop():
            time.sleep(args.duration)
            logger.info(f"ครบกำหนดเวลาทดสอบ {args.duration} วินาทีแล้ว กำลังปิดโปรแกรม...")
            app.is_running = False

        timer_thread = threading.Thread(target=auto_stop, daemon=True)
        timer_thread.start()

    app.run()

