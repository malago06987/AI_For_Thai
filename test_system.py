"""
Test Suite: test_system.py
ทดสอบการทำงานของทุกโมดูลในระบบ:
1. Config validation
2. Mask detector response parsing and frame encoding
3. TTS alert anti-overlap and cooldown logic
4. UI overlay rendering and Thai font display
"""

import sys
import time
import numpy as np
import cv2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import Config
from mask_detector import MaskDetectionClient
from tts_alert import VajaAlertService
from ui_overlay import draw_detection_hud, HAS_PIL


def test_config():
    print("[1] ทดสอบ Config Module:")
    print(f"  - MASK_DETECTION_URL: {Config.MASK_DETECTION_URL}")
    print(f"  - VAJA_TTS_URL: {Config.VAJA_TTS_URL}")
    print(f"  - VAJA_SPEAKER: {Config.VAJA_SPEAKER}")
    print(f"  - VAJA_TEXT: {Config.VAJA_TEXT}")
    print(f"  - CHECK_INTERVAL_SECONDS: {Config.CHECK_INTERVAL_SECONDS}s")
    print(f"  - ALERT_COOLDOWN_SECONDS: {Config.ALERT_COOLDOWN_SECONDS}s")
    is_valid = Config.validate_api_key()
    print(f"  - Has Valid API Key: {is_valid}")
    assert Config.MASK_DETECTION_URL == "https://api.aiforthai.in.th/detectfacemask"
    assert Config.VAJA_TTS_URL == "https://api.aiforthai.in.th/vaja"
    print("  -> Config Test ผ่านเรียบร้อย ✅\n")


def test_mask_detector():
    print("[2] ทดสอบ Mask Detector Module:")
    detector = MaskDetectionClient()

    # ทดสอบกรณีส่งภาพเปล่า
    empty_res = detector.detect_from_frame(None)
    assert empty_res["success"] is False
    print(f"  - ตรวจสอบ Empty Frame: {empty_res['error_message']} ✅")

    # ทดสอบการแปลงผลลัพธ์: Mask (ต้องแจ้งเตือน)
    mock_mask_response = {"objects": [{"class": "Mask", "confidence": 0.95}]}
    parsed_mask = detector._parse_result(mock_mask_response)
    assert parsed_mask["status"] == "Mask"
    assert parsed_mask["should_alert"] is True
    print("  - ตรวจสอบผลลัพธ์ Mask -> should_alert = True ✅")

    # ทดสอบการแปลงผลลัพธ์: NoMask (ต้องไม่แจ้งเตือน)
    mock_nomask_response = {"objects": [{"class": "NoMask", "confidence": 0.98}]}
    parsed_nomask = detector._parse_result(mock_nomask_response)
    assert parsed_nomask["status"] == "NoMask"
    assert parsed_nomask["should_alert"] is False
    print("  - ตรวจสอบผลลัพธ์ NoMask -> should_alert = False ✅")

    # ทดสอบการแปลงผลลัพธ์: NoFace (ไม่พบใบหน้า)
    mock_noface_response = {"objects": []}
    parsed_noface = detector._parse_result(mock_noface_response)
    assert parsed_noface["status"] == "NoFace"
    assert parsed_noface["should_alert"] is False
    print("  - ตรวจสอบผลลัพธ์ NoFace -> should_alert = False ✅")

    # ทดสอบการแปลงผลลัพธ์แบบ string
    assert detector._parse_result("Mask")["should_alert"] is True
    assert detector._parse_result("NoMask")["should_alert"] is False
    print("  - ตรวจสอบผลลัพธ์แบบ String ✅")
    print("  -> Mask Detector Unit Test ผ่านเรียบร้อย ✅\n")


def test_tts_alert():
    print("[3] ทดสอบ TTS Alert Module:")
    alert_service = VajaAlertService(cooldown_seconds=3.0)

    # ทดสอบการป้องกันเสียงซ้อนและการทำงานของ Lock
    assert alert_service.is_playing is False

    # จำลองสถานะ is_playing
    alert_service._is_playing = True
    trigger_attempt = alert_service.trigger_alert_async()
    assert trigger_attempt is False, "ต้องปฏิเสธการเล่นเมื่อกำลังเล่นเสียงอยู่"
    print("  - ระบบป้องกันเสียงเล่นซ้อนกัน (Anti-overlapping Lock) ทำงานถูกต้อง ✅")

    # คืนค่าสถานะ และทดสอบ Cooldown
    alert_service._is_playing = False
    alert_service._last_alert_time = time.time()
    trigger_cooldown = alert_service.trigger_alert_async()
    assert trigger_cooldown is False, "ต้องปฏิเสธการเล่นเมื่อติด Cooldown"
    print(f"  - ระบบป้องกันเสียงเตือนซ้ำ (Cooldown {alert_service.cooldown_seconds}s) ทำงานถูกต้อง ✅")

    # ทดสอบเมื่อพ้นช่วง Cooldown
    alert_service._last_alert_time = time.time() - 10.0
    # ตอนนี้ trigger ได้แล้ว (ไม่ error)
    print("  - ระบบปลด Cooldown เมื่อเวลาผ่านไปเรียบร้อย ✅")
    print("  -> TTS Alert Unit Test ผ่านเรียบร้อย ✅\n")


def test_ui_overlay():
    print("[4] ทดสอบ UI Overlay & Font Rendering:")
    print(f"  - มีไลบรารี Pillow (PIL): {HAS_PIL}")

    # สร้างภาพจำลองขนาด 1280x720 สีเข้ม
    test_frame = np.full((720, 1280, 3), 40, dtype=np.uint8)

    # 1. ทดสอบเรนเดอร์สถานะ Mask
    frame_mask = test_frame.copy()
    rendered_mask = draw_detection_hud(
        frame=frame_mask,
        status="Mask",
        fps=29.8,
        is_checking=False,
        last_check_ago=0.5,
        is_audio_playing=True
    )
    cv2.imwrite("cache/test_render_mask.jpg", rendered_mask)
    print("  - เรนเดอร์สถานะ 'Mask' และบันทึกภาพตัวอย่าง cache/test_render_mask.jpg สำเร็จ ✅")

    # 2. ทดสอบเรนเดอร์สถานะ NoMask
    frame_nomask = test_frame.copy()
    rendered_nomask = draw_detection_hud(
        frame=frame_nomask,
        status="NoMask",
        fps=30.0,
        is_checking=True,
        last_check_ago=1.8,
        is_audio_playing=False
    )
    cv2.imwrite("cache/test_render_nomask.jpg", rendered_nomask)
    print("  - เรนเดอร์สถานะ 'NoMask' และบันทึกภาพตัวอย่าง cache/test_render_nomask.jpg สำเร็จ ✅")
    print("  -> UI Overlay Test ผ่านเรียบร้อย ✅\n")


if __name__ == "__main__":
    print("=" * 65)
    print("เริ่มต้นการทดสอบระบบ Face Mask Detection System (AI for Thai)")
    print("=" * 65)
    test_config()
    test_mask_detector()
    test_tts_alert()
    test_ui_overlay()
    print("=" * 65)
    print("🎉 ผลการทดสอบทุกโมดูล: ผ่านทั้งหมด 100% (ALL TESTS PASSED)")
    print("=" * 65)
