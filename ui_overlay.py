"""
Module: ui_overlay.py
โมดูลสำหรับวาด UI / HUD สวยงามบน OpenCV Frame
รองรับการแสดงผลฟอนต์ภาษาไทยคมชัดผ่าน Pillow และระบบแจ้งเตือนแบบเรียลไทม์
"""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np

# ลองโหลด Pillow สำหรับวาดตัวอักษรภาษาไทย
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ค้นหาฟอนต์ภาษาไทยมาตรฐานในระบบ Windows
WINDOWS_FONTS = [
    Path(r"C:\Windows\Fonts\tahoma.ttf"),
    Path(r"C:\Windows\Fonts\leelawad.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
]

_FONT_CACHE: Dict[int, Any] = {}


def _get_font(size: int = 24):
    """โหลดฟอนต์ภาษาไทยที่เหมาะสมและเก็บในแคช"""
    if not HAS_PIL:
        return None

    if size in _FONT_CACHE:
        return _FONT_CACHE[size]

    for font_path in WINDOWS_FONTS:
        if font_path.exists():
            try:
                font = ImageFont.truetype(str(font_path), size)
                _FONT_CACHE[size] = font
                return font
            except Exception:
                continue

    # หากไม่พบฟอนต์ในรายการ ให้ใช้ฟอนต์เริ่มต้น
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def draw_text_thai(
    img: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_size: int = 22,
    color_bgr: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """วาดข้อความภาษาไทยลงบน OpenCV image (BGR)"""
    if not HAS_PIL:
        # Fallback เป็น cv2.putText ภาษาอังกฤษ/ตัวเลข
        cv2.putText(
            img,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_bgr,
            2,
            cv2.LINE_AA,
        )
        return img

    font = _get_font(font_size)
    if font is None:
        cv2.putText(
            img,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_bgr,
            2,
            cv2.LINE_AA,
        )
        return img

    # แปลง BGR เป็น RGB สำหรับ PIL
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    draw = ImageDraw.Draw(pil_img)

    # PIL ใช้สีแบบ RGB
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

    # วาดเงาเล็กน้อยให้อ่านง่ายขึ้น
    x, y = position
    draw.text((x + 1, y + 1), text, font=font, fill=(20, 20, 20))
    draw.text((x, y), text, font=font, fill=color_rgb)

    # แปลงกลับเป็น BGR สำหรับ OpenCV
    result_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np.copyto(img, result_bgr)
    return img


def draw_detection_hud(
    frame: np.ndarray,
    status: str,
    error_message: Optional[str] = None,
    fps: float = 0.0,
    is_checking: bool = False,
    last_check_ago: Optional[float] = None,
    is_audio_playing: bool = False,
    objects: Optional[List[Dict[str, Any]]] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    next_check_in: Optional[float] = None,
) -> np.ndarray:
    """
    วาด HUD แสดงผลสถานะการตรวจจับและการแจ้งเตือนบนเฟรมกล้อง
    """
    h, w = frame.shape[:2]

    # กำหนดชุดสีตามสถานะ (BGR)
    if status == "Mask":
        theme_color = (0, 0, 230)      # สีแดง (เตือนใส่หน้ากาก)
        status_text = "เตือน: ตรวจพบหน้ากากอนามัย (ถอดหน้ากากออกด้วยดิวะ)"
        sub_text = "STATUS: MASK DETECTED"
        cv2.rectangle(frame, (0, 0), (w, h), theme_color, 8)
    elif status == "NoMask":
        theme_color = (0, 180, 0)      # สีเขียว (ปกติ)
        status_text = "ปกติ: ไม่พบหน้ากากอนามัย"
        sub_text = "STATUS: NO MASK (NORMAL)"
    elif status == "NoFace":
        theme_color = (120, 120, 120)  # สีเทา (ไม่พบใบหน้า)
        status_text = "รอตรวจจับ: กำลังค้นหาใบหน้า..."
        sub_text = "STATUS: SEARCHING FOR FACE"
    elif status == "Error":
        theme_color = (0, 140, 255)    # สีส้มแดง (ข้อผิดพลาด)
        status_text = f"ข้อผิดพลาด: {error_message or 'ไม่สามารถเชื่อมต่อ API ได้'}"
        sub_text = "STATUS: API/NETWORK ERROR"
    else:
        theme_color = (180, 120, 40)   # สีฟ้าอมเทา (เริ่มต้น)
        status_text = "กำลังเริ่มต้นระบบ..."
        sub_text = "STATUS: INITIALIZING"

    # 1. วาดกรอบครอบใบหน้าถ้ามี bbox
    if bbox is not None:
        bx, by, bw, bh = bbox
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), theme_color, 3)
        # ป้ายชื่อบนกรอบหน้า
        tag_text = "MASK" if status == "Mask" else "NO MASK" if status == "NoMask" else "FACE"
        cv2.rectangle(frame, (bx, max(0, by - 26)), (bx + 110, by), theme_color, -1)
        cv2.putText(frame, tag_text, (bx + 6, max(18, by - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 2. แถบ Header ด้านบน (Semi-transparent overlay)
    header_h = 75
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (25, 25, 30), -1)
    cv2.rectangle(overlay, (0, header_h - 4), (w, header_h), theme_color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # แสดงข้อความสถานะหลัก
    draw_text_thai(frame, status_text, (20, 12), font_size=24, color_bgr=theme_color)
    cv2.putText(
        frame,
        sub_text,
        (22, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # ไฟแสดงสถานะการเช็ค API มุมขวาบน
    if is_checking:
        cv2.circle(frame, (w - 30, 25), 10, (0, 255, 255), -1)
        draw_text_thai(frame, "กำลังส่งภาพวิเคราะห์...", (w - 230, 15), font_size=18, color_bgr=(0, 255, 255))
    else:
        cv2.circle(frame, (w - 30, 25), 8, (0, 200, 0), -1)
        draw_text_thai(frame, "ตรวจทุก 5 วินาที", (w - 180, 15), font_size=18, color_bgr=(180, 180, 180))

    # แสดงสถานะเสียงเตือน
    if is_audio_playing:
        draw_text_thai(frame, "🔊 กำลังเล่นเสียงเตือน...", (w - 240, 45), font_size=16, color_bgr=(0, 255, 255))

    # 3. แถบ Footer ด้านล่างแสดงข้อมูลประสิทธิภาพและคีย์ลัด
    footer_h = 45
    footer_y = h - footer_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, footer_y), (w, h), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # ข้อมูลสถิติ
    info_parts = [f"FPS: {fps:.1f}"]
    if next_check_in is not None:
        info_parts.append(f"แคปตรวจครั้งถัดไป: {max(0.0, next_check_in):.1f}s")
    elif last_check_ago is not None:
        info_parts.append(f"ตรวจล่าสุด: {last_check_ago:.1f}s ที่แล้ว")

    info_str = " | ".join(info_parts)
    draw_text_thai(frame, info_str, (20, footer_y + 12), font_size=18, color_bgr=(220, 220, 220))

    # คำแนะนำปุ่มกด
    help_text = "[Q] ปิด | [S] แคปตรวจทันที | [T] ทดสอบเสียง"
    draw_text_thai(frame, help_text, (w - 430, footer_y + 12), font_size=18, color_bgr=(180, 180, 180))

    return frame
