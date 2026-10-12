"""
Web Application: Realtime Face Mask Detection with AI for Thai & Thai LLM
เปิดหน้าเว็บแสดงผลผ่านเบราว์เซอร์ (http://localhost:5000)
- ตรวจจับหน้ากากอนามัยผ่าน AI for Thai (detectfacemask API)
- เจนประโยคแจ้งเตือนอัตโนมัติผ่าน Thai LLM (หลากหลายบุคลิก: สุภาพ, กวนๆ, ครูฝ่ายปกครอง, หุ่นยนต์ AI)
- สังเคราะห์เสียงพูดเตือนด้วย AI for Thai (VAJA TTS API)
"""

import logging
import os
import sys
import threading
import time
import webbrowser
from typing import Optional, Dict, Any, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, send_file, request

from config import Config
from mask_detector import MaskDetectionClient
from tts_alert import VajaAlertService
from thai_llm import thai_llm, ThaiLLMPersona, LOCAL_PERSONA_CORPUS
from ui_overlay import draw_detection_hud

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FaceMaskWebApp")

app = Flask(__name__)

# Singletons
detector = MaskDetectionClient()
alert_service = VajaAlertService()

# Global State
state = {
    "status": "Waiting",
    "error_message": None,
    "last_check_timestamp": None,
    "next_check_in": 5.0,
    "is_api_busy": False,
    "fps": 0.0,
    "is_audio_playing": False,
    "total_checks": 0,
    "mask_count": 0,
    "latest_bbox": None,
    "latest_score": None,
}
state_lock = threading.Lock()

# Thread-safe Camera Frame Buffer
latest_raw_frame = None
latest_annotated_frame = None
frame_lock = threading.Lock()
is_running = True
last_check_time = 0.0


def camera_loop():
    """เธรดอ่านกล้องหลักเพียงตัวเดียว เพื่อป้องกัน Frame Corruption หรือ Race Condition"""
    global latest_raw_frame, latest_annotated_frame, is_running, state, last_check_time

    logger.info(f"เริ่มการทำงานของเธรดกล้อง Webcam #{Config.CAMERA_INDEX}...")
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

    # วอร์มกล้องเล็กน้อยเพื่อปรับแสง
    for _ in range(10):
        cap.read()
        time.sleep(0.03)

    prev_time = time.time()
    last_check_time = time.time()
    check_interval = 5.0  # 5.0 วินาที

    while is_running:
        success, frame = cap.read()
        if not success or frame is None:
            time.sleep(0.04)
            continue

        current_time = time.time()

        # คำนวณ FPS
        delta = current_time - prev_time
        fps_val = 0.0
        if delta > 0:
            fps_val = 1.0 / delta
        prev_time = current_time

        # คำนวณเวลานับถอยหลังตรวจครั้งถัดไป
        elapsed_since_check = current_time - last_check_time
        remaining = max(0.0, check_interval - elapsed_since_check)

        with state_lock:
            state["fps"] = 0.9 * state["fps"] + 0.1 * fps_val
            state["next_check_in"] = remaining
            state["is_audio_playing"] = alert_service.is_playing
            curr_status = state["status"]
            err_msg = state["error_message"]
            is_busy = state["is_api_busy"]
            last_ts = state["last_check_timestamp"]
            cur_bbox = state["latest_bbox"]

        # ตรวจสอบว่าถึงเวลาส่งภาพตรวจหรือยัง (ทุก 5 วินาที)
        if elapsed_since_check >= check_interval:
            if not is_busy:
                last_check_time = current_time
                trigger_api_check(frame)

        last_ago = (current_time - last_ts) if last_ts else None

        # วาด HUD บนเฟรมกล้อง พร้อมแสดงข้อความเตือนจาก Thai LLM
        display_frame = frame.copy()
        annotated = draw_detection_hud(
            frame=display_frame,
            status=curr_status,
            error_message=err_msg,
            fps=state["fps"],
            is_checking=is_busy,
            last_check_ago=last_ago,
            is_audio_playing=alert_service.is_playing,
            bbox=cur_bbox,
            next_check_in=remaining,
            alert_text=alert_service.last_spoken_text,
        )

        with frame_lock:
            latest_raw_frame = frame
            latest_annotated_frame = annotated

        time.sleep(0.02)

    cap.release()
    logger.info("ปิดเธรดกล้อง Webcam เรียบร้อยแล้ว")


def trigger_api_check(frame_to_check: np.ndarray):
    """ส่งภาพไปตรวจจับที่ AI for Thai ในเธรดเบื้องหลัง"""
    with state_lock:
        if state["is_api_busy"]:
            return
        state["is_api_busy"] = True

    snapshot = frame_to_check.copy()

    def worker():
        global state
        try:
            logger.info("📸 กำลังแคปภาพใบหน้า ส่งไปตรวจจับที่ AI for Thai...")
            result = detector.detect_from_frame(snapshot)

            with state_lock:
                state["last_check_timestamp"] = time.time()
                state["total_checks"] += 1
                state["latest_bbox"] = result.get("bbox")
                state["latest_score"] = result.get("score")

                if result.get("success"):
                    state["status"] = result.get("status", "NoFace")
                    state["error_message"] = None

                    logger.info(
                        f"ผลลัพธ์จาก AI for Thai: {state['status']} | คะแนน: {state['latest_score']}"
                    )

                    # ถ้าผลเป็น Mask ให้แจ้งเตือนด้วยเสียง VAJA TTS + Thai LLM
                    if result.get("should_alert"):
                        state["mask_count"] += 1
                        logger.warning("⚠️ ตรวจพบหน้ากากอนามัย! ให้ Thai LLM สร้างประโยคเตือนและสั่ง VAJA TTS พูด")
                        alert_service.trigger_alert_async()
                    else:
                        logger.info("ผลลัพธ์เป็น NoMask (ไม่ใส่หน้ากาก) -> ไม่ต้องแจ้งเตือน")
                else:
                    state["status"] = "Error"
                    state["error_message"] = result.get("error_message")
                    logger.error(f"ตรวจจับล้มเหลว: {state['error_message']}")

        except Exception as e:
            with state_lock:
                state["status"] = "Error"
                state["error_message"] = str(e)
            logger.exception(f"เกิดข้อผิดพลาดในการเรียก API: {e}")
        finally:
            with state_lock:
                state["is_api_busy"] = False
            # ตั้งเวลาเริ่มต้นนับถอยหลัง 5 วินาทีใหม่หลังจากที่การตรวจจับเสร็จสิ้น
            global last_check_time
            last_check_time = time.time()

    t = threading.Thread(target=worker, daemon=True)
    t.start()


def generate_frames():
    """สตรีมภาพ JPEG ไปยัง Web Browser แบบราบรื่น"""
    while is_running:
        with frame_lock:
            frame = latest_annotated_frame

        if frame is None:
            time.sleep(0.04)
            continue

        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(0.03)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI for Thai & Thai LLM - Realtime Face Mask Detection</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Kanit', sans-serif; }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen pb-12">
    <!-- Navbar -->
    <header class="bg-slate-900 border-b border-slate-800 px-6 py-4 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-50 shadow-md">
        <div class="flex items-center space-x-3">
            <span class="inline-block w-3.5 h-3.5 rounded-full bg-emerald-500 animate-pulse"></span>
            <div>
                <h1 class="text-lg md:text-xl font-bold tracking-wide flex items-center gap-2">
                    <span>AI for Thai</span>
                    <span class="text-xs px-2 py-0.5 rounded bg-indigo-600/30 text-indigo-300 border border-indigo-500/40 font-mono">+ Thai LLM</span>
                    <span>ระบบตรวจจับหน้ากากอนามัยอัจฉริยะ</span>
                </h1>
                <p class="text-xs text-slate-400">ตรวจจับด้วย AI for Thai (detectfacemask) • เจนประโยคด้วย Thai LLM • สังเคราะห์เสียงด้วย VAJA TTS</p>
            </div>
        </div>
        <div class="flex items-center space-x-3 text-xs md:text-sm">
            <span class="px-3 py-1 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                รอบการตรวจ: <b class="text-indigo-400">5.0s</b>
            </span>
            <span class="px-3 py-1 rounded-full bg-emerald-950/60 text-emerald-400 border border-emerald-800/60 font-medium">
                ● ระบบพร้อมทำงาน
            </span>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-7xl mx-auto p-4 md:p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        <!-- Left: Live Video Stream & Quick Controls -->
        <section class="lg:col-span-2 space-y-6">
            <div class="bg-slate-900 rounded-2xl border border-slate-800 p-4 shadow-xl flex flex-col items-center">
                <div class="w-full flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
                    <span class="text-base font-semibold text-slate-200 flex items-center gap-2">
                        <span>📹</span> <span>ภาพสดจากกล้อง Webcam (พร้อม Thai HUD)</span>
                    </span>
                    <span id="badge-countdown" class="text-xs px-3 py-1 rounded-full font-medium bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                        นับถอยหลัง: 5.0s
                    </span>
                </div>
                
                <div class="relative w-full aspect-video bg-black rounded-xl overflow-hidden border border-slate-800 shadow-inner flex items-center justify-center">
                    <img src="/video_feed" alt="Webcam Feed" class="w-full h-full object-contain">
                </div>

                <!-- Controls Toolbar -->
                <div class="w-full mt-4 flex flex-wrap items-center justify-between gap-3">
                    <div class="flex flex-wrap gap-2.5">
                        <button onclick="triggerSnapshot()" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 active:scale-95 text-white font-medium rounded-lg text-sm shadow transition flex items-center space-x-2">
                            <span>📸</span>
                            <span>แคปภาพตรวจทันที</span>
                        </button>
                        <button onclick="testAudio()" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 active:scale-95 text-slate-200 font-medium rounded-lg text-sm border border-slate-700 transition flex items-center space-x-2">
                            <span>🔊</span>
                            <span>ทดสอบเสียงพูด (VAJA TTS)</span>
                        </button>
                    </div>
                    
                    <div id="audio-alert-banner" class="hidden text-sm font-semibold text-rose-400 animate-pulse bg-rose-950/40 px-3 py-1.5 rounded-lg border border-rose-800/50 flex items-center gap-2">
                        <span>📢</span>
                        <span>กำลังเปล่งเสียงเตือนผ่านลำโพง...</span>
                    </div>
                </div>
            </div>

            <!-- Thai LLM Dynamic Alert Assistant Box (Feature Highlight) -->
            <div class="bg-gradient-to-br from-slate-900 via-indigo-950/30 to-slate-900 rounded-2xl border-2 border-indigo-500/40 p-6 shadow-2xl space-y-5">
                <div class="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-indigo-500/20">
                    <div class="flex items-center space-x-3">
                        <span class="text-2xl">🧠</span>
                        <div>
                            <h2 class="text-base font-bold text-indigo-300 flex items-center gap-2">
                                <span>Thai LLM อัจฉริยะ (Dynamic Alert Assistant)</span>
                                <span class="text-[10px] uppercase px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 font-mono">LLM Powered</span>
                            </h2>
                            <p class="text-xs text-slate-400">สร้างประโยคเตือนภาษาไทยอัตโนมัติหลากหลายบุคลิก ส่งเข้า VAJA TTS</p>
                        </div>
                    </div>
                    <label class="flex items-center cursor-pointer space-x-2">
                        <input type="checkbox" id="toggle-dynamic-llm" onchange="toggleLLM(this.checked)" class="sr-only peer" checked>
                        <div class="w-11 h-6 bg-slate-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-600"></div>
                        <span class="text-xs font-medium text-slate-300">เปิดใช้ Thai LLM</span>
                    </label>
                </div>

                <!-- Persona Switcher Buttons -->
                <div>
                    <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-2">เลือกบุคลิกของ Thai LLM (Persona):</span>
                    <div class="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                        <button onclick="setPersona('funny')" id="btn-persona-funny" class="persona-btn p-3 rounded-xl border border-slate-700 bg-slate-800/80 hover:bg-slate-700 text-left transition flex flex-col justify-between">
                            <span class="text-xl mb-1">🎭</span>
                            <span class="text-xs font-bold text-slate-200">โหมดกวนๆ</span>
                            <span class="text-[10px] text-slate-400">เป็นกันเอง แซวขำๆ</span>
                        </button>
                        <button onclick="setPersona('polite')" id="btn-persona-polite" class="persona-btn p-3 rounded-xl border border-slate-700 bg-slate-800/80 hover:bg-slate-700 text-left transition flex flex-col justify-between">
                            <span class="text-xl mb-1">🌸</span>
                            <span class="text-xs font-bold text-slate-200">โหมดสุภาพ</span>
                            <span class="text-[10px] text-slate-400">นอบน้อม ไพเราะ</span>
                        </button>
                        <button onclick="setPersona('teacher')" id="btn-persona-teacher" class="persona-btn p-3 rounded-xl border border-slate-700 bg-slate-800/80 hover:bg-slate-700 text-left transition flex flex-col justify-between">
                            <span class="text-xl mb-1">🧑‍🏫</span>
                            <span class="text-xs font-bold text-slate-200">ครูฝ่ายปกครอง</span>
                            <span class="text-[10px] text-slate-400">ดุดัน เข้มงวด</span>
                        </button>
                        <button onclick="setPersona('cyber')" id="btn-persona-cyber" class="persona-btn p-3 rounded-xl border border-slate-700 bg-slate-800/80 hover:bg-slate-700 text-left transition flex flex-col justify-between">
                            <span class="text-xl mb-1">🤖</span>
                            <span class="text-xs font-bold text-slate-200">หุ่นยนต์ AI</span>
                            <span class="text-[10px] text-slate-400">ไฮเทค ล้ำสมัย</span>
                        </button>
                    </div>
                </div>

                <!-- Current Sentence Display Box -->
                <div class="bg-slate-950/80 rounded-xl border border-indigo-500/30 p-4 space-y-2">
                    <div class="flex items-center justify-between text-xs">
                        <span class="text-slate-400 font-medium">💬 ประโยคแจ้งเตือนปัจจุบัน (ส่งพูดผ่าน VAJA):</span>
                        <span id="llm-source-badge" class="px-2 py-0.5 rounded text-[10px] font-mono bg-indigo-900/60 text-indigo-300 border border-indigo-700/50">
                            Thai LLM
                        </span>
                    </div>
                    <div id="llm-spoken-text" class="text-lg font-semibold text-amber-300 italic py-1">
                        "กำลังโหลดประโยค..."
                    </div>
                    <div class="flex items-center justify-end gap-2 pt-2 border-t border-slate-850">
                        <button onclick="generateNewSentence()" class="px-3 py-1.5 bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-200 rounded-lg text-xs font-medium border border-indigo-500/40 transition flex items-center space-x-1.5 active:scale-95">
                            <span>🎲</span>
                            <span>สุ่มประโยคใหม่ทันที</span>
                        </button>
                    </div>
                </div>
            </div>
        </section>

        <!-- Right: Status, Face Preview & Stats -->
        <section class="space-y-6">
            <!-- Detection Status Card -->
            <div id="status-card" class="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl transition-all">
                <div class="flex justify-between items-center mb-2">
                    <h2 class="text-xs uppercase tracking-wider text-slate-400 font-semibold">ผลตรวจการใส่หน้ากาก</h2>
                    <span id="score-badge" class="text-xs text-slate-400 font-mono">-</span>
                </div>
                <div id="status-text" class="text-3xl font-bold text-slate-300">
                    กำลังเริ่มตรวจ...
                </div>
                <p id="status-desc" class="text-sm text-slate-400 mt-2">
                    ระบบจะแคปภาพใบหน้าส่งตรวจทุกๆ 5 วินาที
                </p>
            </div>

            <!-- Last Captured Face Preview -->
            <div class="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl space-y-3">
                <div class="flex justify-between items-center">
                    <h3 class="text-xs uppercase tracking-wider text-slate-400 font-semibold">ภาพใบหน้าที่ส่ง API ล่าสุด</h3>
                    <span class="text-xs text-indigo-400 font-medium">Cropped Face (AI for Thai)</span>
                </div>
                <div class="w-full h-44 bg-slate-950 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center">
                    <img id="last-face-img" src="/last_face.jpg?t=0" alt="Last Captured Face" class="h-full object-contain">
                </div>
                <p class="text-xs text-slate-500 text-center">
                    ระบบครอบตัดเฉพาะบริเวณใบหน้าเพื่อส่ง AI for Thai ป้องกันสิ่งรบกวนในพื้นหลัง
                </p>
            </div>

            <!-- Dashboard Stats -->
            <div class="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl space-y-4">
                <h3 class="text-xs uppercase tracking-wider text-slate-400 font-semibold">ข้อมูลการทำงานและสถิติ</h3>
                
                <div class="grid grid-cols-2 gap-4">
                    <div class="bg-slate-950 p-4 rounded-xl border border-slate-800">
                        <span class="text-xs text-slate-500">เฟรมเรตกล้อง (FPS)</span>
                        <div id="stat-fps" class="text-2xl font-bold text-slate-200">0.0</div>
                    </div>
                    <div class="bg-slate-950 p-4 rounded-xl border border-slate-800">
                        <span class="text-xs text-slate-500">ตรวจพบหน้ากากอนามัย</span>
                        <div id="stat-masks" class="text-2xl font-bold text-rose-400">0 ครั้ง</div>
                    </div>
                </div>

                <div class="pt-3 border-t border-slate-800 space-y-2 text-xs text-slate-400">
                    <div class="flex justify-between">
                        <span>บริการ Vision:</span>
                        <span class="text-slate-300 font-medium">AI for Thai (detectfacemask)</span>
                    </div>
                    <div class="flex justify-between">
                        <span>บริการ Text/LLM:</span>
                        <span class="text-indigo-400 font-medium">Thai LLM (Dynamic Persona)</span>
                    </div>
                    <div class="flex justify-between">
                        <span>บริการ Speech:</span>
                        <span class="text-slate-300 font-medium">AI for Thai (VAJA TTS)</span>
                    </div>
                    <div class="flex justify-between">
                        <span>ตรวจล่าสุด:</span>
                        <span id="stat-last-time" class="text-slate-300">-</span>
                    </div>
                </div>
            </div>
        </section>
    </main>

    <script>
        let currentActivePersona = 'funny';

        function updatePersonaHighlight(persona) {
            currentActivePersona = persona;
            const personas = ['funny', 'polite', 'teacher', 'cyber'];
            personas.forEach(p => {
                const btn = document.getElementById('btn-persona-' + p);
                if (btn) {
                    if (p === persona) {
                        btn.className = 'persona-btn p-3 rounded-xl border-2 border-indigo-400 bg-indigo-950/60 shadow-lg text-left transition flex flex-col justify-between ring-2 ring-indigo-500/30';
                    } else {
                        btn.className = 'persona-btn p-3 rounded-xl border border-slate-700 bg-slate-800/80 hover:bg-slate-700 text-left transition flex flex-col justify-between';
                    }
                }
            });
        }

        async function setPersona(persona) {
            updatePersonaHighlight(persona);
            try {
                const res = await fetch('/api/set_persona', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ persona: persona })
                });
                const data = await res.json();
                if (data.success && data.text) {
                    document.getElementById('llm-spoken-text').innerText = '"' + data.text + '"';
                    document.getElementById('llm-source-badge').innerText = data.source || 'Thai LLM';
                }
            } catch (err) {
                console.error(err);
            }
        }

        async function toggleLLM(enabled) {
            try {
                await fetch('/api/toggle_llm', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ enable: enabled })
                });
            } catch (err) {
                console.error(err);
            }
        }

        async function generateNewSentence() {
            try {
                const btn = event?.currentTarget;
                if (btn) btn.classList.add('animate-spin');
                const res = await fetch('/api/generate_sentence', { method: 'POST' });
                const data = await res.json();
                if (data.success && data.text) {
                    document.getElementById('llm-spoken-text').innerText = '"' + data.text + '"';
                    document.getElementById('llm-source-badge').innerText = data.source || 'Thai LLM';
                }
                if (btn) btn.classList.remove('animate-spin');
            } catch (err) {
                console.error(err);
            }
        }

        async function fetchStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();

                const statusCard = document.getElementById('status-card');
                const statusText = document.getElementById('status-text');
                const statusDesc = document.getElementById('status-desc');
                const badgeCountdown = document.getElementById('badge-countdown');
                const scoreBadge = document.getElementById('score-badge');
                const audioBanner = document.getElementById('audio-alert-banner');
                const spokenText = document.getElementById('llm-spoken-text');
                const sourceBadge = document.getElementById('llm-source-badge');
                const toggleLlm = document.getElementById('toggle-dynamic-llm');

                document.getElementById('stat-fps').innerText = data.fps.toFixed(1);
                document.getElementById('stat-masks').innerText = data.mask_count + ' ครั้ง';
                document.getElementById('stat-last-time').innerText = data.last_check_ago !== null ? data.last_check_ago.toFixed(1) + 's ที่แล้ว' : 'ยังไม่เคย';

                if (data.last_spoken_text) {
                    spokenText.innerText = '"' + data.last_spoken_text + '"';
                }
                if (data.last_sentence_source) {
                    sourceBadge.innerText = data.last_sentence_source;
                }
                if (toggleLlm && toggleLlm.checked !== data.enable_dynamic_llm) {
                    toggleLlm.checked = data.enable_dynamic_llm;
                }
                if (data.current_persona && data.current_persona !== currentActivePersona) {
                    updatePersonaHighlight(data.current_persona);
                }

                if (data.is_api_busy) {
                    badgeCountdown.innerText = "⏳ กำลังส่ง API วิเคราะห์...";
                    badgeCountdown.className = "text-xs px-3 py-1 rounded-full font-medium bg-amber-500/20 text-amber-300 border border-amber-500/30 animate-pulse";
                } else {
                    badgeCountdown.innerText = "แคปตรวจในอีก: " + data.next_check_in.toFixed(1) + "s";
                    badgeCountdown.className = "text-xs px-3 py-1 rounded-full font-medium bg-indigo-500/20 text-indigo-300 border border-indigo-500/30";
                }

                if (data.latest_score) {
                    scoreBadge.innerText = "คะแนน: " + data.latest_score;
                }

                if (data.is_audio_playing) {
                    audioBanner.classList.remove('hidden');
                } else {
                    audioBanner.classList.add('hidden');
                }

                if (data.status === 'Mask') {
                    statusCard.className = "bg-rose-950/40 rounded-2xl border-2 border-rose-500 p-6 shadow-2xl transition-all";
                    statusText.className = "text-3xl font-extrabold text-rose-400";
                    statusText.innerText = "ตรวจพบหน้ากากอนามัย ⚠️";
                    statusDesc.innerText = "ตรวจพบการใส่หน้ากากอนามัย (ส่งเสียงเตือนผ่าน VAJA TTS แล้ว)";
                } else if (data.status === 'NoMask') {
                    statusCard.className = "bg-emerald-950/30 rounded-2xl border-2 border-emerald-500/70 p-6 shadow-xl transition-all";
                    statusText.className = "text-3xl font-bold text-emerald-400";
                    statusText.innerText = "ปกติ: ไม่ใส่หน้ากาก ✅";
                    statusDesc.innerText = "ใบหน้าชัดเจน ไม่มีการสวมใส่หน้ากากอนามัย";
                } else if (data.status === 'NoFace') {
                    statusCard.className = "bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl transition-all";
                    statusText.className = "text-3xl font-bold text-slate-400";
                    statusText.innerText = "กำลังค้นหาใบหน้า 👤";
                    statusDesc.innerText = "กรุณาหันหน้าเข้าหากล้อง Webcam";
                } else if (data.status === 'Error') {
                    statusCard.className = "bg-amber-950/30 rounded-2xl border border-amber-500/50 p-6 shadow-xl transition-all";
                    statusText.className = "text-2xl font-bold text-amber-400";
                    statusText.innerText = "API ขัดข้อง ⚠️";
                    statusDesc.innerText = data.error_message || "ไม่สามารถติดต่อ API ได้";
                }

                // อัปเดตรูปภาพที่ส่งล่าสุด
                const faceImg = document.getElementById('last-face-img');
                if (data.last_check_ago !== null && data.last_check_ago < 2.0) {
                    faceImg.src = '/last_face.jpg?t=' + Date.now();
                }
            } catch (err) {
                console.error(err);
            }
        }

        async function triggerSnapshot() {
            await fetch('/api/check_now', { method: 'POST' });
            fetchStatus();
        }

        async function testAudio() {
            await fetch('/api/test_audio', { method: 'POST' });
            fetchStatus();
        }

        updatePersonaHighlight('funny');
        setInterval(fetchStatus, 600);
        fetchStatus();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/last_face.jpg")
def last_face():
    """ส่งภาพใบหน้าที่ครอบตัดและส่ง API ล่าสุดกลับมาแสดงผล"""
    face_path = "cache/last_sent_face.jpg"
    if os.path.exists(face_path):
        return send_file(face_path, mimetype="image/jpeg")
    # หากยังไม่มี ให้ส่งภาพทดสอบ
    return send_file("cache/warmed_frame.jpg", mimetype="image/jpeg")


@app.route("/api/status")
def api_status():
    current_time = time.time()
    with state_lock:
        last_ago = (
            (current_time - state["last_check_timestamp"])
            if state["last_check_timestamp"]
            else None
        )
        return jsonify({
            "status": state["status"],
            "error_message": state["error_message"],
            "fps": state["fps"],
            "is_api_busy": state["is_api_busy"],
            "is_audio_playing": state["is_audio_playing"],
            "last_check_ago": last_ago,
            "next_check_in": state["next_check_in"],
            "total_checks": state["total_checks"],
            "mask_count": state["mask_count"],
            "latest_score": state["latest_score"],
            "last_spoken_text": alert_service.last_spoken_text,
            "last_sentence_source": alert_service.last_sentence_source,
            "current_persona": thai_llm.current_persona,
            "enable_dynamic_llm": Config.ENABLE_DYNAMIC_LLM_ALERT,
        })


@app.route("/api/set_persona", methods=["POST"])
def api_set_persona():
    data = request.get_json(silent=True) or {}
    persona = data.get("persona", ThaiLLMPersona.FUNNY)
    thai_llm.set_persona(persona)
    # เจนประโยคตัวอย่างของ persona ทันที
    gen = thai_llm.generate_alert_sentence(persona)
    alert_service.last_spoken_text = gen["text"]
    alert_service.last_sentence_source = gen.get("source", "Thai LLM")
    return jsonify({
        "success": True,
        "persona": persona,
        "text": gen["text"],
        "source": gen.get("source", "Thai LLM"),
    })


@app.route("/api/toggle_llm", methods=["POST"])
def api_toggle_llm():
    data = request.get_json(silent=True) or {}
    enable = data.get("enable", True)
    Config.ENABLE_DYNAMIC_LLM_ALERT = bool(enable)
    return jsonify({"success": True, "enable": Config.ENABLE_DYNAMIC_LLM_ALERT})


@app.route("/api/generate_sentence", methods=["POST"])
def api_generate_sentence():
    gen = thai_llm.generate_alert_sentence()
    alert_service.last_spoken_text = gen["text"]
    alert_service.last_sentence_source = gen.get("source", "Thai LLM")
    return jsonify({
        "success": True,
        "text": gen["text"],
        "source": gen.get("source", "Thai LLM"),
        "persona": thai_llm.current_persona,
    })


@app.route("/api/check_now", methods=["POST"])
def api_check_now():
    with frame_lock:
        frame = latest_raw_frame
    if frame is not None:
        trigger_api_check(frame)
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "No frame available"}), 500


@app.route("/api/test_audio", methods=["POST"])
def api_test_audio():
    data = request.get_json(silent=True) or {}
    custom_text = data.get("text")
    success = alert_service.trigger_alert_async(custom_text=custom_text)
    return jsonify({
        "success": success,
        "text": alert_service.last_spoken_text,
        "source": alert_service.last_sentence_source,
    })


if __name__ == "__main__":
    port = 5000
    url = f"http://127.0.0.1:{port}"
    logger.info(f"🌐 กำลังเปิด Web Application ที่: {url}")

    # เริ่ม Background Thread อ่านกล้องตัวเดียวแบบเสถียร
    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    # เปิดเบราว์เซอร์อัตโนมัติใน 1 วินาที
    def open_browser():
        time.sleep(1.2)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    # เริ่ม Flask Server
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
