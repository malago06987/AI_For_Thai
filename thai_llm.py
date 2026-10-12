"""
Module: thai_llm.py
โมดูลผสานการทำงานกับ Thai Large Language Model (Thai LLM)
สำหรับสร้างประโยคแจ้งเตือนแบบไดนามิก (Dynamic Alert Generation) หลากหลายบุคลิก (Personas)
ส่งต่อไปยัง AI for Thai (VAJA TTS)

รองรับ:
1. Typhoon API (SCB 10X - api.opentyphoon.ai)
2. Google Gemini API (gemini-1.5-flash / gemini-2.0-flash / gemini-2.5-flash)
3. OpenAI-Compatible API (OpenThaiGPT, vLLM, Ollama)
4. Local Thai Persona Corpus (คลังประโยคภาษาไทยคุณภาพสูง ทำงานได้ 100% แม้ไม่มี API Key หรือออฟไลน์)
"""

import os
import random
import logging
import json
import re
from typing import Dict, Any, Optional, List
import requests

from config import Config

logger = logging.getLogger("ThaiLLMService")


class ThaiLLMPersona:
    POLITE = "polite"
    FUNNY = "funny"
    TEACHER = "teacher"
    CYBER = "cyber"

    LABELS = {
        POLITE: "🌸 โหมดสุภาพเรียบร้อย (Polite)",
        FUNNY: "🎭 โหมดกวนๆ / สนุกสนาน (Funny)",
        TEACHER: "🧑‍🏫 โหมดครูฝ่ายปกครอง (Disciplinarian)",
        CYBER: "🤖 โหมดหุ่นยนต์ AI ไฮเทค (Cyber/Sci-Fi)",
    }

    DESCRIPTIONS = {
        POLITE: "สุภาพ นอบน้อม พูดจาไพเราะ เหมาะกับงานต้อนรับและบริการ",
        FUNNY: "เป็นกันเอง แซวขำๆ ติดตลก ชวนให้อมยิ้ม",
        TEACHER: "เข้มงวด ดุดัน มีระเบียบวินัย ตามสไตล์อาจารย์ฝ่ายปกครอง",
        CYBER: "เสียงสั่งการเชิงเทคนิค ล้ำสมัย ดั่งระบบความปลอดภัยอัตโนมัติ",
    }


# คลังประโยคภาษาไทยคุณภาพสูงสำหรับแต่ละ Persona (ป้องกันระบบล่ม / ทำงานได้ทันที 0ms)
LOCAL_PERSONA_CORPUS: Dict[str, List[str]] = {
    ThaiLLMPersona.POLITE: [
        "ขอความกรุณาถอดหน้ากากอนามัยสักครู่นะคะ ขอบคุณค่ะ",
        "รบกวนถอดแมสก์ออกชั่วคราว เพื่อความชัดเจนในการคัดกรองนะคะ",
        "เพื่อความปลอดภัย กรุณาเปิดเผยใบหน้าสักครู่ค่ะ ยินดีต้อนรับนะคะ",
        "โปรดถอดหน้ากากอนามัยออกชั่วขณะ ขอบคุณที่ให้ความร่วมมือค่ะ",
        "ขออภัยในความไม่สะดวก รบกวนปลดหน้ากากอนามัยลงสักนิดนะคะ",
    ],
    ThaiLLMPersona.FUNNY: [
        "ถอดหน้ากากออกด้วยดิวะ",
        "หล่อสวยขนาดนี้ จะใส่แมสก์ปิดบังความปังไว้ทำไม ถอดเลย!",
        "จะไปปล้นธนาคารที่ไหนครับ ถอดหน้ากากเดี๋ยวนี้เลย",
        "หน้าตาดีไม่ต้องอายกล้อง ถอดแมสก์โชว์ออร่าหน่อยเร็ว",
        "ใส่แมสก์มิดชิดขนาดนี้ นึกว่านินจามาตรวจงาน ถอดออกด่วน!",
        "กล้องจับความหล่อไม่ได้ เพราะติดแมสก์ ถอดออกสักวินาทีนะจ๊ะ",
    ],
    ThaiLLMPersona.TEACHER: [
        "นักเรียน! ถอดหน้ากากอนามัยเดี๋ยวนี้ อย่าให้ครูต้องพูดซ้ำ!",
        "ตรวจระเบียบใบหน้า ถอดหน้ากากออกทันที ห้ามชักช้า!",
        "นี่ห้องเรียนไม่ใช่สมรภูมิรบ ถอดแมสก์ออกแล้วเงยหน้าขึ้นมา!",
        "กฎระเบียบมีไว้ปฏิบัติตาม ปลดหน้ากากออกเดี๋ยวนี้!",
        "เธอตรงนั้นน่ะ ถอดหน้ากากอนามัยให้เห็นใบหน้าชัดเจนเร็วเข้า!",
    ],
    ThaiLLMPersona.CYBER: [
        "ตรวจพบสิ่งกีดขวางใบหน้า กรุณาปลดหน้ากากเพื่อยืนยันอัตลักษณ์บุคคล",
        "ระบบรักษาความปลอดภัยแจ้งเตือน โปรดถอดหน้ากากอนามัยเพื่อดำเนินการต่อ",
        "สแกนใบหน้าไม่สมบูรณ์เนื่องจากหน้ากาก ปลดอุปกรณ์คลุมใบหน้าทันที",
        "คำสั่งระบบขั้นสูง: เปิดเผยใบหน้าเพื่อการประมวลผลทางชีวมิติ",
    ],
}


class ThaiLLMService:
    """
    บริการสร้างประโยคแจ้งเตือนอัจฉริยะผ่าน Thai LLM
    ผสาน API จริง พร้อม fallback สู่ Local Corpus เพื่อความเสถียรสูงสุด
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        persona: Optional[str] = None,
        thaillm_key: Optional[str] = None,
        thaillm_url: Optional[str] = None,
        thaillm_model: Optional[str] = None,
        typhoon_key: Optional[str] = None,
        gemini_key: Optional[str] = None,
    ):
        self.provider = (provider or getattr(Config, "THAI_LLM_PROVIDER", "auto")).strip().lower()
        self.current_persona = (persona or getattr(Config, "THAI_LLM_PERSONA", ThaiLLMPersona.FUNNY)).strip().lower()
        if self.current_persona not in LOCAL_PERSONA_CORPUS:
            self.current_persona = ThaiLLMPersona.FUNNY

        # ThaiLLM Playground Configuration
        self.thaillm_key = (thaillm_key or getattr(Config, "THAILLM_API_KEY", "")).strip()
        self.thaillm_url = (
            thaillm_url or getattr(Config, "THAILLM_API_URL", "https://api.thaillm.or.th/v1/chat/completions")
        ).strip()
        self.thaillm_model = (
            thaillm_model or getattr(Config, "THAILLM_MODEL", "Typhoon-S-ThaiLLM-8B-Instruct")
        ).strip()

        self.typhoon_key = (typhoon_key or getattr(Config, "TYPHOON_API_KEY", "")).strip()
        self.gemini_key = (gemini_key or getattr(Config, "GEMINI_API_KEY", "")).strip()

        # เก็บประโยคล่าสุดที่ถูกเลือกใช้งาน
        self.last_generated_text = (
            Config.VAJA_TEXT
            if hasattr(Config, "VAJA_TEXT") and Config.VAJA_TEXT
            else "ถอดหน้ากากออกด้วยดิวะ"
        )
        self.last_source = "Default"

    def set_persona(self, persona: str) -> bool:
        """เปลี่ยน Persona การแจ้งเตือน"""
        clean_p = persona.strip().lower()
        if clean_p in LOCAL_PERSONA_CORPUS:
            self.current_persona = clean_p
            logger.info(f"เปลี่ยน Thai LLM Persona เป็น: {clean_p}")
            return True
        return False

    def generate_alert_sentence(self, persona: Optional[str] = None) -> Dict[str, Any]:
        """
        สร้างประโยคเตือนภาษาไทยแบบไดนามิก
        1. ลองยิง ThaiLLM Playground API (thaillm.or.th) หากมี API Key
        2. ลองยิง Typhoon API / Gemini API
        3. หากไม่มี API Key หรือ Network ขัดข้อง จะสุ่มเลือกจาก Local Persona Corpus ทันที
        """
        active_persona = (persona or self.current_persona).strip().lower()
        if active_persona not in LOCAL_PERSONA_CORPUS:
            active_persona = ThaiLLMPersona.FUNNY

        # 1. พยายามเรียก ThaiLLM Playground API (thaillm.or.th)
        if self.thaillm_key and self.thaillm_key not in ("your_thaillm_key", ""):
            llm_res = self._call_thaillm_playground(active_persona)
            if llm_res.get("success"):
                self.last_generated_text = llm_res["text"]
                self.last_source = f"ThaiLLM Playground ({self.thaillm_model})"
                return {
                    "text": llm_res["text"],
                    "persona": active_persona,
                    "source": self.last_source,
                    "is_live_api": True,
                }

        # 2. พยายามเรียก Typhoon LLM API (SCB 10X Thai LLM) หากมีการตั้งค่า Key
        if self.typhoon_key and self.typhoon_key not in ("your_typhoon_key", ""):
            llm_res = self._call_typhoon_llm(active_persona)
            if llm_res.get("success"):
                self.last_generated_text = llm_res["text"]
                self.last_source = "Thai LLM (Typhoon SCB 10X)"
                return {
                    "text": llm_res["text"],
                    "persona": active_persona,
                    "source": self.last_source,
                    "is_live_api": True,
                }

        # 3. พยายามเรียก Gemini API หากมีการตั้งค่า Key
        if self.gemini_key and self.gemini_key not in ("your_gemini_key", ""):
            llm_res = self._call_gemini_llm(active_persona)
            if llm_res.get("success"):
                self.last_generated_text = llm_res["text"]
                self.last_source = "Thai LLM (Google Gemini)"
                return {
                    "text": llm_res["text"],
                    "persona": active_persona,
                    "source": self.last_source,
                    "is_live_api": True,
                }

        # 4. Fallback: สุ่มจาก Local Curated Corpus ประจำ Persona นั้นๆ
        corpus = LOCAL_PERSONA_CORPUS.get(active_persona, LOCAL_PERSONA_CORPUS[ThaiLLMPersona.FUNNY])
        selected_text = random.choice(corpus)
        self.last_generated_text = selected_text
        self.last_source = f"Thai LLM Corpus ({ThaiLLMPersona.LABELS.get(active_persona, active_persona)})"

        return {
            "text": selected_text,
            "persona": active_persona,
            "source": self.last_source,
            "is_live_api": False,
        }

    def _build_prompt(self, persona: str) -> str:
        """สร้าง Prompt กำกับให้ LLM แต่งประโยคสั้น กระชับ สำหรับส่งเข้า TTS"""
        persona_guide = {
            ThaiLLMPersona.POLITE: "น้ำเสียงสุภาพ ไพเราะ นอบน้อม มีคำลงท้าย เช่น คะ/ค่ะ",
            ThaiLLMPersona.FUNNY: "น้ำเสียงกวนๆ ขำขัน เป็นกันเอง แซวคนใส่หน้ากากเบาๆ",
            ThaiLLMPersona.TEACHER: "น้ำเสียงครูฝ่ายปกครอง เข้มงวด ดุดัน สั่งให้ถอดหน้ากากทันที",
            ThaiLLMPersona.CYBER: "น้ำเสียงระบบ AI ไฮเทค ล้ำสมัย แจ้งเตือนความปลอดภัยเชิงเทคนิค",
        }
        guide = persona_guide.get(persona, persona_guide[ThaiLLMPersona.FUNNY])

        return (
            f"คุณคือโมเดลภาษาไทย (Thai LLM) หน้าที่ของคุณคือสร้างประโยคแจ้งเตือนสั้นๆ ภาษาไทย 1 ประโยค "
            f"เพื่อบอกให้ผู้ใช้ที่ยืนหน้ากล้อง 'ถอดหน้ากากอนามัยออก' "
            f"โดยมีเงื่อนไขดังนี้:\n"
            f"1. สไตล์: {guide}\n"
            f"2. ความยาว: สั้นกระชับไม่เกิน 15-20 คำ เพื่อให้ระบบสังเคราะห์เสียง VAJA TTS พูดได้ชัดเจน\n"
            f"3. ตอบเฉพาะประโยคแจ้งเตือนภาษาไทยเท่านั้น ไม่ต้องใส่เครื่องหมายคำพูด ไม่ต้องมีคำเกริ่นนำใดๆ ทั้งสิ้น"
        )

    def _call_thaillm_playground(self, persona: str) -> Dict[str, Any]:
        """เรียก ThaiLLM Playground API (https://api.thaillm.or.th/v1/chat/completions)"""
        try:
            headers = {
                "Authorization": f"Bearer {self.thaillm_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.thaillm_model,
                "messages": [
                    {"role": "user", "content": self._build_prompt(persona)},
                ],
                "max_tokens": 70,
                "temperature": 0.8,
            }
            resp = requests.post(self.thaillm_url, headers=headers, json=payload, timeout=8.0)
            if resp.status_code == 200:
                data = resp.json()
                raw_content = data["choices"][0]["message"]["content"]
                # ทำความสะอาดข้อความ ลบแท็ก <think> และเครื่องหมายคำพูด
                clean_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
                clean_content = clean_content.strip('"').strip("'").strip("`")
                # เอาเฉพาะบรรทัดแรกหากมีการขึ้นบรรทัดใหม่
                clean_content = clean_content.split("\n")[0].strip()
                if clean_content:
                    logger.info(f"✨ ThaiLLM Playground Response ({self.thaillm_model}): '{clean_content}'")
                    return {"success": True, "text": clean_content}
            logger.warning(f"ThaiLLM Playground API error: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.warning(f"ThaiLLM Playground API request failed: {e}")
        return {"success": False}

    def _call_typhoon_llm(self, persona: str) -> Dict[str, Any]:
        """เรียก Typhoon API (SCB 10X Thai LLM)"""
        try:
            url = "https://api.opentyphoon.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.typhoon_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "typhoon-v1.5-instruct",
                "messages": [
                    {"role": "system", "content": "คุณคือผู้ช่วย AI ภาษาไทยที่แต่งประโยคเตือนสั้นกระชับ"},
                    {"role": "user", "content": self._build_prompt(persona)},
                ],
                "max_tokens": 50,
                "temperature": 0.8,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip().strip('"').strip("'")
                if text:
                    return {"success": True, "text": text}
            logger.warning(f"Typhoon API error: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.warning(f"Typhoon API request failed: {e}")
        return {"success": False}

    def _call_gemini_llm(self, persona: str) -> Dict[str, Any]:
        """เรียก Google Gemini REST API"""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
            payload = {
                "contents": [
                    {"parts": [{"text": self._build_prompt(persona)}]}
                ],
                "generationConfig": {
                    "maxOutputTokens": 50,
                    "temperature": 0.8,
                }
            }
            resp = requests.post(url, json=payload, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    text = candidates[0]["content"]["parts"][0]["text"].strip().strip('"').strip("'")
                    if text:
                        return {"success": True, "text": text}
            logger.warning(f"Gemini API error: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.warning(f"Gemini API request failed: {e}")
        return {"success": False}


# Singleton Instance
thai_llm = ThaiLLMService()
