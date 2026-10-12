# Real-Time Face Mask Detection with AI for Thai & Thai LLM

ระบบตรวจจับการใส่หน้ากากอนามัยผ่าน Webcam แบบเรียลไทม์ โดยเชื่อมต่อบริการ **AI for Thai (NECTEC)** อย่างแท้จริง ผสานพลัง **Thai Large Language Model (Thai LLM)** สำหรับสร้างประโยคแจ้งเตือนอัตโนมัติแบบไดนามิก และสังเคราะห์เสียงพูดแจ้งเตือนด้วย **VAJA Text-to-Speech API** รองรับการทำงานทั้งแบบ **หน้าเว็บแดชบอร์ด (Web Application)** และ **โปรแกรมหน้าต่างในเครื่อง (Desktop GUI)**

---

## 🌟 ฟีเจอร์หลัก (Features)

1. **ตรวจจับหน้ากากอนามัยแบบเรียลไทม์ (AI for Thai - Face Mask Detection API)**
   - ส่งภาพใบหน้าไปประมวลผลที่ API จริง: `https://api.aiforthai.in.th/detectfacemask`
   - ระบบ **Auto Face Cropping**: ตรวจจับและครอบตัดเฉพาะบริเวณใบหน้าของผู้ใช้พร้อม Padding 30% ก่อนส่งเข้า API เพื่อความแม่นยำสูงสุดและป้องกันสัญญาณรบกวนจากโปสเตอร์/ผนังในพื้นหลัง
   - ส่งภาพแคปตรวจจับทุกๆ **5 วินาที** (Throttled) เพื่อความเสถียรและประหยัดโควตา
   - วิเคราะห์ผลลัพธ์: หากผลลัพธ์เป็น `Mask` ระบบจะแจ้งเตือน, หากเป็น `NoMask` หรือไม่พบหน้ากาก จะไม่แจ้งเตือน

2. **ระบบสร้างประโยคเตือนอัจฉริยะ (Thai LLM - Dynamic Alert Assistant)**
   - โมดูล [`thai_llm.py`](file:///C:/Users/dacha/OneDrive/Desktop/AILLM/thai_llm.py) ทำงานร่วมกับ Thai Large Language Model
   - รองรับ **4 บุคลิก (Personas)** ให้เลือกสลับได้ทันทีบน Web Dashboard:
     - 🎭 **โหมดกวนๆ / สนุกสนาน (Funny)**: เป็นกันเอง แซวขำๆ ชวนให้อมยิ้ม (เช่น *"หน้าตาดีไม่ต้องอายกล้อง ถอดแมสก์โชว์ออร่าหน่อยเร็ว"*)
     - 🌸 **โหมดสุภาพเรียบร้อย (Polite)**: นอบน้อม ไพเราะ เหมาะกับจุดต้อนรับ (เช่น *"ขอความกรุณาถอดหน้ากากอนามัยสักครู่นะคะ ขอบคุณค่ะ"*)
     - 🧑‍🏫 **โหมดครูฝ่ายปกครอง (Disciplinarian)**: เข้มงวด ดุดัน สั่งการชัดเจน (เช่น *"ถอดหน้ากากอนามัยทันที! ไม่ทำตามไม่ได้!"*)
     - 🤖 **โหมดหุ่นยนต์ AI ไฮเทค (Cyber/Sci-Fi)**: ล้ำสมัย แจ้งเตือนความปลอดภัยเชิงเทคนิค (เช่น *"ถอดหน้ากากออกเพื่อตรวจจับใบหน้าอย่างถูกต้อง"*)
   - เชื่อมต่อ API จริงกับ **ThaiLLM Playground (`https://api.thaillm.or.th`)** โมเดล `Typhoon-S-ThaiLLM-8B-Instruct` จาก NECTEC / SCB 10X สร้างประโยคสดๆ ทันที
   - พร้อมระบบ Fallback รองรับ **Typhoon API**, **Google Gemini**, หรือ **Local Curated Corpus** (ทำงานได้แม้ออฟไลน์)

3. **ระบบแจ้งเตือนด้วยเสียงสังเคราะห์ (AI for Thai - VAJA Text-to-Speech API)**
   - สังเคราะห์เสียงพูดจริงผ่าน API: `https://api.aiforthai.in.th/vaja`
   - นำประโยคที่ **Thai LLM** สร้างขึ้นส่งต่อไปยัง VAJA เพื่อให้ส่งเสียงพูดเตือนสดๆ
   - **Local Cache**: เก็บไฟล์เสียงแคชตาม hash ของประโยค เมื่อดาวน์โหลดแล้วครั้งต่อไปจะเล่นได้ทันทีแบบไม่มีดีเลย์ (0ms) และไม่เปลืองโควตา API
   - **Anti-overlapping Lock**: ระบบล็อกเธรดป้องกันไม่ให้เสียงพูดเล่นทับซ้อนกัน
   - **Cooldown Timer**: หน่วงเวลาไม่ให้ส่งเสียงเตือนซ้ำรัว ๆ เมื่อยังสวมหน้ากากอยู่

4. **รองรับการรัน 2 รูปแบบ (Dual-Mode Execution)**
   - **Web Application (`web_app.py`)**: หน้าเว็บแดชบอร์ดสวยงาม (`http://localhost:5000`) ปรับแต่ง Persona ของ Thai LLM ได้สดๆ กดสุ่มประโยคใหม่ สตรีมภาพกล้องสด ดูภาพครอป และรองรับการเปิดผ่านมือถือ/LAN
   - **Desktop GUI (`main.py`)**: รันเป็นหน้าต่าง OpenCV ในคอมพิวเตอร์โดยตรง มีขอบสีแจ้งเตือนและ HUD ภาษาไทย

---

## 🖥️ สองรูปแบบการใช้งาน (Dual-Mode Execution)

โปรเจกต์นี้ได้รับการออกแบบเป็นโมดูลอิสระ ทำให้คุณสามารถเลือกเปิดใช้งานได้ 2 โหมด:

### 1. โหมดหน้าต่างโปรแกรมในเครื่อง (Desktop GUI)
รันเป็นหน้าต่างโปรแกรม OpenCV ดั้งเดิม เหมาะกับการเปิดที่จุดคัดกรอง:
```powershell
.\venv\Scripts\python.exe main.py
```
* **ปุ่มควบคุม**:
  * `Q` หรือ `ESC` : ปิดโปรแกรมและคืนทรัพยากรกล้อง
  * `S` : แคปภาพส่งตรวจทันที (Manual Snapshot)
  * `T` : ทดสอบเล่นเสียงแจ้งเตือน VAJA

### 2. โหมดหน้าเว็บในเบราว์เซอร์ (Web Application)
รันเป็น Web Dashboard แสดงผลผ่าน Google Chrome / Microsoft Edge:
```powershell
.\venv\Scripts\python.exe web_app.py
```
* **ลิงก์เข้าใช้งาน**: เปิดเบราว์เซอร์ไปที่ **[http://localhost:5000](http://localhost:5000)**
* **จุดเด่น**:
  * แสดงสตรีมภาพสดจากกล้อง Webcam พร้อม HUD ภาษาไทย
  * แสดง **"ภาพใบหน้าที่แคปส่ง API ล่าสุด (Cropped Face)"** แบบเรียลไทม์
  * แสดงตัวนับถอยหลังการแคปตรวจครั้งถัดไป (ทุกๆ 5.0 วินาที)
  * แสดงคะแนนความน่าจะเป็น (Score/Confidence) จาก AI for Thai
  * **รองรับการเปิดผ่านมือถือ/LAN**: อุปกรณ์อื่นในวง Wi-Fi เดียวกัน สามารถพิมพ์ `http://<IPเครื่องของคุณ>:5000` เพื่อดูสถานะจากระยะไกลได้

---

## 🛑 วิธีการหยุดโปรแกรม (How to Stop)

1. **หากรันผ่านหน้าต่าง Desktop (`main.py`)**:
   * คลิกที่หน้าต่างกล้อง แล้วกดปุ่ม **`Q`** หรือ **`ESC`** บนคีย์บอร์ด
2. **หากรันผ่าน Terminal / PowerShell (`web_app.py` หรือ `main.py`)**:
   * คลิกที่หน้าต่าง Terminal แล้วกดปุ่ม **`Ctrl + C`** บนคีย์บอร์ดเพื่อหยุดระบบทันที

---

## 🔬 รายละเอียดเชิงลึกของ API แต่ละตัวจาก AI for Thai

### 1. Face Mask Detection API
* **Endpoint**: `https://api.aiforthai.in.th/detectfacemask`
* **Method**: `POST` (multipart/form-data)
* **Header**: `Apikey: <YOUR_API_KEY>`
* **Field**: `file` (ไฟล์ภาพ JPEG ของใบหน้า)
* **Response ตัวอย่าง**:
  ```json
  {
    "OK": true,
    "objects": [
      { "result": "NoMask", "score": "[0. 1.]" }
    ]
  }
  ```
  *(ถ้าสวมหน้ากากอนามัย `result` จะมีค่าเป็น `"Mask"`)*
* **ระบบ Face Cropping**:
  เนื่องจากโมเดลของ AI for Thai เป็น Face Classifier หากส่งภาพมุมกว้างทั้งห้องที่มีสิ่งของหรือโปสเตอร์สะท้อนแสง ระบบอาจสับสนได้ ในโค้ด [`mask_detector.py`](file:///C:/Users/dacha/OneDrive/Desktop/AILLM/mask_detector.py) จึงมีอัลกอริทึมค้นหาใบหน้าและ Crop เฉพาะใบหน้าพร้อม Padding 30% ก่อนส่งเสมอ ทำให้ผลการตรวจจับแม่นยำสูง

### 2. VAJA Text-to-Speech API
* **Endpoint**: `https://api.aiforthai.in.th/vaja`
* **Method**: `POST` (application/json)
* **Header**: `Apikey: <YOUR_API_KEY>`
* **Payload**:
  ```json
  {
    "text": "ถอดหน้ากากออกด้วยดิวะ",
    "speaker": "nana"
  }
  ```
* **Response ตัวอย่าง**:
  ```json
  {
    "msg": "success",
    "audio_url": "https://api.aiforthai.in.th/vaja-gst-result/storages/.../xxxx.wav"
  }
  ```
* โปรแกรมจะดาวน์โหลดไฟล์เสียงจาก `audio_url` มาบันทึกไว้ในโฟลเดอร์ `cache/` และใช้ `pygame.mixer` (พร้อม fallback `winsound` / PowerShell) เพื่อเล่นเสียงเตือน

### 3. Thai Large Language Model (Thai LLM)
* อยู่ในไฟล์ [`thai_llm.py`](file:///C:/Users/dacha/OneDrive/Desktop/AILLM/thai_llm.py)
* สร้างประโยคเตือนภาษาไทยอัตโนมัติแบบกระชับ (ไม่เกิน 15-20 คำ) สำหรับส่งเข้า VAJA TTS
* รองรับ 4 บุคลิก (funny, polite, teacher, cyber)
* รองรับ Typhoon API, Google Gemini หรือ Local Curated Corpus ออฟไลน์ 100%

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
AILLM/
│
├── .env.example              # ตัวอย่างไฟล์ตั้งค่าคอนฟิกและ API Key
├── .env                      # ไฟล์ตั้งค่าสำหรับใช้งานจริง (ใส่ API Key ที่นี่)
├── requirements.txt          # รายการไลบรารีที่จำเป็น
├── config.py                 # จัดการโหลดค่าและตรวจสอบ Environment Variables
├── thai_llm.py               # โมดูล Thai LLM สร้างประโยคเตือนแบบไดนามิก (4 Personas)
├── mask_detector.py          # โมดูล Face Cropping และเรียก AI for Thai detectfacemask API
├── tts_alert.py              # โมดูลเรียก AI for Thai VAJA TTS API, แคช และเล่นเสียงเตือน
├── ui_overlay.py             # โมดูลวาด HUD กรอบใบหน้า และตัวอักษรภาษาไทยบน OpenCV
├── main.py                   # โปรแกรม Desktop GUI: Webcam Loop และหน้าต่างกล้อง
├── web_app.py                # โปรแกรม Web Dashboard: สตรีมกล้องสด ควบคุม Thai LLM และแดชบอร์ด
├── test_system.py            # สคริปต์ Automated Test ทดสอบการทำงานทุกโมดูล
└── README.md                 # คู่มือการติดตั้งและคำอธิบายระบบอย่างละเอียด
```

---

## 🛠️ ขั้นตอนการติดตั้งและรันบน Windows (Step-by-Step Guide)

### ขั้นตอนที่ 1: ติดตั้ง Python (กรณีที่เครื่องยังไม่มี Python)
เปิด **PowerShell** แล้วติดตั้ง Python 3.11 ผ่าน `winget`:
```powershell
winget install Python.Python.3.11 --source winget
```

### ขั้นตอนที่ 2: สร้างและเปิดใช้งาน Virtual Environment
```powershell
cd C:\Users\dacha\OneDrive\Desktop\AILLM
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(หากติดสิทธิ์ความปลอดภัย ให้รัน `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

### ขั้นตอนที่ 3: ติดตั้ง Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### ขั้นตอนที่ 4: ตั้งค่า API Key ในไฟล์ `.env`
ใส่ API Key ที่ได้จาก [https://aiforthai.in.th](https://aiforthai.in.th) ในไฟล์ [`.env`](file:///C:/Users/dacha/OneDrive/Desktop/AILLM/.env):
```env
AIFORTHAI_API_KEY=JBciaUmiqUjKqxzInMlcDrTt3lTjXWq4
```

### ขั้นตอนที่ 5: เริ่มต้นใช้งาน

* **เปิดแบบหน้าเว็บ (แนะนำ)**:
  ```powershell
  .\venv\Scripts\python.exe web_app.py
  ```
  จากนั้นเปิดเบราว์เซอร์ไปที่: **http://localhost:5000**

* **เปิดแบบหน้าต่าง Desktop**:
  ```powershell
  .\venv\Scripts\python.exe main.py
  ```

---

## 🧪 การทดสอบระบบ (Automated System Test)

คุณสามารถรันชุดทดสอบอัตโนมัติเพื่อตรวจสอบการทำงานของทุกโมดูลได้ด้วยคำสั่ง:
```powershell
.\venv\Scripts\python.exe test_system.py
```

