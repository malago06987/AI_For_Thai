# Real-Time Face Mask Detection with AI for Thai

ระบบตรวจจับการใส่หน้ากากอนามัยผ่าน Webcam แบบเรียลไทม์ โดยเชื่อมต่อบริการ **AI for Thai (NECTEC)** อย่างแท้จริง พร้อมระบบแจ้งเตือนด้วยเสียงสังเคราะห์ **VAJA TTS** รองรับการทำงานทั้งแบบ **โปรแกรมหน้าต่างในเครื่อง (Desktop GUI)** และ **หน้าเว็บแดชบอร์ด (Web Application)**

---

## 🌟 ฟีเจอร์หลัก (Features)

1. **ตรวจจับหน้ากากอนามัยแบบเรียลไทม์ (Face Mask Detection)**
   - ส่งภาพใบหน้าไปประมวลผลที่ API จริง: `https://api.aiforthai.in.th/detectfacemask`
   - ระบบ **Auto Face Cropping**: ตรวจจับและครอบตัดเฉพาะบริเวณใบหน้าของผู้ใช้พร้อม Padding 30% ก่อนส่งเข้า API เพื่อความแม่นยำสูงสุดและป้องกันสัญญาณรบกวนจากโปสเตอร์/ผนังในพื้นหลัง
   - ส่งภาพแคปตรวจจับทุกๆ **5 วินาที** (Throttled) เพื่อความเสถียรและประหยัดโควตา
   - วิเคราะห์ผลลัพธ์: หากผลลัพธ์เป็น `Mask` ระบบจะแจ้งเตือน, หากเป็น `NoMask` หรือไม่พบหน้ากาก จะไม่แจ้งเตือน

2. **ระบบแจ้งเตือนด้วยเสียงสังเคราะห์ (VAJA Text-to-Speech API)**
   - สังเคราะห์เสียงพูดจริงผ่าน API: `https://api.aiforthai.in.th/vaja`
   - ข้อความเตือน: `"ถอดหน้ากากออกด้วยดิวะ"` ด้วยเสียงผู้หญิง: `nana`
   - ดาวน์โหลดและเล่นเสียงจาก `audio_url` ที่ Server ส่งกลับมา
   - **Local Cache**: เก็บไฟล์เสียงไว้ที่ `cache/vaja_nana_alert.wav` เมื่อดาวน์โหลดครั้งแรกแล้ว ครั้งต่อไปจะเล่นได้ทันทีแบบไม่มีดีเลย์ (0ms) และไม่เปลืองโควตา API
   - **Anti-overlapping Lock**: ระบบล็อกเธรดป้องกันไม่ให้เสียงพูดเล่นทับซ้อนกัน
   - **Cooldown Timer**: หน่วงเวลาไม่ให้ส่งเสียงเตือนซ้ำรัว ๆ เมื่อยังสวมหน้ากากอยู่

3. **รองรับการรัน 2 รูปแบบ (Dual-Mode Execution)**
   - **Desktop GUI (`main.py`)**: รันเป็นหน้าต่างโปรแกรม OpenCV ในคอมพิวเตอร์โดยตรง มีขอบสีแจ้งเตือนและตัวอักษรภาษาไทย
   - **Web Application (`web_app.py`)**: รันเป็น Web Dashboard บนเบราว์เซอร์ (`http://localhost:5000`) ดูภาพสด ดูภาพที่แคปตรวจล่าสุด และเข้าถึงผ่านมือถือ/แท็บเล็ตในวง Wi-Fi เดียวกันได้

4. **โมดูล Capgen (Image Captioning) แยกอิสระ**
   - เตรียมโมดูล [`capgen.py`](file:///C:/Users/dacha/OneDrive/Desktop/AILLM/capgen.py) สำหรับสร้างคำบรรยายภาพภาษาไทย
   - มีระบบ **Safety Guard**: สกัดกั้นการใช้ URL ตรวจจับหน้ากากอนามัย (`detectfacemask`) มาสวมรอยโดยเด็ดขาด
   - บังคับตรวจสอบ URL ก่อนยิง Network จริง และมี Mock Mode ให้ทดสอบล่วงหน้า

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

### 3. Thai Image Captioning (Capgen)
* อยู่ในไฟล์ [`capgen.py`](file:///C:/Users/dacha/OneDrive/Desktop/AILLM/capgen.py)
* ออกแบบตามข้อกำหนดเพื่อความปลอดภัย โดยระบบจะปฏิเสธและตัดการทำงานทันทีหากมีการนำ URL ของบริการตรวจจับหน้ากากอนามัยมาใส่แทน
* ต้องทำการยืนยัน Endpoint ผ่านเมธอด `verify_and_set_endpoint(url)` ก่อนใช้งานจริง

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
AILLM/
│
├── .env.example              # ตัวอย่างไฟล์ตั้งค่าคอนฟิกและ API Key
├── .env                      # ไฟล์ตั้งค่าสำหรับใช้งานจริง (ใส่ API Key ที่นี่)
├── requirements.txt          # รายการไลบรารีที่จำเป็น
├── config.py                 # จัดการโหลดค่าและตรวจสอบ Environment Variables
├── mask_detector.py          # โมดูล Face Cropping และเรียก Face Mask Detection API
├── tts_alert.py              # โมดูลเรียก VAJA TTS API, ดาวน์โหลดแคช และเล่นเสียงเตือน
├── ui_overlay.py             # โมดูลวาด HUD กรอบใบหน้า และตัวอักษรภาษาไทยบน OpenCV
├── capgen.py                 # โมดูลสำหรับ Image Captioning (แยกอิสระ พร้อมระบบป้องกัน)
├── main.py                   # โปรแกรม Desktop GUI: Webcam Loop และหน้าต่างกล้อง
├── web_app.py                # โปรแกรม Web Dashboard: สตรีมกล้องสดและแสดงผลบนเบราว์เซอร์
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
และทดสอบโมดูล Capgen:
```powershell
.\venv\Scripts\python.exe capgen.py
```
