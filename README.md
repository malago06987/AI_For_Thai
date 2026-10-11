# Real-Time Face Mask Detection with AI for Thai

ระบบตรวจจับการใส่หน้ากากอนามัยผ่าน Webcam แบบเรียลไทม์ โดยใช้ **OpenCV** และชุด API จาก **AI for Thai (NECTEC)** พร้อมระบบแจ้งเตือนด้วยเสียงสังเคราะห์ **VAJA TTS**

---

## 🌟 ฟีเจอร์หลัก (Features)

1. **ตรวจจับหน้ากากอนามัยแบบเรียลไทม์ (Face Mask Detection)**
   - เชื่อมต่อ API `https://api.aiforthai.in.th/detectfacemask`
   - ส่งภาพจากกล้องด้วย HTTP POST Multipart (`file`)
   - ใช้ Header `Apikey` สำหรับยืนยันตัวตน
   - แยกแยะผลลัพธ์: หากเป็น `Mask` ระบบจะแจ้งเตือน, หากเป็น `NoMask` หรือไม่พบหน้ากาก จะไม่แจ้งเตือน

2. **ระบบแจ้งเตือนด้วยเสียง (Voice Alert - VAJA API)**
   - สังเคราะห์เสียงพูดผ่าน `https://api.aiforthai.in.th/vaja`
   - ข้อความเตือน: `"กรุณาถอดหน้ากาก"`
   - เลือกใช้เสียง: `nana`
   - ดาวน์โหลดและเล่นเสียงจาก `audio_url`
   - มีระบบ **ป้องกันเสียงเตือนซ้ำ (Cooldown Timer)** และ **ห้ามเล่นเสียงซ้อนกัน (Thread Lock)**
   - มีระบบ Local Cache เพื่อประหยัดโควตาและลดความหน่วง

3. **หน้าจอแสดงผลและการทำงานแบบ Asynchronous**
   - แสดงภาพจากกล้อง Webcam พร้อม UI/HUD คมชัด รองรับภาษาไทยผ่าน Pillow
   - เรียก API เป็นช่วงเวลา (Throttled) เพื่อไม่ส่งภาพทุกเฟรม
   - ประมวลผล Network Request ใน Background Thread ทำให้ภาพบนหน้าจอลื่นไหล ไม่กระตุกหรือค้าง
   - จัดการกรณีต่อเน็ตไม่ได้ หรือ API ขัดข้องอย่างนุ่มนวล (Graceful Fallback)

4. **การจัดการความปลอดภัยและการตั้งค่า**
   - เก็บ API Key ในไฟล์ `.env` ไม่เปิดเผยคีย์ในโค้ด
   - มีไฟล์ `.env.example` เป็นตัวอย่าง
   - ควบคุมการตั้งค่ากล้อง ช่วงเวลาตรวจจับ และ Cooldown ผ่าน Config ได้สะดวก

5. **โมดูล Capgen (Image Captioning)**
   - แยกเป็นโมดูลอิสระ (`capgen.py`)
   - มีระบบตรวจสอบความปลอดภัย ไม่อนุญาตให้เรียก API จนกว่าจะระบุ URL ที่ถูกต้อง
   - ป้องกันการนำ URL ตรวจจับหน้ากากอนามัยมาใส่แทนโดยเด็ดขาด

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
AILLM/
│
├── .env.example              # ตัวอย่างไฟล์ตั้งค่าคอนฟิกและ API Key
├── .env                      # ไฟล์ตั้งค่าสำหรับใช้งานจริง (ใส่ API Key ที่นี่)
├── requirements.txt          # รายการไลบรารีที่จำเป็น
├── config.py                 # จัดการโหลดค่าและตรวจสอบ Environment Variables
├── mask_detector.py          # โมดูลเรียก Face Mask Detection API
├── tts_alert.py              # โมดูลเรียก VAJA TTS API และเล่นเสียงเตือน
├── ui_overlay.py             # โมดูลวาด HUD และตัวอักษรภาษาไทยบนเฟรม OpenCV
├── capgen.py                 # โมดูลสำหรับ Image Captioning (แยกอิสระ)
├── main.py                   # โปรแกรม Desktop: Webcam Loop, Worker Thread และ UI หน้าต่าง
├── web_app.py                # โปรแกรม Web Dashboard: เปิดดูผ่าน Web Browser (http://localhost:5000)
└── README.md                 # คู่มือการติดตั้งและใช้งาน
```

---

## 🛠️ ขั้นตอนการติดตั้งและรันบน Windows (Step-by-Step Guide)

### ขั้นตอนที่ 1: ติดตั้ง Python (กรณีที่เครื่องยังไม่มี Python)
เปิด **PowerShell** หรือ **Command Prompt** แล้วติดตั้ง Python 3.11 หรือใหม่กว่าผ่าน `winget`:
```powershell
winget install Python.Python.3.11 --source winget
```
*(หรือดาวน์โหลดตัวติดตั้งจาก [python.org](https://www.python.org/downloads/) และอย่าลืมติ๊กถูกที่ช่อง **"Add Python to PATH"** ขณะติดตั้ง)*

เมื่อติดตั้งเสร็จแล้ว ให้เปิด Terminal ใหม่แล้วตรวจสอบเวอร์ชัน:
```powershell
python --version
```

---

### ขั้นตอนที่ 2: ไปที่โฟลเดอร์โปรเจกต์และสร้าง Virtual Environment
```powershell
cd C:\Users\dacha\OneDrive\Desktop\AILLM
python -m venv venv
```

เปิดใช้งาน Virtual Environment:
```powershell
.\venv\Scripts\Activate.ps1
```
*(หากพบปัญหาเรื่อง Execution Policy ให้รันคำสั่ง `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

---

### ขั้นตอนที่ 3: ติดตั้ง Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

### ขั้นตอนที่ 4: ตั้งค่า API Key ในไฟล์ `.env`
1. เข้าไปที่เว็บไซต์ [https://aiforthai.in.th](https://aiforthai.in.th)
2. เข้าสู่ระบบ แล้วไปที่เมนู **Profile** หรือ **Developer** เพื่อคัดลอก **API Key** ของคุณ
3. เปิดไฟล์ `.env` ในโฟลเดอร์โปรเจกต์ และใส่ API Key ของคุณ:
```env
AIFORTHAI_API_KEY=ใส่_API_Key_จริงของคุณที่นี่
```

---

### ขั้นตอนที่ 5: เริ่มรันโปรแกรม
```powershell
python main.py
```

---

## 🎮 ปุ่มควบคุมขณะโปรแกรมทำงาน (Controls)

| ปุ่มกด | หน้าที่ |
| :---: | :--- |
| **Q** หรือ **ESC** | ปิดโปรแกรมและคืนทรัพยากรกล้อง |
| **S** | สั่งส่งภาพไปตรวจจับที่ API ทันที (Manual Snapshot) |
| **T** | ทดสอบเล่นเสียงแจ้งเตือน VAJA ทันที |

---

## 🧪 การทดสอบโมดูลแยกส่วน (Unit / Component Testing)

### 1. ทดสอบโมดูล Capgen
```powershell
python capgen.py
```
*(ระบบจะแสดงผลการทดสอบความปลอดภัย การปฏิเสธ URL ไม่ถูกต้อง และการทำงานของ Mock Mode)*

### 2. ทดสอบตรวจจับหน้ากากอนามัยกับรูปภาพเดี่ยว
```python
from mask_detector import MaskDetectionClient

client = MaskDetectionClient()
result = client.detect_from_file("test.jpg")
print(result)
```

### 3. ทดสอบเสียงแจ้งเตือน VAJA
```python
from tts_alert import VajaAlertService

service = VajaAlertService()
service.trigger_alert_async()
```
