# 🚀 Production Deployment Guide

Quick guide to deploy for hackathon demo and production.

---

## 📱 Mobile-Friendly Streamlit App

### Quick Start (Local Network):

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Install system dependencies
sudo apt-get install poppler-utils libgl1 libzbar0  # Ubuntu
# brew install poppler zbar  # macOS

# 3. Put best.pt in root directory

# 4. Run app
streamlit run mobile_app.py --server.port 8501 --server.address 0.0.0.0
```

**Access from mobile:**
1. Find your computer's IP: `ip addr show` (Linux) or `ipconfig` (Windows)
2. On phone's browser: `http://YOUR_IP:8501`
3. Make sure phone and computer on same WiFi!

---

## 🌐 Public Deployment (for demo)

### Option 1: Streamlit Cloud (Easiest - 5 min)

```bash
# 1. Push to GitHub (already done!)

# 2. Go to share.streamlit.io
# 3. Connect GitHub repo
# 4. Select mobile_app.py
# 5. Add secrets (if needed)
# 6. Deploy!
```

**Note:** Upload best.pt to GitHub LFS or provide URL in secrets.

### Option 2: ngrok (Local tunnel - 2 min)

```bash
# 1. Install ngrok
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xvzf ngrok-v3-stable-linux-amd64.tgz

# 2. Run Streamlit locally
streamlit run mobile_app.py

# 3. In another terminal, create tunnel
./ngrok http 8501

# 4. Share the ngrok URL (https://xxxx.ngrok.io)
```

**Perfect for hackathon demo!** Share URL with judges, they can test on their phones.

### Option 3: Docker (Production-ready)

```dockerfile
# Dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "mobile_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build and run
docker build -t document-scanner .
docker run -p 8501:8501 document-scanner
```

---

## 🔒 Security Considerations

### 1. QR Code Safety

**Already implemented in qr_decoder.py:**
- Data sanitization
- Type validation
- No automatic URL execution

**Additional protection:**

```python
# Add to qr_decoder.py
import re
from urllib.parse import urlparse

def is_safe_url(url):
    """Check if URL is safe"""
    # Block dangerous protocols
    dangerous = ['javascript:', 'data:', 'file:', 'vbscript:']
    if any(url.lower().startswith(d) for d in dangerous):
        return False

    # Block suspicious patterns
    suspicious = ['<script', 'onerror=', 'onclick=']
    if any(s in url.lower() for s in suspicious):
        return False

    return True
```

### 2. File Upload Limits

Add to mobile_app.py:

```python
# Limit file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

if uploaded_file and uploaded_file.size > MAX_FILE_SIZE:
    st.error("File too large! Max 10MB")
    st.stop()

# Validate file type
ALLOWED_TYPES = ['image/jpeg', 'image/png', 'application/pdf']
if uploaded_file.type not in ALLOWED_TYPES:
    st.error("Invalid file type!")
    st.stop()
```

### 3. Input Sanitization

```python
# Sanitize filenames
import re
def sanitize_filename(filename):
    # Remove dangerous characters
    return re.sub(r'[^\w\s\-\.]', '', filename)
```

### 4. Rate Limiting

For production API:

```python
from functools import wraps
from time import time

requests = {}

def rate_limit(max_requests=10, window=60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ip = get_client_ip()
            now = time()

            if ip not in requests:
                requests[ip] = []

            # Clean old requests
            requests[ip] = [t for t in requests[ip] if now - t < window]

            if len(requests[ip]) >= max_requests:
                raise Exception("Rate limit exceeded")

            requests[ip].append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

---

## 📊 Environment Variables

Create `.env` file:

```bash
# Model settings
MODEL_PATH=best.pt
CONF_THRESHOLD=0.25
IOU_THRESHOLD=0.45

# Security
MAX_FILE_SIZE=10485760
ALLOWED_EXTENSIONS=jpg,jpeg,png,pdf

# Optional: Analytics
ENABLE_ANALYTICS=false
```

Load in app:

```python
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_PATH = os.getenv('MODEL_PATH', 'best.pt')
```

---

## 🎯 Quick Deployment for Hackathon Demo

**Recommended: ngrok (2 minutes)**

```bash
# Terminal 1
streamlit run mobile_app.py

# Terminal 2
ngrok http 8501

# Share the HTTPS URL with judges!
```

**Why ngrok:**
- ✅ Works immediately
- ✅ HTTPS (camera needs HTTPS on mobile!)
- ✅ Share one URL
- ✅ Works from anywhere
- ✅ Free tier sufficient

---

## 📱 Mobile Camera Requirements

**Important:** Modern browsers require HTTPS for camera access!

- ✅ localhost (works without HTTPS)
- ✅ ngrok (provides HTTPS)
- ✅ Streamlit Cloud (HTTPS by default)
- ❌ Local IP (http://192.168.x.x) - camera won't work!

**Solution:** Use ngrok for mobile demos.

---

## 🔧 Troubleshooting

### Camera not working on mobile:
- Check HTTPS (must be https://)
- Check browser permissions
- Try different browser (Chrome recommended)

### Model not loading:
```bash
# Check model exists
ls -lh best.pt

# Check path
python -c "from inference import find_model; print(find_model())"
```

### Dependencies issues:
```bash
# Reinstall
pip install --force-reinstall -r requirements.txt

# Check system deps
dpkg -l | grep -E 'poppler|libgl|libzbar'
```

### Port already in use:
```bash
# Kill process on port 8501
lsof -ti:8501 | xargs kill -9

# Or use different port
streamlit run mobile_app.py --server.port 8502
```

---

## 🎓 Best Practices

### For Hackathon:
1. **Use ngrok** - easiest, works great
2. **Test on your phone first**
3. **Have backup screenshots** in case wifi issues
4. **Share URL in advance** to judges

### For Production:
1. **Use proper hosting** (AWS/GCP/Azure)
2. **Add authentication**
3. **Implement rate limiting**
4. **Monitor errors** (Sentry)
5. **Use CDN** for model files
6. **Add logging**

---

## ⚡ Performance Tips

### 1. Model Optimization:
```python
# Use FP16 for faster inference
model = YOLO('best.pt')
model.to('cuda').half()  # If GPU available
```

### 2. Caching:
```python
@st.cache_resource
def load_model():
    return YOLO('best.pt')
```

### 3. Async Processing:
For production API, use async workers (Celery/RQ).

---

## 📝 Checklist Before Demo

- [ ] Model (`best.pt`) in correct location
- [ ] All dependencies installed
- [ ] System dependencies installed
- [ ] App runs locally
- [ ] Tested on mobile device
- [ ] ngrok tunnel working
- [ ] URL shared with judges
- [ ] Backup plan ready (screenshots)
- [ ] Presentation ready

---

## 🆘 Emergency Backup Plan

If deployment fails during demo:

1. **Show local version** on laptop screen
2. **Use pre-recorded video** (record now!)
3. **Show screenshots** from testing
4. **Explain architecture** with diagrams

**Record demo video NOW:**
```bash
# While running app, record screen
# Show: upload → detect → QR decode → validate → download
```

---

## 📞 Quick Commands Reference

```bash
# Start app locally
streamlit run mobile_app.py

# Start app for network access
streamlit run mobile_app.py --server.address 0.0.0.0

# Start ngrok tunnel
ngrok http 8501

# Check if running
curl http://localhost:8501

# Kill app
pkill -f streamlit

# View logs
streamlit run mobile_app.py --logger.level=debug
```

---

Good luck with the hackathon! 🚀
