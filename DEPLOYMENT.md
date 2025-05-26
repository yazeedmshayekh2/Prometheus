# 🚀 Free Deployment Guide

## Option 1: Railway (Recommended)

### Step 1: Prepare Your Code
1. Push your code to GitHub (create a new repository)
2. Make sure these files are in your repo:
   - `app.py` (main application)
   - `requirements-deploy.txt` (simplified dependencies)
   - `Procfile` (tells Railway how to run the app)
   - `railway.json` (Railway configuration)

### Step 2: Deploy to Railway
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway will automatically detect it's a Python app
6. Set these environment variables in Railway dashboard:
   - `PORT`: (Railway sets this automatically)
   - Add your database connection string if needed

### Step 3: Configure Build
1. In Railway dashboard, go to Settings
2. Under "Build", set:
   - Build Command: `pip install -r requirements-deploy.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### Step 4: Deploy
- Railway will automatically deploy when you push to GitHub
- You'll get a public URL like: `https://your-app-name.railway.app`

---

## Option 2: Render

### Step 1: Prepare
1. Push code to GitHub
2. Create `render.yaml`:

```yaml
services:
  - type: web
    name: insurance-assistant
    env: python
    buildCommand: pip install -r requirements-deploy.txt
    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
```

### Step 2: Deploy
1. Go to [render.com](https://render.com)
2. Connect GitHub
3. Create new "Web Service"
4. Select your repo
5. Render will auto-deploy

---

## Option 3: Fly.io

### Step 1: Install Fly CLI
```bash
curl -L https://fly.io/install.sh | sh
```

### Step 2: Create Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

COPY . .
EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Step 3: Deploy
```bash
fly auth login
fly launch
fly deploy
```

---

## 🔧 Environment Variables Needed

For any hosting service, you may need to set:
- `DATABASE_CONNECTION_STRING`: Your database connection
- `PORT`: (usually set automatically by hosting service)

---

## 📝 Quick Start Commands

```bash
# 1. Create simplified requirements
cp requirements-deploy.txt requirements.txt

# 2. Test locally first
python app.py

# 3. Push to GitHub
git add .
git commit -m "Deploy to cloud"
git push origin main
```

---

## 🎯 Recommended: Railway

**Why Railway?**
- ✅ Easiest setup (just connect GitHub)
- ✅ Automatic deployments
- ✅ Good free tier (500 hours/month)
- ✅ Built-in database options
- ✅ Custom domains on free tier

**Your app will be live at:** `https://your-app-name.railway.app`

Share this URL with your leader for testing! 🚀 