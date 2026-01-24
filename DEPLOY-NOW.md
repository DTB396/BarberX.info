# 🚀 DEPLOY NOW - Railway (Fastest Option)

## One-Command Deploy to Production

```powershell
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Deploy (automatically creates PostgreSQL database)
railway up

# Get your production URL
railway open
```

**Your API will be LIVE in ~2 minutes!** 🎉

---

## What Railway Does Automatically:

1. ✅ Reads `Procfile` → Starts Gunicorn with 4 workers
2. ✅ Reads `requirements.txt` → Installs all dependencies
3. ✅ Creates PostgreSQL database automatically
4. ✅ Sets `DATABASE_URL` environment variable
5. ✅ Generates SSL certificate
6. ✅ Assigns public URL: `https://barberx-api.up.railway.app`
7. ✅ Sets up auto-deploy from GitHub

---

## After Railway Deploy:

### 1. Get Your Production API URL

```
https://barberx-api-production-xxxx.up.railway.app
```

### 2. Set Environment Variables in Railway Dashboard

Click on your service → Variables → Add:

```bash
SECRET_KEY=your-32-char-random-string
FLASK_ENV=production
CORS_ORIGINS=https://barberx.info,https://www.barberx.info
ADMIN_EMAIL=admin@barberx.info
```

### 3. Create Admin Account on Production

```powershell
# SSH into Railway container
railway run python create_admin.py
```

### 4. Update Netlify Site to Use Production API

In your Netlify site's JavaScript:

```javascript
// Change from:
const API_URL = "http://localhost:5000";

// To:
const API_URL = "https://barberx-api-production-xxxx.up.railway.app";
```

---

## Verify Production Deployment

### Health Check:

```powershell
Invoke-RestMethod https://barberx-api-production-xxxx.up.railway.app/health
```

**Expected Response:**

```json
{
  "status": "healthy",
  "timestamp": "2026-01-23T21:30:00.000Z",
  "version": "1.0.0",
  "database": "connected"
}
```

### Test PDF Upload:

```powershell
$apiUrl = "https://barberx-api-production-xxxx.up.railway.app"

Invoke-RestMethod -Uri "$apiUrl/api/upload/pdf" `
  -Method POST `
  -Form @{
    file = Get-Item "test_document.pdf"
    case_number = "2026-TEST-001"
    document_type = "motion"
  }
```

---

## Alternative: Render.com (Automatic with render.yaml)

### One-Click Deploy:

1. **Go to:** https://dashboard.render.com/select-repo
2. **Select:** DTB396/BarberX.info
3. **Click:** "Apply" (Render auto-detects render.yaml)
4. **Wait:** ~3 minutes for build
5. **Done!** Get URL from dashboard

**Render automatically:**

- Creates Flask web service
- Creates PostgreSQL database
- Connects them together
- Sets up SSL
- Configures auto-deploy

---

## Cost Comparison (Monthly)

### Railway

- **Free:** $5 credit/month
- **Pro:** $20/month (unlimited)
- ✅ PostgreSQL included
- ✅ Auto-scaling
- ✅ 512MB RAM minimum

### Render

- **Free:** Limited hours/month
- **Starter:** $25/month
- **Pro:** $85/month
- ✅ PostgreSQL: +$25/month
- ✅ Auto-scaling
- ✅ 512MB RAM

### Recommended for Now:

**Railway** - $5 free credit is enough for testing and initial launch.

---

## Production Checklist

After deployment, verify:

- [ ] Health endpoint returns `{"status": "healthy"}`
- [ ] PostgreSQL database is connected
- [ ] Admin account created
- [ ] CORS allows Netlify domain
- [ ] File uploads work
- [ ] API responds from public URL
- [ ] SSL certificate active (https://)
- [ ] Environment variables set
- [ ] GitHub auto-deploy enabled

---

## Monitoring Your Production App

### Railway Dashboard:

- **Metrics:** CPU, Memory, Network
- **Logs:** Real-time application logs
- **Deployments:** View deploy history
- **Database:** PostgreSQL metrics

### Health Check Monitoring (Free):

```powershell
# Use UptimeRobot.com (free)
# Add monitor: https://barberx-api-production-xxxx.up.railway.app/health
# Get alerts if app goes down
```

---

## Scaling Your Production App

### When you hit limits:

**Railway Free → Pro ($20/month):**

```powershell
railway upgrade
```

**Add more workers:**

```yaml
# Edit Procfile
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 8 --timeout 120
```

**Add Redis cache:**

```powershell
railway add redis
```

**Add background workers:**

```yaml
# Add to Procfile
worker: celery -A app.celery worker --loglevel=info
```

---

## Your Production URLs After Deploy

### API Base URL:

```
https://barberx-api-production-xxxx.up.railway.app
```

### Important Endpoints:

```
GET  /health                        # Health check
POST /api/upload/pdf                # Upload court documents
POST /api/upload                    # Upload BWC videos
POST /api/analyze/{upload_id}       # Start BWC analysis
GET  /api/analysis/{id}/status      # Check analysis status
GET  /api/analysis/{id}/download    # Download report
POST /auth/login                    # Admin login
GET  /admin                         # Admin dashboard
```

---

## Connect Netlify to Production API

### Update form submissions in your Netlify site:

**Before (local):**

```javascript
fetch("http://localhost:5000/api/upload/pdf", {
  method: "POST",
  body: formData,
});
```

**After (production):**

```javascript
const API_URL = "https://barberx-api-production-xxxx.up.railway.app";

fetch(`${API_URL}/api/upload/pdf`, {
  method: "POST",
  body: formData,
});
```

---

## 🎉 Ready to Deploy?

**Run this now:**

```powershell
npm install -g @railway/cli
railway login
railway init
railway up
```

**Then check:**

```
railway open
```

**Your production API will be LIVE!** 🚀

---

## Need Help?

- **Railway Docs:** https://docs.railway.app
- **Render Docs:** https://render.com/docs
- **This Repo:** [DEPLOYMENT-GUIDE.md](DEPLOYMENT-GUIDE.md) (full scaling roadmap)
