# 🚀 Quick Start Guide - Vizera Python Model Service

## ⚡ 5-Minute Setup

### Option 1: Local Development (Recommended for Testing)

```bash
# 1. Navigate to python-model directory
cd python-model

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the service
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# 4. Test it works
curl http://localhost:8001/health
```

**Output:**

```json
{ "status": "ok", "service": "vizera-python-model", "version": "0.3.0" }
```

### Option 2: Docker (Recommended for Production)

```bash
# 1. Build and run
docker-compose up -d python-model --build

# 2. Check logs
docker logs -f vizera-python-model

# 3. Test it works
curl http://localhost:8001/health
```

## ⚙️ Configuration

### Fast Setup (No Configuration)

```bash
# Just run - uses default settings
uvicorn app.main:app --port 8001
```

### Production Setup (With Preloading)

```bash
# Set environment variable before running
export PRELOAD_MODEL=true
export LOG_LEVEL=INFO

# Then run
uvicorn app.main:app --port 8001
```

### Custom Cache Directory (C: Drive Full?)

```bash
# Point to disk with more space
export HF_HOME=D:\huggingface_cache

# Then run
uvicorn app.main:app --port 8001
```

## 📝 Create `.env` File

```bash
# Copy the example
cp .env.example .env

# Edit .env with your settings:
# PRELOAD_MODEL=true
# HF_HOME=/data/huggingface_cache
# LOG_LEVEL=INFO
```

## 🧪 Test the API

### 1. Health Check

```bash
curl http://localhost:8001/health
```

### 2. Check Pipeline Status

```bash
curl http://localhost:8001/debug/pipeline-status
```

### 3. Generate Image from Text

```bash
curl -X POST http://localhost:8001/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful sunset over mountains",
    "width": 512,
    "height": 512
  }'
```

**Response:**

```json
{
  "job_id": "a1b2c3d4-e5f6-47a8-b9c0-d1e2f3a4b5c6",
  "status": "completed",
  "image_url": "http://localhost:8001/outputs/a1b2c3d4-e5f6-47a8-b9c0-d1e2f3a4b5c6.png",
  "width": 512,
  "height": 512
}
```

## 📊 First Run Expectations

### Without Preloading (PRELOAD_MODEL=false)

- ✅ Service starts instantly (~1 second)
- ❌ First request slow (~45-60 seconds) - model downloads and loads
- ✅ Subsequent requests fast (~5-8 seconds)

### With Preloading (PRELOAD_MODEL=true)

- ⏳ Service starts slower (~45-60 seconds) - loads models upfront
- ✅ First request fast (~5-8 seconds)
- ✅ Subsequent requests fast (~5-8 seconds)

**Recommendation:** Use `PRELOAD_MODEL=true` in production!

## 🔧 Troubleshooting

### Issue: "Bus error (core dumped)"

```bash
# Solution: Increase Docker memory limits
# Edit docker-compose.yml:
mem_limit: 12g    # Increase from 8g
shm_size: 3gb     # Increase from 2gb

# Then restart
docker-compose up -d python-model --build
```

### Issue: Models keep re-downloading

```bash
# Solution: Check cache directory
curl http://localhost:8001/debug/pipeline-status

# If cache_dir is `/tmp` or temporary location:
export HF_HOME=/data/huggingface_cache
```

### Issue: CUDA out of memory

```bash
# Solution: Reduce inference steps
export NUM_INFERENCE_STEPS=30

# Or enable memory optimizations (already default)
export ENABLE_ATTENTION_SLICING=true
export ENABLE_VAE_SLICING=true
```

### Issue: First request is slow

```bash
# Solution: Preload models at startup
export PRELOAD_MODEL=true
```

## 📊 View Logs

### Local Development

```bash
# Logs appear in terminal where you ran uvicorn
```

### Docker

```bash
# View live logs
docker logs -f vizera-python-model

# View last 100 lines
docker logs --tail 100 vizera-python-model

# Export logs to file
docker logs vizera-python-model > logs.txt 2>&1
```

## 🔍 Debug Checklist

```bash
# 1. Check service is running
curl http://localhost:8001/health

# 2. Check pipeline status
curl http://localhost:8001/debug/pipeline-status

# 3. Check logs for errors
docker logs vizera-python-model | grep ERROR

# 4. Check disk space (for cache)
df -h /data  # or wherever HF_HOME points

# 5. Check GPU availability
docker exec vizera-python-model nvidia-smi

# 6. Test simple generation
curl -X POST http://localhost:8001/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"test","width":256,"height":256}'
```

## 📚 Next Steps

- Read [PRODUCTION_README.md](./PRODUCTION_README.md) for complete documentation
- Read [REFACTORING_SUMMARY.md](./REFACTORING_SUMMARY.md) for architecture details
- Check [.env.example](./.env.example) for all configuration options

## 🆘 Getting Help

### Check Logs First

```bash
# Local
# (Errors will appear in terminal)

# Docker
docker logs vizera-python-model 2>&1 | tail -50
```

### Enable Debug Logging

```bash
export LOG_LEVEL=DEBUG
uvicorn app.main:app --port 8001
```

### Common Log Messages

**✓ Good:**

```
✓ CUDA available. GPU: NVIDIA GeForce RTX 4090
✓ Using cached model: runwayml/stable-diffusion-v1-5
✓ Text-to-image pipeline loaded successfully
✓ Image saved: a1b2c3d4-e5f6-47a8-b9c0-d1e2f3a4b5c6.png (512x512)
```

**⚠️ Warning (but recoverable):**

```
⚠ CUDA available but not working: ... Falling back to CPU.
ℹ CUDA not available. Using CPU (inference will be slower)
ℹ Model not cached, will download: runwayml/stable-diffusion-v1-5
```

**✗ Error (needs action):**

```
✗ Failed to load text-to-image pipeline: CUDA out of memory
✗ Failed to preload pipelines: RuntimeError: ...
```

---

**Quick Summary:**

1. Install: `pip install -r requirements.txt`
2. Run: `uvicorn app.main:app --port 8001`
3. Test: `curl http://localhost:8001/health`
4. Check status: `curl http://localhost:8001/debug/pipeline-status`
5. Generate: `POST /api/v1/generate` with prompt

For more details, see [PRODUCTION_README.md](./PRODUCTION_README.md) 📖
