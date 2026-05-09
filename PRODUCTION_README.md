# Vizera Python Model Service - Production Grade Refactor

A production-ready FastAPI application for Stable Diffusion image generation and editing with:

- ✅ **Singleton Pattern** - Models loaded only once at startup
- ✅ **Automatic Caching** - Respects HuggingFace cache directories
- ✅ **Memory Optimization** - float16, attention slicing, VAE slicing
- ✅ **Thread-Safe** - Lock-based synchronization for concurrent requests
- ✅ **GPU/CPU Auto-Detection** - Automatic fallback to CPU
- ✅ **Comprehensive Logging** - Production-grade logging with clear status messages
- ✅ **Relocatable Cache** - Use `HF_HOME` env var to store models on any disk
- ✅ **Error Handling** - Graceful error handling and recovery
- ✅ **FastAPI Lifespan** - Startup/shutdown event handling

## Architecture

### New Module: `pipeline.py`

Centralized pipeline management with:

- Global pipeline instances (thread-safe singleton)
- Lazy loading on first request OR preloading at startup
- Memory optimization flags
- Comprehensive logging
- Cache detection and status reporting

### Refactored: `generation.py`

- Removed model loading logic
- Uses centralized `get_*_pipeline()` functions
- Enhanced logging for debugging
- Uses config settings for inference parameters

### Enhanced: `main.py`

- Added FastAPI lifespan events (startup/shutdown)
- Optional model preloading on startup
- Debug endpoint for pipeline status
- Better error handling with HTTP exceptions
- Service documentation and versioning

### Enhanced: `config.py`

- New environment variables for customization
- Model ID configuration
- Cache directory configuration (HF_HOME, TRANSFORMERS_CACHE)
- Memory optimization flags
- Inference parameters (steps, guidance_scale)

## Installation & Setup

### 1. Install Dependencies

```bash
cd python-model
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)

Create `.env` file from `.env.example`:

```bash
cp .env.example .env
```

### 3. Custom Cache Directory (if C: is full)

Set `HF_HOME` to a disk with more space:

```bash
# On Windows (PowerShell)
$env:HF_HOME = "D:\huggingface_cache"

# Or add to .env
HF_HOME=D:\huggingface_cache
```

### 4. Run Locally

```bash
# Without preloading (lazy load on first request)
uvicorn app.main:app --host 0.0.0.0 --port 8001

# Or set preload in .env
PRELOAD_MODEL=true
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

## Docker Deployment

### Build & Run

```bash
cd ..  # Root directory
docker-compose up -d python-model --build
```

### With GPU Support (NVIDIA)

```bash
# Requires: NVIDIA Docker Runtime
docker-compose up -d python-model --build
```

## Environment Variables

### Service Configuration

```bash
SERVICE_NAME=vizera-python-model         # Service identifier
OUTPUT_DIR=outputs                       # Output directory for generated images
BASE_URL=http://localhost:8001          # Base URL for image URLs in responses
LOG_LEVEL=INFO                          # Logging level (DEBUG, INFO, WARNING, ERROR)
```

### Model Configuration

```bash
MODEL_ID=runwayml/stable-diffusion-v1-5           # Text-to-image model
INPAINT_MODEL_ID=runwayml/stable-diffusion-v1-5-inpainting  # Inpainting model
```

### Cache Configuration

```bash
# Auto-detected in this order:
# 1. HF_HOME
# 2. TRANSFORMERS_CACHE
# 3. ~/.cache/huggingface (default)

HF_HOME=/data/huggingface_cache         # Custom cache directory
```

### Memory Optimization

```bash
PRELOAD_MODEL=false                     # Preload models at startup
ENABLE_ATTENTION_SLICING=true           # Reduce memory usage (~30%)
ENABLE_VAE_SLICING=true                 # Reduce VAE memory usage
LOW_CPU_MEM_USAGE=true                  # CPU memory optimization
```

### Inference Configuration

```bash
NUM_INFERENCE_STEPS=50                  # Quality vs speed (30-75)
GUIDANCE_SCALE=7.5                      # Prompt adherence (5-15)
```

## API Endpoints

### Health Check

```bash
GET /health
```

### Generate Images

```bash
POST /api/v1/generate

{
  "prompt": "a beautiful sunset over mountains",
  "width": 512,
  "height": 512,
  "seed": null,
  "input_type": "text-to-image"
}
```

### Edit Images

```bash
POST /api/v1/edit

{
  "method": "EDIT_OBJECT_REMOVAL",
  "image": "base64_encoded_image_or_url",
  "prompt": "seamless background",
  "mask": "optional_mask_image"
}
```

### Pipeline Status (Debug)

```bash
GET /debug/pipeline-status

Response:
{
  "text_to_image": {"loaded": true, "initialized": true},
  "image_to_image": {"loaded": false, "initialized": false},
  "inpaint": {"loaded": false, "initialized": false},
  "device": "cuda",
  "model_id": "runwayml/stable-diffusion-v1-5",
  "cache_dir": "/home/user/.cache/huggingface",
  "cuda_available": true,
  "cuda_device": "NVIDIA GeForce RTX 4090"
}
```

## Performance Optimization

### Memory Usage

- **Attention Slicing**: ~30% memory reduction
- **VAE Slicing**: Additional ~10% reduction
- **float16 on GPU**: ~50% memory vs float32

### Speed Optimization

- Preload models at startup to eliminate first-request latency
- Lower `NUM_INFERENCE_STEPS` for faster inference (quality trade-off)
- GPU acceleration significantly faster than CPU

### Cache Optimization

- Models are cached after first download
- Subsequent requests load from cache (not re-downloaded)
- Cache is relocatable (use `HF_HOME` env var)

## Troubleshooting

### "Bus error (core dumped)"

**Solution**: Models too large for container memory

```bash
# Increase container memory limit
# In docker-compose.yml:
mem_limit: 12g    # Increase from 8g
shm_size: 3gb     # Increase from 2gb
```

### Models constantly re-downloading

**Solution**: Cache directory not persisting

```bash
# Check cache location
GET /debug/pipeline-status

# Use docker volume for cache
volumes:
  - huggingface_cache:/root/.cache/huggingface
```

### CUDA out of memory

**Solution**: Reduce inference steps or enable memory optimizations

```bash
ENABLE_ATTENTION_SLICING=true
ENABLE_VAE_SLICING=true
NUM_INFERENCE_STEPS=30  # Lower for faster inference
```

### First request is slow

**Solution**: Preload models at startup

```bash
PRELOAD_MODEL=true
```

## Logging

### Log Format

```
2025-04-30 10:15:23,456 - app.pipeline - INFO - ✓ CUDA available. GPU: NVIDIA GeForce RTX 4090
2025-04-30 10:15:24,123 - app.pipeline - INFO - 🔄 Loading text-to-image model (لأول مرة)...
2025-04-30 10:15:45,789 - app.generation - INFO - Processing generate request: mode=text-to-image, size=512x512
2025-04-30 10:15:50,234 - app.generation - INFO - ✓ Image saved: a1b2c3d4-e5f6-47a8-b9c0-d1e2f3a4b5c6.png (512x512)
```

### Debug Logging

Enable with: `LOG_LEVEL=DEBUG`

## Production Deployment Checklist

- [ ] Set `PRELOAD_MODEL=true` to reduce first-request latency
- [ ] Configure `HF_HOME` to a disk with sufficient space (at least 5GB)
- [ ] Enable GPU support (NVIDIA Docker Runtime)
- [ ] Set memory limits appropriately (`mem_limit`, `shm_size`)
- [ ] Monitor logs for errors: `docker logs vizera-python-model -f`
- [ ] Test with `/health` and `/debug/pipeline-status` endpoints
- [ ] Set `LOG_LEVEL=INFO` in production
- [ ] Configure cache directory backup/persistence
- [ ] Test error handling with `/api/v1/generate` endpoint

## Performance Benchmarks

Typical inference times (RTX 4090, 512x512, 50 steps):

- Text-to-image: ~5-8 seconds
- Image-to-image: ~4-6 seconds
- Inpainting: ~4-6 seconds
- Object removal: ~3-5 seconds

With `NUM_INFERENCE_STEPS=30`:

- Text-to-image: ~3-4 seconds

## Resources

- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [HuggingFace Cache Documentation](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)

## Version History

### v0.3.0 (Production Refactor)

- ✅ Singleton pattern with thread-safe lazy loading
- ✅ Centralized pipeline management (pipeline.py)
- ✅ FastAPI lifespan events for startup/shutdown
- ✅ Environment-based configuration
- ✅ Comprehensive logging
- ✅ Memory optimization flags
- ✅ Cache directory customization
- ✅ Pipeline status debug endpoint

### v0.2.0

- Initial Stable Diffusion integration

### v0.1.0

- Basic PIL-based image effects
