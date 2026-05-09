# 🏗️ Architecture Diagram & Visual Guide

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application (main.py)                │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Startup/Shutdown Events (Lifespan)                     │   │
│  │  - Optional: preload_all_pipelines()                    │   │
│  │  - PRELOAD_MODEL=true → Load models at startup          │   │
│  │  - PRELOAD_MODEL=false → Load on first request (lazy)   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Endpoints                                              │   │
│  │  - GET /health - Health check                           │   │
│  │  - GET /debug/pipeline-status - Debug info              │   │
│  │  - POST /api/v1/generate - Generate images              │   │
│  │  - POST /api/v1/edit - Edit images                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Image Generation Logic (generation.py)                 │   │
│  │  - process_generate()                                   │   │
│  │  - process_edit()                                       │   │
│  │  - Helper functions (_text_to_image, etc.)              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             ↓                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────┐
        │  Pipeline Manager (pipeline.py)         │
        │  ⭐ Core of the Refactoring ⭐          │
        │                                         │
        │  Global Variables (Thread-Safe):        │
        │  - _text_to_image_pipe = None           │
        │  - _image_to_image_pipe = None          │
        │  - _inpaint_pipe = None                 │
        │  - _pipeline_lock = Lock()              │
        │                                         │
        │  Factory Functions:                     │
        │  - get_text_to_image_pipeline()         │
        │  - get_image_to_image_pipeline()        │
        │  - get_inpaint_pipeline()               │
        │  - preload_all_pipelines()              │
        │  - get_pipeline_status()                │
        │                                         │
        │  Features:                              │
        │  ✓ Singleton pattern                    │
        │  ✓ Lazy loading                         │
        │  ✓ Thread-safe locks                    │
        │  ✓ Memory optimization                  │
        │  ✓ Auto GPU/CPU detection               │
        │  ✓ Comprehensive logging                │
        └─────────────────────────────────────────┘
                        ↓
        ┌──────────────────────────────────────┐
        │  HuggingFace Models                  │
        │                                      │
        │  ┌────────────────────────────────┐  │
        │  │ runwayml/sd-v1-5 (4GB)         │  │
        │  │ - Text-to-image                │  │
        │  │ - Image-to-image               │  │
        │  └────────────────────────────────┘  │
        │                                      │
        │  ┌────────────────────────────────┐  │
        │  │ sd-v1-5-inpainting (4GB)       │  │
        │  │ - Object removal               │  │
        │  │ - Inpainting                   │  │
        │  └────────────────────────────────┘  │
        └──────────────────────────────────────┘
                        ↓
        ┌──────────────────────────────────────┐
        │  Cache Directory                     │
        │  (Relocatable via HF_HOME env var)   │
        │                                      │
        │  ~/.cache/huggingface/ (default)     │
        │  or custom location                  │
        │  or /other/disk (if C: is full)      │
        └──────────────────────────────────────┘
                        ↓
        ┌──────────────────────────────────────┐
        │  GPU/CPU                             │
        │  - CUDA (GPU) - float16              │
        │  - CPU - float32                     │
        └──────────────────────────────────────┘
```

## Request Flow

### First Request (Without Preloading)

```
Request arrives at /api/v1/generate
        ↓
    process_generate()
        ↓
    _text_to_image()
        ↓
    get_text_to_image_pipeline()  ← First time!
        ↓
    Check cache with _check_cache_exists()
        ├─ ✓ Model found in cache → Load from disk (~10s)
        └─ ✗ Model not cached → Download (~40s)
        ↓
    Apply memory optimizations:
    - set dtype=float16 (GPU) or float32 (CPU)
    - enable_attention_slicing()
    - enable_vae_slicing()
        ↓
    Store in global _text_to_image_pipe
        ↓
    Use cached pipeline for inference
        ↓
    Return image (5-8s)
```

### Subsequent Requests (With Caching)

```
Request arrives at /api/v1/generate
        ↓
    process_generate()
        ↓
    _text_to_image()
        ↓
    get_text_to_image_pipeline()
        ↓
    Check: Is _text_to_image_pipe already loaded?
        ├─ ✓ YES → Return immediately (no re-load)
        └─ (No download, no re-initialization)
        ↓
    Use cached pipeline for inference
        ↓
    Return image (5-8s)
```

### With Preloading (PRELOAD_MODEL=true)

```
Application startup
        ↓
    lifespan() context manager enters
        ↓
    Check PRELOAD_MODEL setting
        ├─ ✓ true → Call preload_all_pipelines()
        └─ false → Skip (lazy load on first request)
        ↓
    preload_all_pipelines():
    - get_text_to_image_pipeline()  → Download & cache (~60s total)
    - get_image_to_image_pipeline() → Download & cache
    - get_inpaint_pipeline()         → Download & cache
        ↓
    Ready to accept requests!
        ↓
    First request arrives
        ↓
    Pipeline already loaded → Use immediately (5-8s)
```

## Configuration Hierarchy

```
┌────────────────────────────────────────┐
│  Environment Variables                  │
│  (Highest Priority)                    │
└────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────┐
│  .env file                              │
│  (Second Priority)                     │
└────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────┐
│  Default Values in config.py            │
│  (Lowest Priority)                     │
└────────────────────────────────────────┘
```

## Memory Optimization Strategy

```
Normal Setup
─────────────
GPU Memory: 8GB (full float32)
CPU Memory: Heavy

With Optimizations
──────────────────
GPU Memory: 4GB (float16 + slicing)
CPU Memory: Optimized

Optimization Techniques:
1. torch_dtype=torch.float16 (GPU)
   └─ 50% memory reduction

2. enable_attention_slicing()
   └─ ~30% memory reduction

3. enable_vae_slicing()
   └─ ~10% additional reduction

4. low_cpu_mem_usage=True
   └─ CPU memory optimization

Total Reduction: ~60-70%
```

## Thread Safety

```
Request 1                Request 2
    ↓                        ↓
get_pipeline()          get_pipeline()
    ↓                        ↓
Try to acquire _pipeline_lock
    ↓                        ↓
ACQUIRED ✓              WAITING (blocked)
    ↓
Check if _pipeline_pipe is None?
    ↓ YES
Load pipeline...
    ↓
Store in global
    ↓
RELEASE lock
    ↓                        ↓
Return cached pipe      ACQUIRED ✓
                            ↓
                        Check if _pipeline_pipe is None?
                            ↓ NO (already loaded!)
                        Return cached pipe immediately
                            ↓
                        RELEASE lock
```

## File Structure

```
python-model/
│
├── app/
│   ├── __init__.py                    # Package init
│   ├── main.py                        # ⭐ FastAPI app (enhanced)
│   ├── pipeline.py                    # ⭐ NEW: Pipeline manager
│   ├── generation.py                  # ⭐ Refactored: Uses pipeline
│   ├── config.py                      # ⭐ Enhanced: Config management
│   └── models.py                      # Pydantic models
│
├── outputs/                           # Generated images stored here
│   └── (generated PNGs)
│
├── requirements.txt                   # Dependencies (updated)
├── .env                               # Local configuration
├── .env.example                       # Configuration template
│
├── README.md                          # Original README
├── README_REFACTORING.md              # ⭐ NEW: Executive summary
├── QUICKSTART.md                      # ⭐ NEW: 5-minute setup
├── PRODUCTION_README.md               # ⭐ NEW: Complete reference
├── REFACTORING_SUMMARY.md             # ⭐ NEW: Technical details
└── ARCHITECTURE.md                    # ⭐ NEW: This file

Key: ⭐ = New or significantly changed
```

## Configuration Flow

```
User starts service
        ↓
Read environment variables (or .env file)
        ↓
settings = Settings()  [in config.py]
        ↓
Load configured values:
- HF_HOME (cache directory)
- PRELOAD_MODEL (load at startup?)
- ENABLE_ATTENTION_SLICING
- ENABLE_VAE_SLICING
- NUM_INFERENCE_STEPS
- GUIDANCE_SCALE
- LOG_LEVEL
        ↓
Pass settings to pipeline manager
        ↓
Pipeline manager uses settings for loading models
        ↓
Models loaded with optimizations
        ↓
Ready to generate images!
```

## Performance Comparison

```
Scenario 1: Development (No Preload)
─────────────────────────────────────
Startup: 1s
Request 1: 50s (download + init)
Request 2: 8s (cached)
Request 3: 8s (cached)
Total: 67s for 3 requests

Scenario 2: Production (With Preload)
──────────────────────────────────────
Startup: 60s (preload all)
Request 1: 8s (already loaded)
Request 2: 8s (cached)
Request 3: 8s (cached)
Total: 84s for startup + 3 requests
... but first request is FAST!

Better UX in production! ✓
```

## Logging Flow

```
Application Start
      ↓
"Starting Vizera Python Model Service..."
      ↓
Check PRELOAD_MODEL setting
      ├─ true:  "Preloading all pipelines..."
      └─ false: "Model preloading disabled. Lazy loading."
      ↓
Startup complete
      ↓
Request arrives: POST /api/v1/generate
      ├─ "Processing generate request: mode=text-to-image"
      │
      ├─ First time?
      │  └─ "🔄 Loading text-to-image model (لأول مرة)..."
      │     "✓ Using cached model" or "Model not cached, will download"
      │     "✓ Text-to-image pipeline loaded successfully"
      │
      ├─ Inference
      │  └─ "Generating image: prompt='...', size=512x512"
      │
      └─ Save
         └─ "✓ Image saved: abc123.png (512x512)"
      ↓
Response sent to client
```

## Error Handling Strategy

```
Error occurs during:

1. Pipeline Loading
   ├─ Log: "✗ Failed to load [pipeline]: [error]"
   ├─ Suggest: Check cache, disk space, GPU
   └─ Response: HTTP 500 with error message

2. Model Download (first time)
   ├─ Log: "Model not cached, will download"
   ├─ Show: Progress implicitly through logs
   └─ Timeout: 20min by default

3. Inference
   ├─ Log: "Error in process_generate: [error]"
   ├─ Log: exc_info=True (full traceback)
   └─ Response: HTTP 500 with error message

4. Out of Memory
   ├─ Log: "CUDA out of memory"
   ├─ Suggest: Enable attention slicing, reduce steps
   └─ Response: HTTP 500 with clear message
```

## Cache Directory Detection

```
HF_HOME environment variable?
      ├─ YES → Use HF_HOME
      └─ NO → Next check
           ↓
TRANSFORMERS_CACHE environment variable?
      ├─ YES → Use TRANSFORMERS_CACHE
      └─ NO → Next check
           ↓
Default: ~/.cache/huggingface
      ↓
Create if doesn't exist
      ↓
Download models to this location
      ↓
Subsequent requests load from cache
```

---

**This architecture ensures:**

- ✅ Efficient resource usage
- ✅ Fast inference
- ✅ No re-downloading
- ✅ Thread-safe operations
- ✅ Clear error messages
- ✅ Production reliability
