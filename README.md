# Python Model Service (Local Vision)

`python-model` is a local FastAPI service used by `worker` as an internal image engine.

## Supported operations
- Text to image (`/api/v1/generate`)
- Image to image / style transfer (`/api/v1/generate`)
- Line drawing to image (`/api/v1/generate`)
- Upscaling (`/api/v1/generate` and `/api/v1/edit`)
- Object removal (`/api/v1/edit`)
- Fill inpaint (`/api/v1/edit`)
- Fill extend / outpaint (`/api/v1/edit`)

All outputs are written to local `outputs/` and served via `/outputs/{file}`.

## Run locally
```bash
cd python-model
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

## Quick examples
Generate:
```bash
curl -X POST http://localhost:8001/api/v1/generate ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"modern tropical house\", \"width\":1024, \"height\":768}"
```

Edit (style transfer):
```bash
curl -X POST http://localhost:8001/api/v1/edit ^
  -H "Content-Type: application/json" ^
  -d "{\"method\":\"EDIT_STYLE_TRANSFER\",\"image\":\"data:image/png;base64,<...>\",\"style_prompt\":\"warm sunset cinematic\"}"
```
# python-model
