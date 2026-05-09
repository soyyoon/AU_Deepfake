# Deepfake Guard — Server

Two-stage deepfake detection pipeline served via FastAPI.

## Architecture

```
image → MTCNN face crop → Stage1 (Xception artifact) → P(fake)
                                ↓ (if 0.4 ≤ p ≤ 0.6)
                          Stage2 (MediaPipe blendshapes → MLP) → P(fake)
                                ↓
                          weighted average → final label
```

## Endpoints

- `GET  /api/v1/health` — liveness check
- `POST /api/v1/detect` — body: `{"image_url": "..."}` or `{"image_b64": "..."}`
  - optional `"return_debug": true` for per-stage probs and timings

Response:
```json
{
  "label": "fake",
  "confidence": 0.74,
  "fake_prob": 0.87,
  "stage_used": "stage1+stage2",
  "debug": null
}
```

## Run locally (dummy mode)

The pipeline runs in **dummy mode** when model weights aren't found. This
lets you wire up the Chrome extension and test the API contract before
training is done.

```bash
cd server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then:
```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/face.jpg", "return_debug": true}'
```

## Run with Docker

```bash
docker compose up --build
```

## Drop in trained weights

Once Stage 1 is trained, save it as:
```python
torch.save(model.state_dict(), "server/weights/artifact_xception.pt")
```
And Stage 2:
```python
torch.save(au_mlp.state_dict(), "server/weights/au_mlp.pt")
```
Restart the server — `dummy_mode` flag will flip off automatically.

## Configuration

All knobs live in `app/config.py` and can be overridden via env vars:
- `STAGE1_UNCERTAINTY_LOW` / `STAGE1_UNCERTAINTY_HIGH` — when to invoke Stage 2
- `STAGE1_WEIGHT` / `STAGE2_WEIGHT` — combination weights
- `DECISION_THRESHOLD` — 0.5 by default
- `DEVICE` — `cuda` or `cpu`
- `REDIS_URL`

## Tests

```bash
cd server && pytest -v
```
