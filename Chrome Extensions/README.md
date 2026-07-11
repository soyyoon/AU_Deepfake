# Deepfake Signal

Deepfake Signal is a browser-integrated deepfake detection prototype that connects a Chrome Extension with a local Python inference server.

The system detects visible images and videos in the active browser tab, captures candidate media regions, sends cropped frames to a local backend, performs face-aware artifact analysis, and visualizes the detection result directly in the extension popup and page overlay.

## Demo

![Deepfake Signal demo](assets/demo.gif)

For the full-resolution screen recording, open [assets/result_1.mov](assets/result_1.mov).

GitHub-hosted video preview:

https://github.com/user-attachments/assets/8cd9bce1-564e-4157-aeba-0075a242f9ab

## Why This Project Exists

Synthetic media is becoming easier to create and harder to evaluate in everyday browsing contexts. Many deepfake detectors are presented as offline notebooks, isolated model checkpoints, or dataset-level experiments, but real users encounter suspicious media inside browsers, social platforms, news pages, and video sites.

Deepfake Signal was created to explore how a deepfake detector can move from a standalone model into a usable browser-facing workflow. The goal is not to claim definitive authenticity verification, but to provide a practical risk signal that helps users inspect visible media with face-level evidence.

## Problem Statement

The project focuses on three practical problems:

- Browser media is difficult to inspect because images, videos, thumbnails, and background images appear in many different page layouts.
- Full-frame analysis can produce noisy predictions when the model sees irrelevant background regions instead of face-centered evidence.
- A useful detection tool should explain what it analyzed, where the suspicious region is, and how confident the signal is.

Deepfake Signal addresses these problems by combining visible media discovery, screenshot-based cropping, local API inference, face detection, artifact scoring, and page-level overlay visualization.

## Core Features

- Chrome Extension interface for visible image and video inspection
- Local inference API running on `127.0.0.1`
- Face-aware preprocessing using OpenCV YuNet with Haar fallback
- ConvNeXt-T based artifact detector
- Multi-face detection and risk aggregation
- Page overlay visualization with face bounding boxes
- Frame-level and short sequence analysis endpoints

## Usage

Download the trained checkpoint from Google Drive:

- [best_model.pt](https://drive.google.com/file/d/1hO3YnJJMeAfyYk6bnG5C-KRXvYYLOdLP/view?usp=drivesdk)

Place the downloaded file at:

```text
backend/models/best_model.pt
```

Expected file size and SHA-256 checksum:

```text
334,113,208 bytes
39fd9ca3165bdb78863f3c9d21770594fa54f671242e74f707d8967aca0eb451
```

Create the local Python environment and install dependencies:

```bash
python3.11 -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements.txt
```

Start the local backend server:

```bash
backend/.venv/bin/python backend/server.py
```

Check that the backend is running:

```bash
curl http://127.0.0.1:8000/health
```

Load the Chrome Extension:

1. Open `chrome://extensions`
2. Enable Developer mode
3. Click `Load unpacked`
4. Select the `extension/` directory
5. Open a page containing images or videos
6. Click the Deepfake Signal extension icon and start analysis

Available backend endpoints:

```text
GET  /health
POST /analyze/frame
POST /analyze/sequence
```

## How It Works

1. The user starts analysis from the Chrome Extension popup.
2. The extension scans the active page for visible `video`, `img`, and background image candidates.
3. The selected media region is captured from the visible tab.
4. Cropped frame data is sent to the local backend API.
5. The backend detects faces and crops face-centered regions.
6. The ConvNeXt-T artifact detector estimates the probability of manipulation.
7. The extension displays the result as `Deepfake_Signal`, `Review_Needed`, `Safe_Signal`, or `No_Face`.

## Project Structure

```text
backend/
  server.py          # Local JSON API server
  inference.py       # Model loading, face detection, and inference logic
  requirements.txt   # Python dependencies
  models/            # Face detector and separately downloaded checkpoint

extension/
  manifest.json
  popup.html
  popup.css
  popup.js
  content.js
  service_worker.js
```

## Model

The current backend uses a ConvNeXt-T based artifact detector checkpoint for frame-level deepfake signal estimation.

The detector first localizes faces and performs inference on face-centered crops instead of the full screenshot. This helps reduce false positives from irrelevant background regions.

Current decision thresholds:

```text
P(fake) >= 0.95        -> Deepfake_Signal
0.75 <= P(fake) < 0.95 -> Review_Needed
P(fake) < 0.75         -> Safe_Signal
No detected face       -> No_Face
```

For more detail, see [MODEL_CARD.md](MODEL_CARD.md).

## Development Status

This repository is an actively maintained prototype of a browser-based deepfake detection system. The current focus is to keep the project reproducible, documented, and extensible while the model and extension pipeline continue to improve.

Implemented:

- Local backend inference server
- Chrome Extension popup workflow
- Visible media candidate detection
- Screenshot-based crop analysis
- Face detection and face crop inference
- Overlay visualization

Near-term roadmap:

- Longer video sequence analysis
- Temporal modeling
- AU-based sequence detector
- Score calibration and benchmark evaluation
- Chrome Web Store packaging
- Expanded API documentation and test coverage

## Privacy

The current system is designed to run inference through a local backend server. Captured frames are sent to `127.0.0.1` for analysis and are not intentionally stored by the application.

For the current data handling policy, see [PRIVACY.md](PRIVACY.md).

## Limitations

Deepfake detection is inherently uncertain. This project should be treated as a risk signal system, not as a definitive authenticity verifier.

The current model may be affected by:

- Low-resolution faces
- Heavy compression
- Motion blur
- Occlusion
- Extreme lighting
- Out-of-distribution generation methods
- Non-face media regions

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
