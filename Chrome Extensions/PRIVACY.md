# Privacy Policy

This document describes the privacy behavior of the current Deepfake Signal prototype.

Deepfake Signal is currently designed as a local browser-extension prototype. The Chrome Extension communicates with a Python backend server running on the user's own machine.

## Summary

- The extension analyzes visible image and video regions only after the user starts analysis.
- Captured frame crops are sent to a local backend API at `127.0.0.1`.
- The current prototype does not intentionally store screenshots, video frames, face crops, or prediction requests.
- The current prototype does not intentionally upload captured media to a remote server.
- The current prototype is not yet distributed through the Chrome Web Store.

## Data Processed

When the user starts analysis, the extension may process:

- Visible image regions on the active browser tab
- Visible video frame regions on the active browser tab
- Candidate media metadata such as bounding box size and media type
- Cropped frame data encoded as a base64 image data URL
- Optional page URL metadata if included in the request payload

The backend uses this data to detect faces, crop face-centered regions, run model inference, and return a risk score and label.

## Local Backend Communication

The extension sends analysis requests to:

```text
http://127.0.0.1:8000
```

The current backend endpoints are:

```text
GET  /health
POST /analyze/frame
POST /analyze/sequence
```

This means the current prototype is intended to run locally on the user's machine.

## Storage

The current implementation does not intentionally save:

- Original screenshots
- Uploaded frame crops
- Detected face crops
- Full video frames
- Prediction history
- User browsing history

Runtime logs may still appear in the local terminal while the backend server is running. Developers should avoid adding logs that include raw image data, sensitive page URLs, or personally identifying information.

## Network Transmission

The current prototype does not intentionally send captured media to external APIs or remote servers.

If future versions introduce cloud inference, telemetry, analytics, crash reporting, or remote model services, this document should be updated before release.

## Chrome Extension Permissions

The current extension uses Chrome permissions to inspect and capture visible media from the active tab.

The prototype currently requests:

```text
activeTab
scripting
```

It also allows communication with the local backend:

```text
http://127.0.0.1:8000/*
http://localhost:8000/*
```

These permissions should be reviewed and minimized before public distribution.

## User Control

Analysis is initiated by the user through the Chrome Extension interface.

The prototype is not intended to continuously monitor browsing activity in the background.

## Limitations

This privacy policy describes the current local prototype. It does not yet represent a final production release or Chrome Web Store listing.

Before public distribution, the project should add:

- A final Chrome Web Store privacy disclosure
- A clear explanation of all extension permissions
- A data retention policy
- A policy for any future telemetry or analytics
- A policy for cloud inference, if added

## Contact

Project maintainer contact information will be added before public release.
