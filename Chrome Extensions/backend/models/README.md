# Model checkpoint

The trained `best_model.pt` checkpoint is not stored in Git because it exceeds GitHub's regular file-size limit.

1. Download [`best_model.pt` from Google Drive](https://drive.google.com/file/d/1hO3YnJJMeAfyYk6bnG5C-KRXvYYLOdLP/view?usp=drivesdk).
2. Place it at `backend/models/best_model.pt` relative to the project root.

Expected file size:

```text
334,113,208 bytes
```

Expected SHA-256:

```text
39fd9ca3165bdb78863f3c9d21770594fa54f671242e74f707d8967aca0eb451
```

On macOS or Linux, verify the downloaded file with:

```bash
shasum -a 256 backend/models/best_model.pt
```
