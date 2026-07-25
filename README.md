# PDF → Hindi Audiobook Converter (Streamlit)

Converts uploaded book PDFs into Hindi audiobooks (.mp3), with automatic
language detection, chunked translation, chunked TTS, retry/backoff error
handling, and a downloadable MP3 (+ optional segments ZIP).

## Files
- `app.py` — the complete application (single file)
- `requirements.txt` — Python dependencies
- `packages.txt` — system dependency (`ffmpeg`, required by `pydub`) for
  Streamlit Cloud / Hugging Face Spaces

## 1. Run locally (Windows / Mac / Linux)

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 2. Install ffmpeg (required by pydub for MP3 encoding/decoding)
# Windows: choco install ffmpeg   (or download from ffmpeg.org and add to PATH)
# Mac:     brew install ffmpeg
# Linux:   sudo apt-get install ffmpeg

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Allow larger uploads (optional)
Create `.streamlit/config.toml` in the project folder:
```toml
[server]
maxUploadSize = 200
```

## 2. Deploy for FREE — Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `packages.txt` to a public GitHub repo.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **"New app"**, select your repo/branch, and set the main file to `app.py`.
4. Click **Deploy**. Streamlit Cloud automatically installs `requirements.txt`
   and the `ffmpeg` system package from `packages.txt`.
5. You'll get a public URL like `https://your-app-name.streamlit.app`
   that works on desktop and mobile browsers.

## 3. Deploy for FREE — Hugging Face Spaces

1. Go to https://huggingface.co/new-space
2. Choose **Streamlit** as the Space SDK.
3. Upload `app.py`, `requirements.txt`, and `packages.txt` (Spaces also reads
   `packages.txt` for apt system dependencies).
4. The Space builds automatically and gives you a public URL.

## Notes & Limitations
- Translation and TTS depend on external network calls (Google Translate /
  Google TTS). Very large books (300+ pages) should be converted in
  page-range batches (e.g., 1–50, 51–100, ...) to stay within free hosting
  timeout/memory limits — the sidebar's page-range control is built for this.
- Scanned/image-only PDFs have no extractable text layer; the app will
  detect this and prompt you to run OCR first (e.g., `ocrmypdf`, Adobe
  Acrobat) before uploading.
- Playback speed (1.25x/1.5x/2x) is applied via audio time-stretching after
  synthesis using `pydub`.
