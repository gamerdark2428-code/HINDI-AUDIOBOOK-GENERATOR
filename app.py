# -*- coding: utf-8 -*-
"""
PDF → Hindi Audiobook Converter
================================
A production-ready Streamlit application that converts uploaded book PDFs
into Hindi audiobooks (MP3), with automatic language detection, chunked
translation, chunked neural/standard TTS, and full error resilience.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Author: Principal Full-Stack & Audio AI Engineer (generated)
"""

import io
import os
import re
import time
import zipfile
import tempfile
import traceback
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import streamlit as st
import pdfplumber
from deep_translator import GoogleTranslator
from gtts import gTTS
from pydub import AudioSegment
from pydub.effects import speedup

try:
    from langdetect import detect as lang_detect
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False


# =============================================================================
# PAGE CONFIG & GLOBAL STYLING
# =============================================================================

st.set_page_config(
    page_title="Hindi Audiobook Converter",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main { background-color: #0e1117; }
    .stApp { background: linear-gradient(180deg, #0e1117 0%, #14171f 100%); }
    h1, h2, h3 { color: #f5f5f7 !important; }
    .stButton>button {
        background: linear-gradient(90deg,#6d28d9,#9333ea);
        color: white; border: none; border-radius: 10px;
        padding: 0.6em 1.2em; font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(147,51,234,0.35); }
    .metric-card {
        background: #1a1d29; border: 1px solid #2a2e3f; border-radius: 14px;
        padding: 18px; text-align:center;
    }
    .status-box {
        background: #171a25; border-left: 4px solid #9333ea;
        padding: 10px 16px; border-radius: 8px; margin-bottom: 6px;
        font-family: monospace; font-size: 0.85em; color: #d1d5db;
    }
    .chapter-badge {
        display:inline-block; background:#2a1f3d; color:#c4b5fd;
        padding: 3px 10px; border-radius: 20px; font-size: 0.75em; margin-right:6px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

DEFAULT_STATE = {
    "processing": False,
    "final_audio_bytes": None,
    "segment_zip_bytes": None,
    "log_lines": [],
    "pdf_meta": None,
    "hindi_preview": "",
    "error": None,
    "done": False,
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_result_state():
    """Clear previous run results/memory before starting a new job."""
    st.session_state["final_audio_bytes"] = None
    st.session_state["segment_zip_bytes"] = None
    st.session_state["log_lines"] = []
    st.session_state["hindi_preview"] = ""
    st.session_state["error"] = None
    st.session_state["done"] = False


def log(msg: str):
    """Append a timestamped log line to session state (bounded length)."""
    st.session_state["log_lines"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    # keep log bounded to avoid memory growth on very large books
    if len(st.session_state["log_lines"]) > 400:
        st.session_state["log_lines"] = st.session_state["log_lines"][-400:]


# =============================================================================
# PDF EXTRACTION & TEXT CLEANING
# =============================================================================

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MULTI_WS_RE = re.compile(r"[ \t]+")
MULTI_NL_RE = re.compile(r"\n{2,}")
PAGE_NUM_LINE_RE = re.compile(r"^\s*\d{1,4}\s*$")
HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")


def safe_extract_pdf(file_obj, start_page: int, end_page: int) -> Tuple[List[str], int]:
    """
    Extract raw text per page from a PDF using pdfplumber.
    Returns (list_of_page_texts, total_pages_in_document).
    Raises a user-friendly RuntimeError on corrupt/unreadable files.
    """
    try:
        with pdfplumber.open(file_obj) as pdf:
            total_pages = len(pdf.pages)
            if total_pages == 0:
                raise RuntimeError("The PDF appears to contain zero pages.")

            start_page = max(1, min(start_page, total_pages))
            end_page = max(start_page, min(end_page, total_pages))

            page_texts = []
            for i in range(start_page - 1, end_page):
                try:
                    page = pdf.pages[i]
                    text = page.extract_text() or ""
                except Exception:
                    # A single corrupt page shouldn't kill the whole job
                    text = ""
                page_texts.append(text)
            return page_texts, total_pages
    except Exception as e:
        raise RuntimeError(
            f"Could not read this PDF. It may be corrupted, password-protected, "
            f"or not a valid PDF file. Details: {e}"
        )


def strip_repeating_headers_footers(page_texts: List[str]) -> List[str]:
    """
    Detects lines (typically first/last line of each page) that repeat across
    a large fraction of pages -- these are almost always running headers,
    footers, or book titles -- and removes them from every page.
    """
    if len(page_texts) < 3:
        return page_texts

    first_lines, last_lines = {}, {}
    for txt in page_texts:
        lines = [l.strip() for l in txt.split("\n") if l.strip()]
        if not lines:
            continue
        first_lines[lines[0]] = first_lines.get(lines[0], 0) + 1
        last_lines[lines[-1]] = last_lines.get(lines[-1], 0) + 1

    n = len(page_texts)
    threshold = max(3, int(n * 0.35))
    noise_lines = set()
    for line, count in first_lines.items():
        if count >= threshold and len(line) < 90:
            noise_lines.add(line)
    for line, count in last_lines.items():
        if count >= threshold and len(line) < 90:
            noise_lines.add(line)

    cleaned = []
    for txt in page_texts:
        lines = txt.split("\n")
        lines = [l for l in lines if l.strip() not in noise_lines]
        cleaned.append("\n".join(lines))
    return cleaned


def clean_page_text(text: str) -> str:
    """Normalize a single page's raw text: fix hyphenation, strip noise, collapse whitespace."""
    if not text:
        return ""
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)          # fix line-wrap hyphenation
    text = URL_RE.sub("", text)                         # strip raw URLs
    lines = text.split("\n")
    lines = [l for l in lines if not PAGE_NUM_LINE_RE.match(l)]  # strip bare page numbers
    text = "\n".join(lines)
    text = text.replace("\n", " ")
    text = MULTI_WS_RE.sub(" ", text)
    text = MULTI_NL_RE.sub("\n", text)
    return text.strip()


def build_clean_fulltext(page_texts: List[str]) -> str:
    page_texts = strip_repeating_headers_footers(page_texts)
    cleaned_pages = [clean_page_text(p) for p in page_texts]
    return "\n\n".join([p for p in cleaned_pages if p])


# =============================================================================
# LANGUAGE DETECTION
# =============================================================================

def detect_language_safe(sample_text: str) -> str:
    """Best-effort language detection with a safe English fallback."""
    sample = sample_text.strip()[:800]
    if not sample:
        return "unknown"
    if not LANGDETECT_AVAILABLE:
        return "unknown"
    try:
        return lang_detect(sample)
    except Exception:
        return "unknown"


# =============================================================================
# CHUNKING (sentence-boundary aware, TTS/translation-safe sizes)
# =============================================================================

SENTENCE_SPLIT_RE = re.compile(r"(?<=[।\.\?\!])\s+")


def chunk_text(text: str, max_chars: int = 450) -> List[str]:
    """
    Split text into chunks no larger than max_chars, breaking only on
    sentence boundaries (Hindi danda '।', period, '?', '!') so that no
    sentence is ever cut mid-way -- this prevents garbled/truncated audio.
    """
    if not text:
        return []

    sentences = SENTENCE_SPLIT_RE.split(text)
    chunks, current = [], ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # If a single "sentence" is itself too long, hard-split it as a fallback
        if len(sentence) > max_chars:
            for i in range(0, len(sentence), max_chars):
                piece = sentence[i:i + max_chars]
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(piece)
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


# =============================================================================
# TRANSLATION (with exponential backoff retry)
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_translator():
    return GoogleTranslator(source="auto", target="hi")


def translate_chunk_with_retry(chunk: str, max_retries: int = 3) -> str:
    """
    Translate a text chunk to Hindi with exponential backoff on failure.
    On total failure, returns the original chunk with a marker rather than
    crashing the whole job.
    """
    translator = get_translator()
    last_err = None
    for attempt in range(max_retries):
        try:
            result = translator.translate(chunk)
            if result and result.strip():
                return result.strip()
            raise ValueError("Empty translation result")
        except Exception as e:
            last_err = e
            wait = (2 ** attempt) + 0.5
            time.sleep(wait)
    log(f"⚠️ Translation failed after {max_retries} retries for a chunk: {last_err}")
    return chunk  # graceful degradation: keep original text rather than losing content


# =============================================================================
# TEXT-TO-SPEECH (chunked, retry-safe)
# =============================================================================

def sanitize_devanagari(text: str) -> str:
    """Sanitize Hindi text before sending to the TTS engine."""
    if not text:
        return ""
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = MULTI_WS_RE.sub(" ", text)
    text = text.strip()
    # gTTS chokes on totally empty or punctuation-only strings
    if not re.search(r"[\u0900-\u097F A-Za-z0-9]", text):
        return ""
    return text


def synthesize_chunk_with_retry(
    hindi_text: str, tmp_dir: str, idx: int, max_retries: int = 3
) -> Optional[AudioSegment]:
    """
    Convert a single Hindi text chunk into an AudioSegment via gTTS,
    with retries and graceful failure (returns None on total failure so
    the pipeline can skip it instead of crashing).
    """
    hindi_text = sanitize_devanagari(hindi_text)
    if not hindi_text:
        return None

    tmp_path = os.path.join(tmp_dir, f"chunk_{idx:05d}.mp3")
    last_err = None
    for attempt in range(max_retries):
        try:
            tts = gTTS(text=hindi_text, lang="hi", slow=False)
            tts.save(tmp_path)
            segment = AudioSegment.from_mp3(tmp_path)
            return segment
        except Exception as e:
            last_err = e
            wait = (2 ** attempt) + 0.5
            time.sleep(wait)
        finally:
            # remove the temp mp3 immediately after loading into memory
            if os.path.exists(tmp_path) and attempt == max_retries - 1:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    log(f"⚠️ TTS failed after {max_retries} retries for chunk #{idx}: {last_err}")
    return None


# =============================================================================
# MAIN PIPELINE
# =============================================================================

@dataclass
class PipelineResult:
    final_audio: Optional[bytes] = None
    segment_zip: Optional[bytes] = None
    hindi_preview: str = ""
    total_chunks: int = 0
    failed_chunks: int = 0


def run_pipeline(
    file_obj,
    start_page: int,
    end_page: int,
    mode: str,
    playback_speed: float,
    progress_bar,
    status_text,
    make_zip_segments: bool,
) -> PipelineResult:
    """
    Full end-to-end pipeline: extract -> clean -> (translate) -> TTS -> stitch.
    Progress callbacks update the Streamlit UI in real time.
    """
    # ---- STAGE 1: EXTRACTION ----
    status_text.markdown("📄 **Stage 1/4:** Extracting text from PDF...")
    log("Starting PDF extraction")
    page_texts, total_pages = safe_extract_pdf(file_obj, start_page, end_page)
    progress_bar.progress(0.10)

    non_empty_pages = sum(1 for p in page_texts if p.strip())
    if non_empty_pages == 0:
        raise RuntimeError(
            "No extractable text was found in the selected page range. "
            "This PDF is likely a scanned image without an OCR text layer. "
            "Please run OCR (e.g. Adobe Acrobat, ocrmypdf) before uploading."
        )
    if non_empty_pages < len(page_texts) * 0.5:
        log("⚠️ Warning: more than half of selected pages had no extractable text (possible scanned pages).")

    fulltext = build_clean_fulltext(page_texts)
    if not fulltext.strip():
        raise RuntimeError("Text extraction produced no usable content after cleaning.")

    log(f"Extracted and cleaned {len(fulltext)} characters from {len(page_texts)} pages")
    progress_bar.progress(0.18)

    # ---- STAGE 2: LANGUAGE DETECTION & CHUNKING ----
    status_text.markdown("🔤 **Stage 2/4:** Detecting language and preparing text chunks...")
    detected_lang = detect_language_safe(fulltext)
    log(f"Detected source language: {detected_lang}")

    needs_translation = True
    if mode == "Direct Hindi PDF":
        needs_translation = False
    elif mode == "Auto-detect":
        needs_translation = detected_lang != "hi"
    # mode == "English/Other → Hindi" always translates

    source_chunks = chunk_text(fulltext, max_chars=450)
    total_chunks = len(source_chunks)
    if total_chunks == 0:
        raise RuntimeError("No valid text chunks could be generated from the extracted text.")
    log(f"Split content into {total_chunks} chunks (needs_translation={needs_translation})")
    progress_bar.progress(0.22)

    # ---- STAGE 3: TRANSLATION + TTS (interleaved, chunk by chunk) ----
    hindi_chunks: List[str] = []
    audio_segments: List[AudioSegment] = []
    failed_chunks = 0

    tmp_dir = tempfile.mkdtemp(prefix="hindi_audiobook_")
    try:
        for i, chunk in enumerate(source_chunks):
            pct_stage = 0.22 + (0.68 * (i + 1) / total_chunks)

            if needs_translation:
                status_text.markdown(
                    f"🌐 **Stage 3/4:** Translating to Hindi — chunk {i+1}/{total_chunks}"
                )
                hindi_text = translate_chunk_with_retry(chunk)
            else:
                status_text.markdown(
                    f"🌐 **Stage 3/4:** Preparing Hindi text — chunk {i+1}/{total_chunks}"
                )
                hindi_text = chunk

            hindi_chunks.append(hindi_text)

            status_text.markdown(
                f"🔊 **Stage 3/4:** Synthesizing Hindi audio — chunk {i+1}/{total_chunks}"
            )
            segment = synthesize_chunk_with_retry(hindi_text, tmp_dir, i)
            if segment is not None:
                audio_segments.append(segment)
            else:
                failed_chunks += 1

            progress_bar.progress(min(pct_stage, 0.90))

        if not audio_segments:
            raise RuntimeError(
                "Audio synthesis failed for all chunks. This is usually caused by a "
                "temporary network issue with the translation/TTS service — please try again."
            )

        # ---- STAGE 4: STITCHING ----
        status_text.markdown("🧵 **Stage 4/4:** Stitching audio chunks into final audiobook...")
        log("Stitching audio segments")

        silence = AudioSegment.silent(duration=180)
        final_audio = AudioSegment.empty()
        segment_files_for_zip = []

        for idx, seg in enumerate(audio_segments):
            final_audio += seg + silence
            if make_zip_segments:
                segment_files_for_zip.append((idx, seg))

        if abs(playback_speed - 1.0) > 0.01:
            try:
                final_audio = speedup(final_audio, playback_speed=playback_speed)
                log(f"Applied playback speed adjustment: {playback_speed}x")
            except Exception as e:
                log(f"⚠️ Speed adjustment failed, using normal speed: {e}")

        progress_bar.progress(0.96)

        # Export final master MP3 to memory
        final_buf = io.BytesIO()
        final_audio.export(final_buf, format="mp3", bitrate="128k")
        final_bytes = final_buf.getvalue()
        final_buf.close()

        # Optionally build a ZIP of ~10-chunk segments (acts as "chapters")
        segment_zip_bytes = None
        if make_zip_segments:
            zip_buf = io.BytesIO()
            group_size = max(1, len(segment_files_for_zip) // 10) if len(segment_files_for_zip) > 10 else len(segment_files_for_zip)
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                group_audio = AudioSegment.empty()
                group_idx = 1
                count_in_group = 0
                for idx, seg in segment_files_for_zip:
                    group_audio += seg + silence
                    count_in_group += 1
                    if count_in_group >= max(group_size, 1):
                        part_buf = io.BytesIO()
                        group_audio.export(part_buf, format="mp3", bitrate="128k")
                        zf.writestr(f"Part_{group_idx:02d}_Hindi.mp3", part_buf.getvalue())
                        part_buf.close()
                        group_audio = AudioSegment.empty()
                        group_idx += 1
                        count_in_group = 0
                if len(group_audio) > 0:
                    part_buf = io.BytesIO()
                    group_audio.export(part_buf, format="mp3", bitrate="128k")
                    zf.writestr(f"Part_{group_idx:02d}_Hindi.mp3", part_buf.getvalue())
                    part_buf.close()
            segment_zip_bytes = zip_buf.getvalue()
            zip_buf.close()

        progress_bar.progress(1.0)
        status_text.markdown("✅ **Done!** Your Hindi audiobook is ready below.")
        log(f"Pipeline complete. {failed_chunks} chunk(s) failed and were skipped.")

        preview_text = " ".join(hindi_chunks[:6])[:1200]

        return PipelineResult(
            final_audio=final_bytes,
            segment_zip=segment_zip_bytes,
            hindi_preview=preview_text,
            total_chunks=total_chunks,
            failed_chunks=failed_chunks,
        )
    finally:
        # ---- MEMORY / DISK CLEANUP ----
        try:
            for f in os.listdir(tmp_dir):
                try:
                    os.remove(os.path.join(tmp_dir, f))
                except OSError:
                    pass
            os.rmdir(tmp_dir)
        except OSError:
            pass
        # Dro
