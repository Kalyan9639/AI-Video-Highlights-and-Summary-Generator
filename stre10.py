import os
import tempfile
import streamlit as st
# PAGE CONFIG
st.set_page_config(page_title="Video Transcription & Highlights", layout="wide")

from moviepy.editor import VideoFileClip
import librosa
import soundfile as sf
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from google.genai import Client, types
import pyttsx3
from PIL import Image
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
import easyocr

# Insert your Google GenAI API key here
api_key = "AIzaSyCQIYPtpOxSorTaLhsVvQuG1_ZHD-nZct4"

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
model_name = st.sidebar.selectbox("Whisper Model", ["openai/whisper-base", "openai/whisper-medium"], index=0)
chunk_minutes = st.sidebar.number_input("Highlight Chunk Size (minutes)", min_value=1, max_value=10, value=2)
words_per_minute = st.sidebar.number_input("Avg Words/Minute", min_value=50, max_value=300, value=150)
summary_length = st.sidebar.selectbox(
    "Summary Length",
    ["Small (~50 words)", "Medium (~150 words)", "Large (~250 words)"],
    index=1
)
visual_analysis = st.sidebar.checkbox(
    "Analyze Visual Content (frame caption & OCR)",
    help="May take several minutes."
)

# Main UI
st.title("🎬 AI Video Transcription & Highlights Generator")
video_file = st.file_uploader("Upload Video (mp4, mov, avi)")

# Helper functions
@st.cache_data
def extract_audio(video_path: str) -> str:
    tmp = tempfile.mkdtemp()
    mp3 = os.path.join(tmp, "audio.mp3")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(mp3, logger=None)
    y, sr = librosa.load(mp3, sr=16000)
    wav = os.path.join(tmp, "audio.wav")
    sf.write(wav, y, sr)
    return wav

@st.cache_data
def transcribe_audio(audio_path: str, model_name: str) -> str:
    proc = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    y, sr = librosa.load(audio_path, sr=16000)
    seg = 30 * sr
    texts = []
    for i in range(0, len(y), seg):
        chunk = y[i:i+seg]
        inp = proc(chunk, sampling_rate=sr, return_tensors="pt").input_features.to(device)
        ids = model.generate(inp)
        texts.append(proc.batch_decode(ids, skip_special_tokens=True)[0])
    return "\n".join(texts)

@st.cache_data
def split_chunks(text: str, minutes: int, wpm: int) -> list:
    words = text.split()
    size = minutes * wpm
    return [' '.join(words[i:i+size]) for i in range(0, len(words), size)]

@st.cache_data
def generate_highlights(chunks: list) -> list:
    client = Client(api_key=api_key)
    hl = []
    for idx, chunk in enumerate(chunks):
        instr = (
            f"You are an AI summarizer. Generate key highlights for transcript segment "
            f"(minutes {idx*chunk_minutes}-{(idx+1)*chunk_minutes}):\n{chunk}"
        )
        chat = client.chats.create(
            model="gemini-2.0-flash",
            history=[],
            config=types.GenerateContentConfig(
                temperature=0.5,
                max_output_tokens=512,
                top_p=0.8,
                system_instruction=instr
            )
        )
        resp = chat.send_message("Generate highlights.")
        hl.append({
            "start": idx * chunk_minutes,
            "end": (idx + 1) * chunk_minutes,
            "text": resp.text.strip()
        })
    return hl

@st.cache_data
def summarize_highlights(highlights: list, word_target: int) -> str:
    client = Client(api_key=api_key)
    context = "\n".join([
        f"{h['start']:02d}-{h['end']:02d} min: {h['text']}" for h in highlights
    ])
    instr = (
        f"You are an AI summarizer. Given these highlights, provide a concise summary "
        f"of about {word_target} words:\n" + context
    )
    chat = client.chats.create(
        model="gemini-2.0-flash",
        history=[],
        config=types.GenerateContentConfig(
            temperature=0.5,
            max_output_tokens=512,
            top_p=0.8,
            system_instruction=instr
        )
    )
    resp = chat.send_message("Generate overall summary.")
    return resp.text.strip()

@st.cache_data
def extract_frames_and_caption(video_path: str, output_txt: str):
    blip_p = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_m = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    ocr_reader = easyocr.Reader(['en'])
    keywords = ['computer', 'webpage', 'screen', 'presentation']
    vid = cv2.VideoCapture(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    bi, oi = int(fps * 30), int(fps * 60)
    with open(output_txt, 'w', encoding='utf-8') as out:
        frame_num = 0
        last_caption = ""
        while True:
            ret, frame = vid.read()
            if not ret:
                break
            frame_num += 1
            if frame_num % bi == 0:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = blip_p(img, return_tensors="pt")
                with torch.no_grad():
                    output_ids = blip_m.generate(**inputs)
                caption = blip_p.decode(output_ids[0], skip_special_tokens=True)
                last_caption = caption
                out.write(f"Caption at {frame_num/fps:.1f}s: {caption}\n")
            if visual_analysis and frame_num % oi == 0 and any(k in last_caption.lower() for k in keywords):
                texts = [t for _, t, _ in ocr_reader.readtext(frame)]
                out.write(f"OCR at {frame_num/fps:.1f}s: {' '.join(texts)}\n")
        vid.release()

# Text-to-Speech function
def text_to_speech(text, filename):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    engine.save_to_file(text, filename)
    engine.runAndWait()
    return filename

# Main logic
if video_file:
    source_path = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]).name
    with open(source_path, 'wb') as f:
        f.write(video_file.read())

    if visual_analysis:
        st.warning("🔍 Visual analysis will take several minutes...")
        vis_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
        with st.spinner("Analyzing visuals..."):
            extract_frames_and_caption(source_path, vis_file)
        st.success("✔️ Visual analysis complete")
        vis_text = open(vis_file, 'r', encoding='utf-8').read()
        st.subheader("🖼️ Frame Captions & OCR")
        st.text_area("", vis_text, height=300)
        st.download_button("Download Visual Analysis", vis_text, file_name="visual_analysis.txt")

    st.info("🔊 Extracting and normalizing audio...")
    audio_path = extract_audio(source_path)
    st.success("✔️ Audio ready")

    st.info("📝 Transcribing audio with Whisper...")
    transcript = transcribe_audio(audio_path, model_name)
    st.success("✔️ Transcription complete")

    st.subheader("🎯 Highlights")
    chunks = split_chunks(transcript, chunk_minutes, words_per_minute)
    highlights = generate_highlights(chunks)
    highlights_txt = "\n".join([
        f"**{h['start']:02d}:00 - {h['end']:02d}:00**: {h['text']}" for h in highlights
    ])
    st.markdown(highlights_txt)

    # Highlights audio
    highlights_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    highlight_texts = "\n".join([h['text'] for h in highlights])
    text_to_speech(highlight_texts, highlights_audio_path)
    st.audio(highlights_audio_path, format='audio/mp3')
    st.download_button("Download Highlights Audio", open(highlights_audio_path, 'rb'), file_name="highlights.mp3")

    st.subheader("📄 Summary")
    target_map = {"Small (~50 words)":50, "Medium (~150 words)":150, "Large (~250 words)":250}
    word_target = target_map[summary_length]
    summary_text = summarize_highlights(highlights, word_target)
    st.write(summary_text)

    # Summary audio
    summary_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    text_to_speech(summary_text, summary_audio_path)
    st.audio(summary_audio_path, format='audio/mp3')
    st.download_button("Download Summary Audio", open(summary_audio_path, 'rb'), file_name="summary.mp3")

else:
    st.info("Upload a video file to begin.")
