# 🎬 AI Video Highlights, Key Moments Generator, and Summarizer

This repository contains a suite of Python scripts designed to process video files, extract key moments, generate highlights, and provide concise summaries. The project leverages state-of-the-art AI models for transcription, visual analysis, and summarization.

---

## 🚀 Features

- **Video-to-Audio Conversion**: Extracts audio from video files and normalizes it for further processing.
- **Audio Transcription**: Converts audio to text using OpenAI's Whisper model.
- **Highlight Generation**: Automatically identifies and generates key moments from the transcript.
- **Visual Analysis**: Captures frames from the video, generates captions using BLIP, and performs OCR on relevant frames.
- **Summarization**: Provides concise summaries of the video content in different lengths (small, medium, large).
- **Text-to-Speech**: Converts highlights and summaries into audio files for easy consumption.

---

## 📂 Project Structure

### 1. `stre10.py`
The main Streamlit application that integrates all functionalities:
- Upload video files or provide URLs.
- Extract audio, transcribe it, and generate highlights.
- Perform visual analysis (frame captions and OCR).
- Generate summaries and provide audio versions of highlights and summaries.

### 2. `video-audio-to-text-extracter.py`
Handles:
- Video-to-audio conversion.
- Audio normalization.
- Audio-to-text transcription using Whisper.

### 3. `visual-info+ocr.py`
Performs:
- Frame extraction and captioning using BLIP.
- OCR on frames containing specific keywords (e.g., "computer", "screen").

### 4. `Summary.py`
Generates:
- Summaries of the transcript and visual information using Google GenAI.
- Processes transcript in chunks to create detailed highlights.

---

## 📖 How It Works
* Audio Extraction: Extracts audio from the video and normalizes it for transcription.
* Transcription: Converts audio to text using Whisper, processing it in chunks for efficiency.
* Highlight Generation: Splits the transcript into smaller chunks and generates key moments using Google GenAI.
* Visual Analysis: Captures frames, generates captions using BLIP, and performs OCR on frames with relevant content.
* Summarization: Combines transcript and visual data to create concise summaries.
* Text-to-Speech: Converts highlights and summaries into audio files for easy playback.

## 🤝 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.
