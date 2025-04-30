from moviepy.editor import VideoFileClip
import librosa
import soundfile as sf
import os
import json

video = VideoFileClip(r"E:\IITB Projects Videos\Python projeccts\Healing potion python project.mp4")

video.audio.write_audiofile("audio.mp3")

current_dir = os.getcwd()
audio_path = os.path.join(current_dir,"audio.mp3")



y,sr = librosa.load(audio_path, sr=16000)

output_path = os.path.join(current_dir,"normalized_audio.wav")
sf.write(output_path,y,16000)





# --------------------------------------audio-to-text-converter----------------------------------------------------
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Step 1: Load audio
file_name = "normalized_audio.wav"  # Audio should be 16kHz mono
audio_path = os.path.join(os.getcwd(), file_name)
audio, sr = librosa.load(audio_path, sr=16000)

# Step 2: Load Whisper model & processor
model_name = "openai/whisper-base"  # You can use 'base' or 'medium' depending on accuracy vs speed
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

# Step 3: Define chunking logic
chunk_duration = 30  # seconds
chunk_size = chunk_duration * sr  # samples per chunk

# Step 4: Prepare output structures
output_text_file = "transcription.txt"
output_json_file = "transcription_timestamps.json"
transcriptions_with_timestamps = [] 

# Step 4: Transcribe chunks and save to text file
# output_text_file = "transcription.txt"
with open(output_text_file, "w", encoding="utf-8") as f:
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features.to(device)

        # Generate transcription
        predicted_ids = model.generate(inputs)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Calculate timestamps
        start_time = i / sr
        end_time = min((i + chunk_size), len(audio)) / sr

        # Write to file
        f.write(transcription + "\n")

        # Save to JSON structure
        transcriptions_with_timestamps.append({
            "chunk": i // chunk_size + 1,
            "start_time_sec": round(start_time, 2),
            "end_time_sec": round(end_time, 2),
            "text": transcription
        })

        print(f"Chunk {i // chunk_size + 1}: {round(start_time, 2)}s - {round(end_time, 2)}s")

# Step 6: Write JSON output
with open(output_json_file, "w", encoding="utf-8") as f_json:
    json.dump(transcriptions_with_timestamps, f_json, indent=4)

print(f"\n Transcription complete.\n- Text saved to: {output_text_file}\n- Timestamps saved to: {output_json_file}")

print("Transcription complete. Saved to transcription.txt.")


