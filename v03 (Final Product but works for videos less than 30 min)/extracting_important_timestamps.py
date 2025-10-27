import os
import subprocess
import tempfile
import json
from pathlib import Path
from typing import List, Dict, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from groq import Groq

load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")


class VideoTimestampState(TypedDict):
    video_path: str
    audio_path: str
    transcription_data: dict
    important_timestamps: List[Dict]
    error: str


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def extract_audio_node(state: VideoTimestampState) -> VideoTimestampState:
    try:
        video_path = state["video_path"]
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio_path = temp_audio.name
        temp_audio.close()

        print(f"[NODE 1] Extracting audio...")
        subprocess.run([
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', audio_path
        ], capture_output=True, text=True, check=True)

        print(f"[NODE 1] ✅ Audio extracted: {audio_path}")
        return {**state, "audio_path": audio_path, "error": ""}

    except Exception as e:
        print(f"[NODE 1] ❌ Error: {e}")
        return {**state, "error": str(e)}


def split_audio(audio_path: str, video_duration: float) -> List[str]:
    """Split audio depending on duration or file size."""
    output_dir = Path(audio_path).with_suffix('').as_posix() + "_chunks"
    os.makedirs(output_dir, exist_ok=True)

    if video_duration <= 20 * 60:
        print(f"[NODE 1.1] Short video detected ({video_duration/60:.1f} min). Using fixed 60s chunks.")
        cmd = ['ffmpeg', '-i', audio_path, '-f', 'segment', '-segment_time', '60', '-c', 'copy', f'{output_dir}/chunk_%03d.wav']
    else:
        print(f"[NODE 1.1] Long video detected ({video_duration/60:.1f} min). Using adaptive splitting.")
        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        num_chunks = max(1, int(size_mb // 20) + 1)
        segment_time = max(30, video_duration / num_chunks)
        cmd = ['ffmpeg', '-i', audio_path, '-f', 'segment', '-segment_time', str(segment_time), '-c', 'copy', f'{output_dir}/chunk_%03d.wav']

    subprocess.run(cmd, check=True)
    return [str(Path(output_dir) / f) for f in sorted(os.listdir(output_dir)) if f.endswith('.wav')]


def transcribe_audio_node(state: VideoTimestampState) -> VideoTimestampState:
    try:
        if state.get("error"):
            return state

        audio_path = state["audio_path"]
        video_path = state["video_path"]
        video_duration = get_video_duration(video_path)

        chunk_paths = split_audio(audio_path, video_duration)
        print(f"[NODE 2] Split into {len(chunk_paths)} chunks for transcription.")

        groq_api_key = os.getenv("GROQ_API_KEY")
        client = Groq(api_key=groq_api_key)

        merged_segments = []
        offset = 0.0
        for chunk in chunk_paths:
            with open(chunk, 'rb') as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=(chunk, audio_file.read()),
                    model='whisper-large-v3',
                    response_format='verbose_json',
                    language='en',
                    timestamp_granularities=['segment']
                )
            data = json.loads(transcription.model_dump_json())
            for seg in data.get('segments', []):
                seg['start'] += offset
                seg['end'] += offset
                merged_segments.append(seg)
            offset += data.get('duration', 0)

        transcription_data = {
            "segments": merged_segments,
            "text": ' '.join(seg['text'] for seg in merged_segments),
            "duration": offset
        }

        print(f"[NODE 2] ✅ Transcription complete. {len(merged_segments)} segments.")
        return {**state, "transcription_data": transcription_data, "error": ""}

    except Exception as e:
        print(f"[NODE 2] ❌ Error: {e}")
        return {**state, "error": str(e)}


def extract_important_timestamps_node(state: VideoTimestampState) -> VideoTimestampState:
    try:
        if state.get("error"):
            return state

        segments = state['transcription_data'].get('segments', [])
        if not segments:
            raise ValueError("No transcription segments.")

        print(f"[NODE 3] Building segments text for LLM...")
        segments_text = ''
        for i, seg in enumerate(segments):
            start = format_timestamp(seg['start'])
            end = format_timestamp(seg['end'])
            text = seg.get('text', '').strip()
            # Provide the LLM with all the info it needs
            segments_text += f"[{i}] [{start}-{end}] {text}\n"

        # --- FIX 1: Use a reliable model for JSON tasks ---
        # 'openai/gpt-oss-20b' is not a standard Groq model.
        llm = ChatGroq(
            api_key=os.getenv('GROQ_API_KEY'), 
            model_name='openai/gpt-oss-20b', 
            temperature=0.1 # Low temperature for predictable JSON
        )

        # --- FIX 2: Update prompt to ask for RANGES (clips) not single segments ---
        system_prompt = f"""You are an expert video analyst. You will be provided with a transcript, broken into small, timestamped segments. Each segment has an ID (e.g., [0]).
        
        Your task is to identify 5-10 key topics or "chapters" in this video.
        For each key topic, you must identify a *range* of segments that cover it.
        
        RULES:
        1.  Identify a key topic.
        2.  Find the 'start_segment_id' and 'end_segment_id' that form a coherent discussion on that topic.
        3.  The *total duration* of this range (from the start of the first segment to the end of the last) should ideally be between 10 and 45 seconds.
        4.  The segments must be contiguous (e.g., from ID 10 to 12).
        5.  Provide a brief 'description' of what this topic/clip is about.
        6.  Output 5–10 such ranges.
        
        Return ONLY a valid JSON object with the following structure:
        {{"important_clips": [
            {{"start_segment_id": int, "end_segment_id": int, "description": "str"}}
        ]}}
        """
        
        human_message = f"Here is the transcript:\n\n{segments_text}"

        print(f"[NODE 3] Finding key clips using LLM...")
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message),
        ])
        llm_output = response.content.strip()

        # --- FIX 3: New logic to parse clips and handle ranges ---
        json_start = llm_output.find('{')
        json_end = llm_output.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            print(f"LLM output was not valid JSON: {llm_output}")
            raise ValueError("LLM did not return valid JSON.")
            
        llm_result = json.loads(llm_output[json_start:json_end])
        
        important_clips = llm_result.get('important_clips', [])
        if not important_clips:
            print(f"LLM returned no important clips. Response: {llm_result}")

        important_timestamps = []
        for clip in important_clips:
            start_id = clip.get('start_segment_id')
            end_id = clip.get('end_segment_id')
            desc = clip.get('description', '')

            # Basic validation
            if start_id is None or end_id is None or \
               not (isinstance(start_id, int) and isinstance(end_id, int)) or \
               not (0 <= start_id < len(segments)) or \
               not (0 <= end_id < len(segments)) or \
               end_id < start_id:
                print(f"[NODE 3] ⚠️ Skipping invalid clip data from LLM: {clip}")
                continue

            # Get the start of the first segment and end of the last segment
            start_seg = segments[start_id]
            end_seg = segments[end_id]

            # Combine text from all segments in the range
            clip_text = " ".join([
                segments[i]['text'].strip() 
                for i in range(start_id, end_id + 1)
            ])

            important_timestamps.append({
                'segment_id': start_id, # Use start_id as the primary reference
                'timestamp_start': format_timestamp(start_seg['start']),
                'timestamp_end': format_timestamp(end_seg['end']),
                'start_seconds': start_seg['start'],
                'end_seconds': end_seg['end'],
                'description': desc,
                'text': clip_text
            })

        if not important_timestamps:
            print("[NODE 3] ⚠️ No valid clips were processed after LLM response.")
        
        print(f"[NODE 3] ✅ Found {len(important_timestamps)} key clips.")
        return {**state, "important_timestamps": important_timestamps, "error": ""}

    except Exception as e:
        print(f"[NODE 3] ❌ Error: {e}")
        return {**state, "error": str(e)}

def save_timestamps_node(state: VideoTimestampState) -> VideoTimestampState:
    try:
        if state.get("error"):
            return state

        data = state['important_timestamps']
        if not data:
            print("[NODE 4] ⚠️ No timestamps to save.")
            return state

        with open('important_timestamps.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("[NODE 4] 💾 JSON saved.")

        with open('important_timestamps.csv', 'w', encoding='utf-8') as f:
            f.write('Index,Start Time,End Time,Start(s),End(s),Description,Text\n')
            for i, e in enumerate(data, 1):
                f.write(f'{i},"{e["timestamp_start"]}","{e["timestamp_end"]}",{e["start_seconds"]},{e["end_seconds"]},"{e["description"]}","{e["text"].replace("\"","''")}"\n')
        print("[NODE 4] 💾 CSV saved.")

        return {**state, "error": ""}

    except Exception as e:
        print(f"[NODE 4] ❌ Error: {e}")
        return {**state, "error": str(e)}


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def build_timestamp_extraction_graph():
    g = StateGraph(VideoTimestampState)
    g.add_node('extract_audio', extract_audio_node)
    g.add_node('transcribe_audio', transcribe_audio_node)
    g.add_node('extract_timestamps', extract_important_timestamps_node)
    g.add_node('save_timestamps', save_timestamps_node)

    g.set_entry_point('extract_audio')
    g.add_edge('extract_audio', 'transcribe_audio')
    g.add_edge('transcribe_audio', 'extract_timestamps')
    g.add_edge('extract_timestamps', 'save_timestamps')
    g.add_edge('save_timestamps', END)

    return g.compile()

def process_video(video_path: str) -> dict:
    try:
        print(f"\n{'='*70}\n🎬 VIDEO TIMESTAMP EXTRACTION PIPELINE\n{'='*70}\n")
        initial_state = {
            "video_path": video_path,
            "audio_path": "",
            "transcription_data": {},
            "important_timestamps": [],
            "error": ""
        }
        graph = build_timestamp_extraction_graph()
        final_state = graph.invoke(initial_state)

        if final_state.get("error"):
            print(f"❌ ERROR: {final_state['error']}")
            return final_state

        print(f"✅ PROCESS COMPLETED SUCCESSFULLY!")
        print(f"📁 Output Files: important_timestamps.json, important_timestamps.csv")
        print(f"📊 Segments: {len(final_state['transcription_data'].get('segments', []))}")
        print(f"⭐ Important: {len(final_state['important_timestamps'])}")
        return final_state

    except Exception as e:
        error_msg = f"Process failed: {str(e)}"
        print(f"❌ ERROR: {error_msg}")
        return {"error": error_msg}


if __name__ == "__main__":
    video_file_path = "E:/Downloads/Live Day 4- Discussing Decision Tree And Ensemble Machine Learning Algorithms.mp4"
    if not os.getenv("GROQ_API_KEY"):
        print("❌ ERROR: Please set GROQ_API_KEY environment variable")
    else:
        result = process_video(video_file_path)
