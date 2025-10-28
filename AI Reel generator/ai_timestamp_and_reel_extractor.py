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
import re # Import regex for cleaning

load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

class VideoTimestampState(TypedDict):
    video_path: str
    audio_path: str
    transcription_data: dict
    important_timestamps: List[Dict]
    summary_mode: str  # "summary" or "reel"
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
        # Use pcm_s16le for broad compatibility, 16kHz sample rate, mono
        subprocess.run([
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', audio_path
        ], capture_output=True, text=True, check=True)

        print(f"[NODE 1] ✅ Audio extracted: {audio_path}")
        return {**state, "audio_path": audio_path, "error": ""}

    except Exception as e:
        print(f"[NODE 1] ❌ Error: {e}")
        return {**state, "error": str(e)}


def split_audio(audio_path: str, video_duration: float) -> List[str]:
    """Split audio based on duration to avoid API file size limits."""
    output_dir = Path(audio_path).with_suffix('').as_posix() + "_chunks"
    os.makedirs(output_dir, exist_ok=True)
    
    # Groq Whisper API limit is 25MB
    # WAV (pcm_s16le, 16kHz, mono) is ~ 31.25 kB/s
    # 25MB / 31.25 kB/s = 800 seconds max per chunk
    # Let's use 780s (13 minutes) to be safe.
    segment_time_seconds = 780 

    print(f"[NODE 1.1] Splitting audio into {segment_time_seconds}s chunks.")
    cmd = ['ffmpeg', '-i', audio_path, '-f', 'segment', '-segment_time', str(segment_time_seconds), '-c', 'copy', f'{output_dir}/chunk_%03d.wav']

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        raise

    chunk_files = [str(Path(output_dir) / f) for f in sorted(os.listdir(output_dir)) if f.endswith('.wav')]
    print(f"[NODE 1.1] Audio split into {len(chunk_files)} chunks.")
    return chunk_files


def transcribe_audio_node(state: VideoTimestampState) -> VideoTimestampState:
    try:
        if state.get("error"):
            return state

        audio_path = state["audio_path"]
        video_path = state["video_path"]
        video_duration = get_video_duration(video_path)

        chunk_paths = split_audio(audio_path, video_duration)
        print(f"[NODE 2] Transcribing {len(chunk_paths)} audio chunks...")

        groq_api_key = os.getenv("GROQ_API_KEY")
        client = Groq(api_key=groq_api_key)

        merged_segments = []
        time_offset = 0.0
        
        for i, chunk in enumerate(chunk_paths):
            print(f"  > Transcribing chunk {i+1}/{len(chunk_paths)}...")
            with open(chunk, 'rb') as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=(chunk, audio_file.read()),
                    model='whisper-large-v3',
                    response_format='verbose_json',
                    language='en',
                    timestamp_granularities=['segment']
                )
            
            # The API returns a JSON string, so we parse it
            data = json.loads(transcription.model_dump_json())
            
            chunk_segments = data.get('segments', [])
            if not chunk_segments:
                print(f"  > No segments found in chunk {i+1}.")
                # We still need to account for this chunk's time
                chunk_duration = get_video_duration(chunk) # Check chunk duration
                time_offset += chunk_duration
                continue

            for seg in chunk_segments:
                seg['start'] += time_offset
                seg['end'] += time_offset
                merged_segments.append(seg)
            
            # Use the *end time of the last segment* to set the new offset
            # This is more accurate than data.get('duration') if there's silence at the end
            time_offset = merged_segments[-1]['end']

        transcription_data = {
            "segments": merged_segments,
            "text": ' '.join(seg['text'] for seg in merged_segments),
            "duration": time_offset
        }

        print(f"[NODE 2] ✅ Transcription complete. {len(merged_segments)} segments.")
        return {**state, "transcription_data": transcription_data, "error": ""}

    except Exception as e:
        print(f"[NODE 2] ❌ Error: {e}")
        return {**state, "error": str(e)}


def format_timestamp(seconds: float) -> str:
    """Helper to format seconds into hh:mm:ss or mm:ss"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def extract_important_timestamps_node(state: VideoTimestampState) -> VideoTimestampState:
    """
    This is Node 3 ("find_all_clips"). It chunks the transcript and
    finds ALL potential clips.
    """
    try:
        if state.get("error"):
            return state

        segments = state['transcription_data'].get('segments', [])
        if not segments:
            raise ValueError("No transcription segments.")

        llm = ChatGroq(
            api_key=os.getenv('GROQ_API_KEY'),
            model_name='openai/gpt-oss-20b',
            temperature=0.1,
            max_tokens=4096
        )
        
        # This limit is based on CHARACTERS to stay under the 8k TPM token limit
        MAX_CHARS_PER_CHUNK = 7000

        system_prompt = f"""You are an expert video analyst. You will be provided with a transcript, broken into small, timestamped segments. Each segment has an ID (e.g., [0]).
        
        Your task is to identify 5-10 key topics or "chapters" *within this transcript chunk*.
        For each key topic, you must identify a *range* of segments that cover it.
        
        RULES:
        1.  Identify a key topic.
        2.  Find the 'start_segment_id' and 'end_segment_id' that form a coherent discussion on that topic.
        3.  The *total duration* of this range (from the start of the first segment to the end of the last) should ideally be between 10 and 45 seconds.
        4.  The segments must be contiguous (e.g., from ID 10 to 12).
        5.  Provide a brief 'description' of what this topic/clip is about.
        
        Return ONLY a valid JSON object with the following structure:
        {{"important_clips": [
            {{"start_segment_id": int, "end_segment_id": int, "description": "str"}}
        ]}}
        """
        
        all_found_clips = []
        current_chunk_text = ""
        current_chunk_start_index = 0
        
        print(f"[NODE 3] Starting chunked analysis of {len(segments)} segments...")

        for i, seg in enumerate(segments):
            start = format_timestamp(seg['start'])
            end = format_timestamp(seg['end'])
            text = seg.get('text', '').strip()
            
            segment_line = f"[{i}] [{start}-{end}] {text}\n"

            if len(current_chunk_text) + len(segment_line) > MAX_CHARS_PER_CHUNK and current_chunk_text:
                print(f"[NODE 3] Processing chunk (Segments {current_chunk_start_index}-{i-1})...")
                human_message = f"Here is the transcript chunk:\n\n{current_chunk_text}"
                
                try:
                    response = llm.invoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=human_message),
                    ])
                    llm_output = response.content.strip()
                    
                    json_start = llm_output.find('{')
                    json_end = llm_output.rfind('}') + 1
                    if json_start == -1 or json_end == 0:
                        raise ValueError("No JSON found in LLM output")
                        
                    llm_result = json.loads(llm_output[json_start:json_end])
                    found_clips = llm_result.get('important_clips', [])
                    print(f"[NODE 3]   > Found {len(found_clips)} clips in this chunk.")
                    all_found_clips.extend(found_clips)
                
                except Exception as e:
                    print(f"[NODE 3] ⚠️ Failed to process chunk {current_chunk_start_index}-{i-1}: {e}")
                    print(f"LLM Output/Error for failed chunk: {str(e)[:200]}...")

                # Start a new chunk
                current_chunk_text = segment_line
                current_chunk_start_index = i
            
            else:
                current_chunk_text += segment_line

        # --- Process the final remaining chunk ---
        if current_chunk_text:
            print(f"[NODE 3] Processing final chunk (Segments {current_chunk_start_index}-{len(segments)-1})...")
            human_message = f"Here is the transcript chunk:\n\n{current_chunk_text}"
            
            try:
                response = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_message),
                ])
                llm_output = response.content.strip()
                
                json_start = llm_output.find('{')
                json_end = llm_output.rfind('}') + 1
                if json_start == -1 or json_end == 0:
                    raise ValueError("No JSON found in LLM output")

                llm_result = json.loads(llm_output[json_start:json_end])
                found_clips = llm_result.get('important_clips', [])
                print(f"[NODE 3]   > Found {len(found_clips)} clips in this final chunk.")
                all_found_clips.extend(found_clips)

            except Exception as e:
                print(f"[NODE 3] ⚠️ Failed to process final chunk: {e}")
                print(f"LLM Output/Error for failed chunk: {str(e)[:200]}...")

        # --- Consolidate all clips ---
        print(f"[NODE 3] Consolidating {len(all_found_clips)} clips from all chunks...")
        
        important_timestamps = []
        for clip in all_found_clips:
            start_id = clip.get('start_segment_id')
            end_id = clip.get('end_segment_id')
            desc = clip.get('description', '')

            if start_id is None or end_id is None or \
               not (isinstance(start_id, int) and isinstance(end_id, int)) or \
               not (0 <= start_id < len(segments)) or \
               not (0 <= end_id < len(segments)) or \
               end_id < start_id:
                print(f"[NODE 3] ⚠️ Skipping invalid clip data from LLM: {clip}")
                continue

            start_seg = segments[start_id]
            end_seg = segments[end_id]
            
            # Combine text from all segments in the range
            clip_text = " ".join([
                segments[i]['text'].strip() 
                for i in range(start_id, end_id + 1)
            ])
            # Clean up excessive whitespace
            clip_text = re.sub(r'\s+', ' ', clip_text).strip()


            important_timestamps.append({
                'segment_id': start_id, 
                'timestamp_start': format_timestamp(start_seg['start']),
                'timestamp_end': format_timestamp(end_seg['end']),
                'start_seconds': start_seg['start'],
                'end_seconds': end_seg['end'],
                'description': desc,
                'text': clip_text
            })

        if not important_timestamps:
            print("[NODE 3] ⚠️ No valid clips were processed after LLM response.")
        
        print(f"[NODE 3] ✅ Found {len(important_timestamps)} total potential clips.")
        # We pass this *full list* to the next node
        return {**state, "important_timestamps": important_timestamps, "error": ""}

    except Exception as e:
        print(f"[NODE 3] ❌ Critical Error: {e}")
        return {**state, "error": str(e)}


def summarize_clips_node(state: VideoTimestampState) -> VideoTimestampState:
    """
    This is Node 3.5 ("summarize_clips"). It takes the full list of clips
    and filters them down based on the summary_mode.
    
    *** THIS IS THE FIXED VERSION ***
    """
    llm_output = "" # Define here for error logging
    try:
        if state.get("error"):
            return state

        clips = state.get("important_timestamps", [])
        if not clips:
            print("[NODE 3.5] No clips found to summarize.")
            return state

        mode = state.get("summary_mode", "summary")

        # --- 1. Define the Prompts for Each Tool ---

        # PROMPT 1: For the "Video Summary & Chapter" Tool
        prompt_for_summary = f"""You are an expert video editor. You will be given {len(clips)} potential "highlight clips".
        
        Your task is to select the 10-15 *best* and most representative clips that would create an excellent, detailed summary video.
        
        RULES:
        1. Read all clip descriptions.
        2. Identify the main topics.
        3. Select 10-15 clips that provide the best overview, cover the most important topics, and are not redundant.
        4. You MUST return ONLY the `segment_id` of the clips you select.
        
        Return ONLY a valid JSON object: {{"top_clip_ids": [int, int, int, ...]}}
        """

        # PROMPT 2: For the "Reel Creator" Tool (Stricter)
        prompt_for_reel = f"""You are an expert social media video editor. You will be given {len(clips)} potential clips.
        
        Your task is to select the 2-3 *best* clips to create a "viral" reel that is **under 90 seconds TOTAL**.
        
        RULES:
        1. Look for high-energy clips with clear "hooks" (a question or a bold statement).
        2. Prioritize clips that summarize a key takeaway or a "wow" moment.
        3. Select only 2-3 clips that are the *most engaging* and *shareable*.
        4. The *total combined duration* of your selected clips should be **between 45 and 90 seconds.**
        5. You MUST return ONLY the `segment_id` of the clips you select.
        
        Return ONLY a valid JSON object: {{"top_clip_ids": [int, int]}}
        """

        # --- 2. Choose the prompt based on the mode ---
        system_prompt = ""
        
        if mode == "reel":
            print(f"[NODE 3.5] Using 'Reel Creator' mode. Finding 2-3 viral clips...")
            system_prompt = prompt_for_reel
            
        else: # Default to "summary"
            print(f"[NODE 3.5] Using 'Video Summary' mode. Finding 10-15 key chapters...")
            system_prompt = prompt_for_summary
            
            if len(clips) <= 15:
                print(f"[NODE 3.5] Only {len(clips)} clips found. Skipping summarization for 'summary' mode.")
                return state

        # --- 3. Build the clip text (THE FIX IS HERE) ---
        # We ONLY send the descriptions, not the full text, to avoid large prompts.
        clips_text = ""
        for clip in clips:
            clips_text += f"ID: {clip['segment_id']}\n"
            clips_text += f"Timestamp: {clip['timestamp_start']} - {clip['timestamp_end']}\n"
            clips_text += f"Description: {clip['description']}\n\n"
            # We removed clip['text'] to keep the prompt small and reliable

        # --- 4. Create the LLM and Invoke ---
        llm = ChatGroq(
            api_key=os.getenv('GROQ_API_KEY'),
            model_name='openai/gpt-oss-20b',
            temperature=0.0
        )
        human_message = f"Here is the list of all {len(clips)} potential clips:\n\n{clips_text}"
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message),
        ])
        llm_output = response.content.strip()

        # --- 5. Parse and Filter ---
        json_start = llm_output.find('{')
        json_end = llm_output.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            # This is how we catch the "Expecting value" error
            raise ValueError("LLM returned non-JSON output.")

        llm_result = json.loads(llm_output[json_start:json_end])
        
        top_ids = llm_result.get('top_clip_ids', [])
        
        if not top_ids:
            raise ValueError("LLM returned 'top_clip_ids' but the list was empty.")

        top_ids_set = set(top_ids)
        final_clips = [clip for clip in clips if clip['segment_id'] in top_ids_set]

        print(f"[NODE 3.5] ✅ Summarization complete. Selected {len(final_clips)} clips.")
        
        return {**state, "important_timestamps": final_clips, "error": ""}

    except Exception as e:
        print(f"[NODE 3.5] ❌ Error in summarization: {e}.")
        print(f"[NODE 3.5] ⚠️ LLM Output was: {llm_output}")
        # If summarization fails, we just return the *original* long list
        # This way, the user still gets a result (the "Rough Cut")
        print(f"[NODE 3.5] ⚠️ Skipping summarization step and returning all clips.")
        return state


def save_timestamps_node(state: VideoTimestampState) -> VideoTimestampState:
    try:
        if state.get("error"):
            # Don't save if a critical error happened before
            print(f"[NODE 4] Skipping save due to previous error: {state.get('error')}")
            return state

        data = state['important_timestamps']
        mode = state.get('summary_mode', 'summary') # Get the mode
        
        if not data:
            print("[NODE 4] ⚠️ No timestamps to save.")
            return state

        # --- Save to different files based on the mode ---
        json_filename = f'timestamps_{mode}.json'
        csv_filename = f'timestamps_{mode}.csv'

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[NODE 4] 💾 JSON saved: {json_filename}")

        with open(csv_filename, 'w', encoding='utf-8', newline='') as f:
            # Use csv writer for proper CSV formatting
            import csv
            writer = csv.writer(f)
            writer.writerow(['Index', 'Start Time', 'End Time', 'Start(s)', 'End(s)', 'Description', 'Text'])
            for i, e in enumerate(data, 1):
                writer.writerow([
                    i, 
                    e["timestamp_start"], 
                    e["timestamp_end"], 
                    e["start_seconds"], 
                    e["end_seconds"], 
                    e["description"], 
                    e["text"]
                ])
        print(f"[NODE 4] 💾 CSV saved: {csv_filename}")

        return {**state, "error": ""}

    except Exception as e:
        print(f"[NODE 4] ❌ Error: {e}")
        return {**state, "error": str(e)}


def build_timestamp_extraction_graph():
    g = StateGraph(VideoTimestampState)
    g.add_node('extract_audio', extract_audio_node)
    g.add_node('transcribe_audio', transcribe_audio_node)
    g.add_node('find_all_clips', extract_important_timestamps_node) 
    g.add_node('summarize_clips', summarize_clips_node) 
    g.add_node('save_timestamps', save_timestamps_node)

    g.set_entry_point('extract_audio')
    g.add_edge('extract_audio', 'transcribe_audio')
    g.add_edge('transcribe_audio', 'find_all_clips')
    g.add_edge('find_all_clips', 'summarize_clips')
    g.add_edge('summarize_clips', 'save_timestamps')
    g.add_edge('save_timestamps', END)

    return g.compile()


def process_video(video_path: str, summary_mode: str = "summary") -> dict:
    """
    Processes the video using the graph.
    
    Args:
        video_path: Path to the video file.
        summary_mode: "summary" (for 10-15 clips) or "reel" (for 3-5 clips).
    """
    try:
        print(f"\n{'='*70}\n🎬 VIDEO TIMESTAMP EXTRACTION PIPELINE\n{'='*70}\n")
        print(f"🚀 Starting in '{summary_mode}' mode.")
        
        initial_state = {
            "video_path": video_path,
            "audio_path": "",
            "transcription_data": {},
            "important_timestamps": [],
            "summary_mode": summary_mode,
            "error": ""
        }
        graph = build_timestamp_extraction_graph()
        final_state = graph.invoke(initial_state)

        if final_state.get("error"):
            print(f"❌ ERROR: {final_state['error']}")
            return final_state

        print(f"\n{'='*70}\n✅ PROCESS COMPLETED SUCCESSFULLY!\n{'='*70}")
        print(f"📁 Output Files: timestamps_{summary_mode}.json, timestamps_{summary_mode}.csv")
        print(f"📊 Segments: {len(final_state['transcription_data'].get('segments', []))}")
        print(f"⭐ Important: {len(final_state['important_timestamps'])}")
        return final_state

    except Exception as e:
        error_msg = f"Process failed: {str(e)}"
        print(f"❌ ERROR: {error_msg}")
        return {"error": error_msg}


if __name__ == "__main__":
    
    # --- IMPORTANT: Set your video file path here ---
    video_file_path = "E:/Downloads/Code Basics MCP Server Tutorial.mp4"
    
    if not os.getenv("GROQ_API_KEY"):
        print("❌ ERROR: Please set GROQ_API_KEY environment variable")
    else:
        # --- 1. Run the "Detailed Summary" Tool ---
        # This will create 'timestamps_summary.json'
        process_video(video_file_path, summary_mode="summary")
        
        
        # --- 2. Run the "Reel Creator" Tool ---
        # This will create 'timestamps_reel.json'
        process_video(video_file_path, summary_mode="reel")

