import os
import json
from dotenv import load_dotenv
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    CompositeVideoClip,
    concatenate_videoclips,
    concatenate_audioclips,
    ColorClip,
    ImageClip,
    CompositeAudioClip,
)
from PIL import Image, ImageDraw, ImageFont
import moviepy.video.fx.all as vfx
from typing import Optional, List, Dict, Any, Literal, Tuple
from typing_extensions import Annotated
import random
import numpy as np

# --- LangGraph & LangChain Imports ---
from langgraph.graph import StateGraph, END, add_messages
from langgraph.graph.message import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from pydantic import BaseModel, Field

load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# --- 1. Define AI Tools ---

@tool
class GenerateTitleTool(BaseModel):
    """Generate a short, italic-style title for this video clip."""
    title: str = Field(..., description="A short, catchy title (max 5 words) based on the clip's description.")

@tool
class ChooseTransitionTool(BaseModel):
    """Choose a video transition to use *after* this clip."""
    transition_name: Literal["crossfade", "slide_left", "slide_right", "dip_to_black"] = Field(
        ..., 
        description="Choose one transition from the allowed list."
    )

# --- 2. Define LangGraph State ---

class ReelGenerationState(MessagesState):
    """Holds the state for the video generation graph."""
    
    # --- Inputs (set at start) ---
    source_video_path: str
    json_path: str
    bgm_path: str
    output_path: str
    
    # --- Loaded Assets ---
    main_video_clip: Optional[VideoFileClip] = None
    bgm_audio_clip: Optional[AudioFileClip] = None
    timestamps: List[Dict] = []
    
    # --- Loop State ---
    current_index: int = 0
    
    # --- Audio Tracking (NEW) ---
    # Store tuples of (audio_clip_path, duration, is_title)
    audio_segments: List[Tuple[Optional[str], float, bool]] = []
    
    # --- Video Tracking (SIMPLIFIED) ---
    # Instead of storing clips, store temp video file paths
    video_segment_paths: List[str] = []
    transition_names: List[str] = []
    
    # --- Final Assets ---
    silent_video_path: str = "temp_silent_video.mp4"
    final_audio_path: str = "temp_final_audio.mp3"
    total_reel_duration: float = 0.0

# --- 3. Helper Functions ---

def _create_title_image_with_pillow(
    text: str,
    width: int,
    height: int,
    output_path: str = "temp_title.png"
) -> str:
    """Uses Pillow to create a static title card image with centered text."""
    print(f"  Creating title card: '{output_path}'")
    
    img = Image.new('RGB', (width, height), color='black')
    d = ImageDraw.Draw(img)
    
    font_path_italic = "C:/Windows/Fonts/ariali.ttf"
    font_path_bold = "C:/Windows/Fonts/arialbd.ttf"
    font_size = 40
    
    font = None
    try:
        font = ImageFont.truetype(font_path_italic, font_size)
    except IOError:
        try:
            font = ImageFont.truetype(font_path_bold, font_size)
        except IOError:
            font = ImageFont.load_default()

    try:
        bbox = d.textbbox((0, 0), text, font=font, align="center")
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (width - text_width) / 2
        y = (height - text_height) / 2 - bbox[1]
    except AttributeError:
        text_width, text_height = d.textsize(text, font=font)
        x = (width - text_width) / 2
        y = (height - text_height) / 2

    d.text((x, y), text, font=font, fill='white', align="center")
    img.save(output_path)
    return output_path

# --- 4. Graph Nodes ---

def setup_assets(state: ReelGenerationState) -> dict:
    """Load video, BGM, and timestamps."""
    print("\n" + "="*60)
    print("--- 1. SETUP ASSETS ---")
    print("="*60)
    
    main_video_clip = VideoFileClip(state["source_video_path"])
    bgm_audio_clip = AudioFileClip(state["bgm_path"], fps=44100)
    
    with open(state["json_path"], "r") as f:
        timestamps = json.load(f)
        
    print(f"✓ Loaded video: {state['source_video_path']}")
    print(f"✓ Loaded BGM: {state['bgm_path']} (Duration: {bgm_audio_clip.duration:.2f}s)")
    print(f"✓ Loaded {len(timestamps)} timestamps")
    
    return {
        "main_video_clip": main_video_clip,
        "bgm_audio_clip": bgm_audio_clip,
        "timestamps": timestamps,
        "current_index": 0,
        "audio_segments": [],
        "video_segment_paths": [],
        "transition_names": [],
        "total_reel_duration": 0.0,
        
        # --- FIX: ADD THESE LINES ---
        "silent_video_path": "temp_silent_video.mp4",
        "final_audio_path": "temp_final_audio.mp3"
    }

def get_llm_plan(state: ReelGenerationState) -> dict:
    """Calls the LLM to get a title and transition for the current clip."""
    print(f"\n--- 2. GETTING LLM PLAN (Segment {state['current_index'] + 1}/{len(state['timestamps'])}) ---")
    
    current_item = state["timestamps"][state["current_index"]]
    description = current_item.get("description", "No description")

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    tools = [GenerateTitleTool, ChooseTransitionTool]
    llm_with_tools = llm.bind_tools(tools)
    
    prompt = f"""
    You are an expert video editor creating a short, dynamic Instagram reel.
    For the upcoming video clip described below, you must perform two actions:
    1. Generate a short, catchy title for a black title screen *before* the clip.
    2. Choose a transition to use *after* the clip to move to the *next* segment.
    
    Clip Description: "{description}"
    
    Call both 'GenerateTitleTool' and 'ChooseTransitionTool' to provide your choices.
    """
    
    previous_messages = state["messages"]
    messages_to_send = previous_messages + [{"role": "user", "content": prompt}]
    
    ai_response = llm_with_tools.invoke(messages_to_send)
    
    return {"messages": [{"role": "user", "content": prompt}, ai_response]}

def execute_video_editing(state: ReelGenerationState) -> dict:
    """
    Creates SILENT video segments and saves them as temp files.
    Extracts audio clips separately.
    """
    print(f"\n--- 3. EXECUTING VIDEO EDITING (Segment {state['current_index'] + 1}) ---")
    
    # --- A. Parse LLM response ---
    ai_message = state["messages"][-1]
    tool_calls = ai_message.additional_kwargs.get("tool_calls", [])
    
    title = "Reel Highlight"
    transition_name = "crossfade"  # Changed default
    
    for call in tool_calls:
        if call["function"]["name"] == "GenerateTitleTool":
            title = json.loads(call["function"]["arguments"]).get("title", title)
        elif call["function"]["name"] == "ChooseTransitionTool":
            transition_name = json.loads(call["function"]["arguments"]).get("transition_name", transition_name)

    print(f"  ✓ AI Title: '{title}'")
    print(f"  ✓ AI Transition: '{transition_name}'")

    # --- B. Get clip data ---
    item = state["timestamps"][state["current_index"]]
    main_video = state["main_video_clip"]
    segment_idx = state["current_index"]
    
    start_t = item.get("start_seconds")
    end_t = item.get("end_seconds")
    
    if start_t is None or end_t is None or start_t >= end_t:
        print(f"⚠ Warning: Invalid timestamps. Skipping segment.")
        return {"current_index": state["current_index"] + 1}

    # --- C. Create SILENT Title Clip and save as video file ---
    title_duration = 2.0
    print(f"  → Creating title video ({title_duration}s)")
    
    temp_image_path = f"temp_title_img_{segment_idx}.png"
    title_image_path = _create_title_image_with_pillow(
        text=title,
        width=main_video.w,
        height=main_video.h,
        output_path=temp_image_path
    )
    
    title_clip = ImageClip(title_image_path).set_duration(title_duration)
    
    # Save title as video file
    title_video_path = f"temp_title_video_{segment_idx}.mp4"
    title_clip.write_videofile(
        title_video_path,
        codec="libx264",
        audio=False,
        fps=main_video.fps,
        verbose=False,
        logger=None
    )
    title_clip.close()
    
    try:
        os.remove(title_image_path)
    except OSError:
        pass
    
    # --- D. Create SILENT Video Clip & Extract Audio ---
    print(f"  → Creating video clip ({start_t}s → {end_t}s)")
    
    # Extract video WITHOUT audio
    video_clip_silent = main_video.subclip(start_t, end_t).without_audio()
    
    # Save as video file
    clip_video_path = f"temp_clip_video_{segment_idx}.mp4"
    video_clip_silent.write_videofile(
        clip_video_path,
        codec="libx264",
        audio=False,
        fps=main_video.fps,
        verbose=False,
        logger=None
    )
    clip_duration = video_clip_silent.duration
    video_clip_silent.close()
    
    # Extract and save audio separately
    audio_clip_path = f"temp_audio_{segment_idx}.mp3"
    print(f"  → Extracting audio to: {audio_clip_path}")
    original_audio = main_video.audio.subclip(start_t, end_t)
    original_audio.write_audiofile(audio_clip_path, fps=44100, codec='mp3', verbose=False, logger=None)
    original_audio.close()
    
    # --- E. Update State ---
    new_video_paths = state["video_segment_paths"] + [title_video_path, clip_video_path]
    new_transitions = state["transition_names"] + [transition_name]
    
    new_audio_segments = state["audio_segments"] + [
        (None, title_duration, True),  # Title segment
        (audio_clip_path, clip_duration, False)  # Clip segment
    ]
    
    new_total_duration = state["total_reel_duration"] + title_duration + clip_duration
    
    print(f"  ✓ Segment saved | Total duration: {new_total_duration:.2f}s")
    
    return {
        "video_segment_paths": new_video_paths,
        "transition_names": new_transitions,
        "audio_segments": new_audio_segments,
        "current_index": state["current_index"] + 1,
        "total_reel_duration": new_total_duration
    }

def finalize_silent_video(state: ReelGenerationState) -> dict:
    """Loads and stitches all video segments with transitions."""
    print("\n" + "="*60)
    print("--- 4. FINALIZING SILENT VIDEO ---")
    print("="*60)
    
    video_paths = state["video_segment_paths"]
    if not video_paths:
        print("❌ Error: No video segments were processed.")
        return {}

    print(f"  → Loading {len(video_paths)} video segments...")
    
    # Load all video clips from files
    clips = [VideoFileClip(path) for path in video_paths]
    
    # Use crossfadein for all transitions (simplest and most reliable)
    print(f"  → Applying crossfade transitions...")
    fade_duration = 0.5
    
    # Apply crossfadein to all clips except the first
    clips_with_fade = [clips[0]]
    for clip in clips[1:]:
        clips_with_fade.append(clip.crossfadein(fade_duration))
    
    # Concatenate with negative padding to create overlap
    final_clip = concatenate_videoclips(clips_with_fade, padding=-fade_duration, method="compose")
    
    silent_video_path = state["silent_video_path"]
    print(f"  → Writing silent video to: {silent_video_path}")
    
    final_clip.write_videofile(
        silent_video_path,
        codec="libx264",
        audio=False,
        fps=(state["main_video_clip"].fps or 24),
        verbose=False,
        logger=None
    )
    
    actual_duration = final_clip.duration
    
    # Close all clips
    for clip in clips:
        clip.close()
    final_clip.close()
    
    # Clean up temp video files
    print(f"  → Cleaning up temp video files...")
    for path in video_paths:
        try:
            os.remove(path)
        except OSError:
            pass
    
    print(f"  ✓ Silent video created! Duration: {actual_duration:.2f}s")
    
    return {"total_reel_duration": actual_duration}

def create_audio_track(state: ReelGenerationState) -> dict:
    """
    Creates the complete audio track with BGM and original audio.
    BGM volume: 0.8 for titles, 0.15 for clips (with smooth fades)
    """
    print("\n" + "="*60)
    print("--- 5. CREATING AUDIO TRACK ---")
    print("="*60)
    
    total_duration = state["total_reel_duration"]
    bgm_audio = state["bgm_audio_clip"]
    audio_segments = state["audio_segments"]
    
    print(f"  → Total reel duration: {total_duration:.2f}s")
    print(f"  → BGM duration: {bgm_audio.duration:.2f}s")
    
    # --- A. Prepare BGM (loop or clip) ---
    if bgm_audio.duration < total_duration:
        print(f"  → BGM is shorter. Looping...")
        num_loops = int(np.ceil(total_duration / bgm_audio.duration))
        bgm_clips = [bgm_audio] * num_loops
        bgm_full = concatenate_audioclips(bgm_clips)
        bgm_prepared = bgm_full.subclip(0, total_duration)
        print(f"    Looped BGM {num_loops} times")
    else:
        print(f"  → BGM is longer. Clipping...")
        bgm_prepared = bgm_audio.subclip(0, total_duration)
    
    # --- B. Create volume envelope for BGM ---
    print(f"  → Creating BGM volume envelope...")
    
    def make_volume_envelope(segments_info, total_dur):
        """
        Creates a volume function: 0.8 for titles, 0.15 for clips, with 0.5s fades.
        """
        fade_duration = 0.5
        
        def volume_function(t):
            # ... (NO CHANGES inside this function) ...
            current_time = 0
            for i, (audio_path, duration, is_title) in enumerate(segments_info):
                segment_end = current_time + duration
                
                if current_time <= t < segment_end:
                    current_vol = 0.8 if is_title else 0.15
                    
                    # Fade-in from previous segment
                    if t < current_time + fade_duration and i > 0:
                        prev_is_title = segments_info[i-1][2]
                        prev_vol = 0.8 if prev_is_title else 0.15
                        fade_progress = (t - current_time) / fade_duration
                        return prev_vol + (current_vol - prev_vol) * fade_progress
                    
                    # Fade-out to next segment
                    elif t > segment_end - fade_duration and i < len(segments_info) - 1:
                        next_is_title = segments_info[i+1][2]
                        next_vol = 0.8 if next_is_title else 0.15
                        fade_progress = (t - (segment_end - fade_duration)) / fade_duration
                        return current_vol + (next_vol - current_vol) * fade_progress
                    
                    else:
                        return current_vol
                
                current_time = segment_end
            
            return 0.15
        
        return volume_function
    
    # --- START OF FIX ---

    # 1. Create the original function that expects a single float
    scalar_volume_envelope = make_volume_envelope(audio_segments, total_duration)

    # 2. Create a vectorized version of it using np.vectorize
    vectorized_volume_envelope = np.vectorize(scalar_volume_envelope)
    
    # 3. Use the new vectorized function in your .fl() call
    bgm_with_envelope = bgm_prepared.fl(lambda gf, t: gf(t) * vectorized_volume_envelope(t)[:, np.newaxis])
    
    # --- C. Composite with original audio clips ---
    print(f"  → Compositing BGM with clip audio...")
    
    audio_clips_to_composite = [bgm_with_envelope]
    
    current_time = 0
    for audio_path, duration, is_title in audio_segments:
        if not is_title and audio_path:
            print(f"    Adding clip audio at {current_time:.2f}s")
            clip_audio = AudioFileClip(audio_path, fps=44100)
            clip_audio_positioned = clip_audio.set_start(current_time)
            audio_clips_to_composite.append(clip_audio_positioned)
        current_time += duration
    
    # --- D. Create final composite audio ---
    print(f"  → Creating final composite audio...")
    final_audio = CompositeAudioClip(audio_clips_to_composite)
    
    # --- E. Write audio file ---
    final_audio_path = state["final_audio_path"]
    print(f"  → Writing audio to: {final_audio_path}")
    final_audio.write_audiofile(
        final_audio_path,
        fps=44100,
        codec='mp3',
        bitrate='192k',
        verbose=False,
        logger=None
    )
    
    final_audio.close()
    print(f"  ✓ Audio track created!")
    
    return {}

def merge_video_and_audio(state: ReelGenerationState) -> dict:
    """Merges the silent video with the complete audio track."""
    print("\n" + "="*60)
    print("--- 6. MERGING VIDEO + AUDIO ---")
    print("="*60)
    
    silent_video = VideoFileClip(state["silent_video_path"])
    final_audio = AudioFileClip(state["final_audio_path"], fps=44100)
    
    print(f"  → Combining video and audio...")
    final_video = silent_video.set_audio(final_audio)
    
    output_path = state["output_path"]
    print(f"  → Writing final reel to: {output_path}")
    
    final_video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        audio_fps=44100,
        fps=silent_video.fps,
        verbose=False,
        logger=None
    )
    
    # --- Cleanup ---
    print(f"\n  → Cleaning up...")
    silent_video.close()
    final_audio.close()
    state["main_video_clip"].close()
    state["bgm_audio_clip"].close()
    
    try:
        os.remove(state["silent_video_path"])
        os.remove(state["final_audio_path"])
        for audio_path, _, is_title in state["audio_segments"]:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        print(f"  ✓ Cleanup complete")
    except Exception as e:
        print(f"  ⚠ Cleanup warning: {e}")
    
    print("\n" + "="*60)
    print(f"✓✓✓ SUCCESS! Reel created: {output_path}")
    print("="*60)
    
    return {}

# --- 5. Conditional Edge ---

def should_continue(state: ReelGenerationState) -> Literal["get_llm_plan", "finalize_silent_video"]:
    """Decides whether to loop or finalize video."""
    if state["current_index"] < len(state["timestamps"]):
        return "get_llm_plan"
    else:
        return "finalize_silent_video"

# --- 6. Build and Run the Graph ---

def run_graph(video_path: str, json_path: str, bgm_path: str, output_path: str):
    """Compiles and runs the LangGraph state machine."""
    
    if "GROQ_API_KEY" not in os.environ:
        print("="*50)
        print("ERROR: GROQ_API_KEY not set.")
        print("="*50)
        return

    workflow = StateGraph(ReelGenerationState)

    workflow.add_node("setup_assets", setup_assets)
    workflow.add_node("get_llm_plan", get_llm_plan)
    workflow.add_node("execute_video_editing", execute_video_editing)
    workflow.add_node("finalize_silent_video", finalize_silent_video)
    workflow.add_node("create_audio_track", create_audio_track)
    workflow.add_node("merge_video_and_audio", merge_video_and_audio)

    workflow.set_entry_point("setup_assets")
    
    workflow.add_edge("setup_assets", "get_llm_plan")
    workflow.add_edge("get_llm_plan", "execute_video_editing")
    
    workflow.add_conditional_edges(
        "execute_video_editing",
        should_continue,
        {
            "get_llm_plan": "get_llm_plan",
            "finalize_silent_video": "finalize_silent_video"
        }
    )
    
    workflow.add_edge("finalize_silent_video", "create_audio_track")
    workflow.add_edge("create_audio_track", "merge_video_and_audio")
    workflow.add_edge("merge_video_and_audio", END)

    app = workflow.compile()

    initial_state = ReelGenerationState(
        source_video_path=video_path,
        json_path=json_path,
        bgm_path=bgm_path,
        output_path=output_path,
        messages=[]
    )
    
    for s in app.stream(initial_state):
        node_name = list(s.keys())[0]
        print(f"\n{'='*60}")
        print(f"NODE COMPLETED: {node_name}")
        print(f"{'='*60}")


# --- Example Usage ---
if __name__ == "__main__":
    
    MY_LONG_VIDEO = "E:/Downloads/Code Basics MCP Server Tutorial.mp4"
    MY_JSON_FILE = "timestamps_reel.json"
    MY_BGM_FILE = "jjk_background_music.mp3"
    MY_OUTPUT_FILE = "ai_generated_reel.mp4"

    if (os.path.exists(MY_LONG_VIDEO) and 
        os.path.exists(MY_JSON_FILE) and 
        os.path.exists(MY_BGM_FILE)):
        
        run_graph(
            video_path=MY_LONG_VIDEO,
            json_path=MY_JSON_FILE,
            bgm_path=MY_BGM_FILE,
            output_path=MY_OUTPUT_FILE,
        )
    else:
        print(f"Error: Files not found.")