import os
import json
from moviepy.editor import VideoFileClip, concatenate_videoclips
from typing import Optional, Union

"""
This file contains a function to create a "summary video" by
extracting and combining clips from a longer video based on
a JSON timestamp file.

REQUIREMENTS:
-------------------------------------------------------------------------------
    pip install moviepy
-------------------------------------------------------------------------------
"""

def _parse_time_to_seconds(time_input: Optional[Union[float, str, int]]) -> Optional[float]:
    """
    Helper function to parse time strings (like "mm:ss" or "hh:mm:ss") 
    or numbers into total seconds.
    """
    if time_input is None:
        return None
    
    if isinstance(time_input, (int, float)):
        return float(time_input)

    if isinstance(time_input, str):
        try:
            parts = time_input.split(':')
            parts.reverse() # Start from seconds
            
            total_seconds = 0.0
            if len(parts) > 0:
                total_seconds += float(parts[0]) # seconds
            if len(parts) > 1:
                total_seconds += int(parts[1]) * 60 # minutes
            if len(parts) > 2:
                total_seconds += int(parts[2]) * 3600 # hours
            
            return total_seconds
        except ValueError:
            print(f"Warning: Could not parse time string '{time_input}'. Defaulting to 0.")
            return 0.0
        except Exception as e:
            print(f"Warning: Error parsing time '{time_input}' ({e}). Defaulting to 0.")
            return 0.0
            
    return None


def create_summary_video(
    video_path: str,
    json_path: str,
    output_path: str = "summary_video.mp4"
) -> None:
    """
    Creates a summary video by extracting and concatenating clips
    from a source video based on timestamps in a JSON file.

    Args:
        video_path: Path to the long-form input video file.
        json_path: Path to the JSON file with timestamps.
                   (e.g., important_timestamps.json)
        output_path: Path for the final summary video.
    """
    print(f"--- Starting Video Summary Method ---")

    # --- 1. Validate Inputs ---
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    # --- Variables for cleanup ---
    main_video = None
    final_clip = None
    clips_list = []

    try:
        # --- 2. Load Main Video ---
        print(f"Loading main video: {video_path}")
        main_video = VideoFileClip(video_path)

        # --- 3. Load and Parse JSON Timestamps ---
        print(f"Loading timestamps from: {json_path}")
        with open(json_path, 'r') as f:
            timestamps = json.load(f)

        if not isinstance(timestamps, list):
            print(f"Error: JSON file does not contain a list of timestamps.")
            return

        # --- 4. Extract Clips ---
        print("Extracting clips based on timestamps...")
        for i, item in enumerate(timestamps):
            # Use .get() for safety
            start_str = item.get("timestamp_start")
            end_str = item.get("timestamp_end")

            # We'll use our reliable parser
            start_t = _parse_time_to_seconds(start_str)
            end_t = _parse_time_to_seconds(end_str)

            if start_t is None or end_t is None:
                print(f"Warning: Skipping item {i+1} due to missing timestamps.")
                continue
            
            if start_t >= end_t:
                print(f"Warning: Skipping item {i+1}, start time ({start_str}) is after end time ({end_str}).")
                continue
            
            if end_t > main_video.duration:
                print(f"Warning: Item {i+1} end time ({end_str}) exceeds video duration. Clipping to end.")
                end_t = main_video.duration

            print(f"  Extracting clip {i+1}: {start_str} -> {end_str}")
            # This is the core logic: subclip the main video.
            # The audio is included automatically.
            clip = main_video.subclip(start_t, end_t)
            clips_list.append(clip)

        if not clips_list:
            print("Error: No valid clips were extracted. Aborting.")
            return

        # --- 5. Combine and Write Final Video ---
        print(f"Combining {len(clips_list)} clips into final video...")
        
        # This is the key function to join all the clips
        final_clip = concatenate_videoclips(clips_list)
        
        print(f"Writing final summary video to {output_path}...")
        # We'll use the same reliable write settings we found earlier
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='mp3',      # Use mp3 for reliability
            audio_bitrate='192k',
            audio_fps=44100,        # Standardize audio sample rate
            fps=main_video.fps or 24 # Use original FPS or default to 24
        )
        
        print(f"Successfully created summary video: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # --- 6. Cleanup ---
        print("Cleaning up clips...")
        if main_video:
            main_video.close()
        if final_clip:
            final_clip.close()
        for clip in clips_list:
            clip.close()
        print("--- Video Summary Method Finished ---")


# --- Example Usage ---
#
if __name__ == "__main__":
    # MY_LONG_VIDEO = "E:/Images, Videos and Word-Doc's/AI Powered Internal Chatbot - Made with Clipchamp (1).mp4"
    MY_LONG_VIDEO = "E:/Downloads/Live Day 4- Discussing Decision Tree And Ensemble Machine Learning Algorithms.mp4"
    MY_JSON_FILE = "important_timestamps.json" # The file you provided

    if os.path.exists(MY_LONG_VIDEO) and os.path.exists(MY_JSON_FILE):
        create_summary_video(
            video_path=MY_LONG_VIDEO,
            json_path=MY_JSON_FILE,
            output_path="my_final_summary.mp4"
        )
    else:
        print(f"Please provide valid paths for your video and JSON files.")
#
