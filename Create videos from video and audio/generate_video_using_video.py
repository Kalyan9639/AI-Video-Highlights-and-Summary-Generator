import os
from moviepy.editor import VideoFileClip, AudioFileClip
import moviepy.video.fx.all as vfx  # Import the effects module for looping
from typing import Optional, Union

"""
This file contains a function to create a video from a short, looping
video clip and a segment of an audio file using MoviePy.

It re-uses the "audio laundering" technique from our previous version
to ensure audio compatibility.

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
            
    # Fallback for unexpected types
    return None


def create_looped_video_from_audio(
    video_path: str,  # CHANGED: Was image_path
    audio_path: str,
    output_path: str = "output_looped_video.mp4",
    start_sec: Optional[Union[float, str]] = None,
    end_sec: Optional[Union[float, str]] = None,
    fps: int = 24
) -> None:
    """
    Creates a video by looping a shorter video clip to match the
    duration of a specified audio segment.

    Args:
        video_path: Path to the input *video* file (e.g., 7-8s clip).
        audio_path: Path to the input audio file.
        output_path: Path for the final output video.
        start_sec: The start time for the audio (float, int, or "mm:ss").
        end_sec: The end time for the audio (float, int, or "mm:ss").
        fps: Frames per second for the output video.
    """
    print(f"--- Starting Looped Video Method ---")

    # --- 1. Validate Inputs ---
    if not os.path.exists(video_path): # CHANGED
        print(f"Error: Video file not found at {video_path}")
        return
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return

    # --- Clip variables for cleanup ---
    audio_clip = None
    video_clip = None
    laundered_audio_clip = None
    looped_video_clip = None
    final_clip = None
    temp_audio_file_path = "temp_laundered_audio.mp3" # Our reliable fix

    try:
        # --- 2. Audio Pipeline (Our known-good "laundering" method) ---
        print(f"Loading and slicing audio: {audio_path}")
        audio_clip = AudioFileClip(audio_path, fps=44100)

        start_t = _parse_time_to_seconds(start_sec)
        end_t = _parse_time_to_seconds(end_sec)
        start_t = start_t if start_t is not None else 0
        end_t = end_t if end_t is not None else audio_clip.duration

        if start_t >= end_t:
            print(f"Error: Start time ({start_sec}) is after or at end time ({end_sec}).")
            return
        if end_t > audio_clip.duration:
            print(f"Warning: end_sec ({end_sec}) is past audio duration. Using max duration.")
            end_t = audio_clip.duration

        audio_clip = audio_clip.subclip(start_t, end_t)
        print(f"Audio sliced. Target duration: {audio_clip.duration:.2f} seconds.")

        print(f"Writing temporary 'laundered' audio to {temp_audio_file_path}...")
        audio_clip.write_audiofile(
            temp_audio_file_path, 
            codec='mp3', 
            bitrate='192k', 
            fps=44100,
            logger=None
        )
        
        # Load the clean, laundered audio clip
        print(f"Loading 'laundered' audio clip: {temp_audio_file_path}")
        laundered_audio_clip = AudioFileClip(temp_audio_file_path)
        audio_duration = laundered_audio_clip.duration
        print(f"Final audio duration set to: {audio_duration:.2f}s")


        # --- 3. NEW: Load and Loop Video ---
        print(f"Loading input video: {video_path}")
        video_clip = VideoFileClip(video_path)
        print(f"Input video duration: {video_clip.duration:.2f}s")
        
        # Loop the video to match the audio's duration
        print(f"Looping video to match audio duration ({audio_duration:.2f}s)...")
        looped_video_clip = video_clip.fx(vfx.loop, duration=audio_duration)


        # --- 4. Combine and Write File ---
        print("Attaching final audio to looped video...")
        # Set the audio of the (now correctly-durationed) looped video
        final_clip = looped_video_clip.set_audio(laundered_audio_clip)

        # Write the file using our reliable settings
        print(f"Writing final video to {output_path}...")
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='mp3',
            audio_bitrate='192k',
            audio_fps=44100,
            audio=True,
            fps=fps,
            logger=None # Set to None for cleaner output, or remove to see logs
        )
        print(f"Successfully created looped video: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # --- 5. Cleanup ---
        print("Cleaning up all temporary clips and files...")
        if audio_clip:
            audio_clip.close()
        if video_clip:
            video_clip.close()
        if laundered_audio_clip:
            laundered_audio_clip.close()
        if looped_video_clip:
            looped_video_clip.close()
        if final_clip:
            final_clip.close()
        
        if os.path.exists(temp_audio_file_path):
            try:
                os.remove(temp_audio_file_path)
                print(f"Cleaned up temporary audio file: {temp_audio_file_path}")
            except Exception as e:
                print(f"Warning: Could not remove temp file {temp_audio_file_path}: {e}")

        print("--- Looped Video Method Finished ---")


# --- Example Usage ---
#
if __name__ == "__main__":
    MY_VIDEO = "Lord_Shiva_Video.mp4" # e.g., your 7s clip
    MY_AUDIO = "Shiv Shambo Kailashi.mp3"     # e.g., your Suno song

    if os.path.exists(MY_VIDEO) and os.path.exists(MY_AUDIO):
        create_looped_video_from_audio(
            video_path=MY_VIDEO,
            audio_path=MY_AUDIO,
            output_path="lord_shiv_music_video.mp4",
            start_sec="1:07",
            end_sec="1:42"
        )
    else:
        print(f"Please provide valid paths for your video and audio files.")
#

