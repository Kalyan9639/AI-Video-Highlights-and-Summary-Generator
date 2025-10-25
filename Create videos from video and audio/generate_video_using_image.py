import os
from moviepy.editor import ImageClip, AudioFileClip
from typing import Optional, Union

"""
This file contains a function to create a video from a single image
and a segment of an audio file using MoviePy.

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


def create_video_with_moviepy(
    image_path: str,
    audio_path: str,
    output_path: str = "output_moviepy.mp4",
    start_sec: Optional[Union[float, str]] = None,
    end_sec: Optional[Union[float, str]] = None,
    fps: int = 24
) -> None:
    """
    Creates a video from an image and audio using MoviePy.

    This all-in-one Python method handles audio slicing,
    image-to-video conversion, and merging internally.

    Args:
        image_path: Path to the input image file.
        audio_path: Path to the input audio file.
        output_path: Path for the final output video.
        start_sec: The start time (in seconds as float/int, or "mm:ss", "hh:mm:ss" as str).
                     If None, starts from the beginning (0).
        end_sec: The end time (in seconds as float/int, or "mm:ss", "hh:mm:ss" as str).
                   If None, goes to the end of the audio.
        fps: Frames per second for the output video.
    """
    print(f"--- Starting MoviePy Method ---")

    # --- 1. Validate Inputs ---
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return

    audio_clip = None
    video_clip = None
    laundered_audio_clip = None # NEW: For our laundered clip
    temp_audio_file_path = "temp_laundered_audio.mp3" # NEW: Path for laundered file

    try:
        # --- 2. Load and Slice Audio ---
        print(f"Loading and slicing audio: {audio_path}")
        audio_clip = AudioFileClip(audio_path, fps=44100)

        # Parse time inputs
        start_t = _parse_time_to_seconds(start_sec)
        end_t = _parse_time_to_seconds(end_sec)

        # Determine start and end
        start_t = start_t if start_t is not None else 0
        end_t = end_t if end_t is not None else audio_clip.duration

        if start_t >= end_t:
            print(f"Error: Start time ({start_sec}) is after or at end time ({end_sec}).")
            return
        
        if end_t > audio_clip.duration:
            print(f"Warning: end_sec ({end_sec}) is past the audio duration ({audio_clip.duration}s). Using audio duration as end time.")
            end_t = audio_clip.duration

        audio_clip = audio_clip.subclip(start_t, end_t)
        print(f"Audio sliced. New duration: {audio_clip.duration:.2f} seconds.")

        # --- 3. NEW: Manually "Launder" the Audio ---
        # We write the sliced clip to a new file to fix any corruption
        # or incompatibilities from the source (Suno AI) file.
        print(f"Writing temporary 'laundered' audio to {temp_audio_file_path}...")
        audio_clip.write_audiofile(
            temp_audio_file_path, 
            codec='mp3', 
            bitrate='192k', 
            fps=44100,
            logger=None # Suppress logs for this temp write
        )
        print("Temporary audio written successfully.")

        # --- 4. Create Video and Load Laundered Audio ---
        print(f"Loading image: {image_path}")
        video_clip = ImageClip(image_path)

        print(f"Loading 'laundered' audio clip: {temp_audio_file_path}")
        # We load the file we JUST wrote. This clip is clean and reliable.
        laundered_audio_clip = AudioFileClip(temp_audio_file_path)

        # --- 5. Combine and Write File ---
        # Set the duration of the image clip to match the LAUNDERED audio
        final_clip = video_clip.set_duration(laundered_audio_clip.duration).set_audio(laundered_audio_clip)

        # --- Debugging Step ---
        if final_clip.audio is None:
            print("CRITICAL ERROR: Audio object is None before writing.")
            return
        else:
            print(f"Debug: Audio clip duration before write: {final_clip.audio.duration:.2f}s")

        # Set FPS and write the final video file
        print(f"Writing final video to {output_path}...")
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='mp3',      # Use mp3
            audio_bitrate='192k',
            audio_fps=44100,
            audio=True,             # Explicitly set audio to True
            
            # REMOVED: temp_audiofile and remove_temp, as they caused the 'ext' bug
            
            fps=fps,
            # Logs are still on from last time
        )
        print(f"Successfully created video: {output_path}")

    except Exception as e:
        print(f"An error occurred with moviepy: {e}")

    finally:
        # --- 6. Updated Cleanup ---
        # Close all file handles
        if audio_clip:
            audio_clip.close()
        if video_clip:
            video_clip.close()
        if laundered_audio_clip: # NEW: Close the laundered clip
            laundered_audio_clip.close()
        
        # NEW: Manually remove our temporary file
        if os.path.exists(temp_audio_file_path):
            try:
                os.remove(temp_audio_file_path)
                print(f"Cleaned up temporary audio file: {temp_audio_file_path}")
            except Exception as e:
                print(f"Warning: Could not remove temp file {temp_audio_file_path}: {e}")

        print("--- MoviePy Method Finished ---")


# --- Example Usage ---
#
# To use this function, you would call it from another Python script
# or from a terminal after ensuring you have your own media files.
#
# if __name__ == "__main__":
#     # Ensure you have "my_image.png" and "my_audio.mp3"
#     # or change these paths to your own files.
#     MY_IMAGE = "path/to/your/image.png"
#     MY_AUDIO = "path/to/your/audio.mp3"
#
#     if os.path.exists(MY_IMAGE) and os.path.exists(MY_AUDIO):
#         create_video_with_moviepy(
#             image_path=MY_IMAGE,
            # audio_path=MY_AUDIO,
#             output_path="my_final_video.mp4",
#             start_sec="1:08",  # Start at 1 min 8 sec
#             end_sec="1:42"   # End at 1 min 42 sec
#         )
#     else:
#         print(f"Please provide valid paths for your image and audio files.")
#





# --- Example Usage ---
#
# To use this function, you would call it from another Python script
# or from a terminal after ensuring you have your own media files.
#
if __name__ == "__main__":
    # Ensure you have "my_image.png" and "my_audio.mp3"
    # or change these paths to your own files.
    MY_IMAGE = "Lord_Shiva.png"
    MY_AUDIO = "Shiv Shambo Kailashi.mp3"

    if os.path.exists(MY_IMAGE) and os.path.exists(MY_AUDIO):
        create_video_with_moviepy(
            image_path=MY_IMAGE,
            audio_path=MY_AUDIO,
            output_path="my_final_video.mp4",
            start_sec="1:08",  # Start at 1 min 8 sec
            end_sec="1:42"   # End at 1 min 42 sec
        )
    else:
        print(f"Please provide valid paths for your image and audio files.")
#


