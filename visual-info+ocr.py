from transformers import BlipProcessor, BlipForConditionalGeneration
import easyocr
import cv2
import torch
from PIL import Image



# Initialize BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Add more languages if needed

# Keywords that indicate frames worth OCR'ing
trigger_keywords = [
    'computer', 'webpage', 'screen', 'presentation'
]

def extract_frames_and_caption(video_path, output_txt_file):
    """
    Extracts frames from a video, captions them using BLIP, and performs OCR on relevant frames,
    with separate processing intervals.

    Args:
        video_path (str): Path to the video file.
        output_txt_file (str): Path to the output text file.
    """
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    blip_frame_rate = int(fps * 30)  # Process for BLIP every 30 seconds
    ocr_frame_rate = int(fps * 60)    # Process for OCR every 60 seconds

    with open(output_txt_file, "a", encoding="utf-8") as output_file:
        frame_count = 0
        frame_number = 0
        blip_frame_count = 0 #keep track of blip frame count
        ocr_frame_count = 0 #keep track of ocr frame count

        while True:
            ret, frame = video.read()
            if not ret:
                break

            if frame_number % blip_frame_rate == 0:
                frame_count += 1
                blip_frame_count += 1 #increment blip frame count
                frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Use BLIP to get a caption
                inputs = processor(frame_image, return_tensors="pt")
                with torch.no_grad():
                    output = model.generate(**inputs)
                caption = processor.decode(output[0], skip_special_tokens=True)

                # Write the BLIP result
                timestamp = frame_number / float(fps)
                output_file.write(f"Frame {blip_frame_count} (at {timestamp:.2f} seconds):\n{caption}\n")

            if frame_number % ocr_frame_rate == 0:
                ocr_frame_count += 1 #increment ocr frame count
                # Check if the caption warrants OCR
                if any(keyword in caption.lower() for keyword in trigger_keywords): #use the caption from the closest blip processing
                    results = reader.readtext(frame)
                    ocr_text = '\n'.join([text for _, text, _ in results])
                    output_file.write(f"OCR Extracted Text:\n{ocr_text.strip()}\n")

            if frame_number % blip_frame_rate == 0 or frame_number % ocr_frame_rate == 0:
                output_file.write("\n") #write new line only if either blip or ocr is processed

            frame_number += 1

        video.release()
        print(f"Frame captions and OCR text saved to {output_txt_file}")

# Paths
video_path = r"E:\Downloads\saveinsta.cc_1080p-nvidia-ceo-jensen-huang-and-the-2-trillion-company-powering-todays-ai-60-minutes.mp4"
output_txt_file = "video_captions.txt"

# Run the function
extract_frames_and_caption(video_path, output_txt_file)
