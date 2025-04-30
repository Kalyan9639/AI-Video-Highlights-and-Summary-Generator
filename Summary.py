from google.genai import Client,types

with open(r"C:\Users\madda\Desktop\Hackattack\transcription.txt", "r") as f:
    transcript = f.read()

# with open(r"C:\Users\madda\Desktop\Hackattack\video_captions.txt", "r") as f:
#     video_captions = f.read()


client = Client(api_key="YOUR_GEMINI_API_KEY")

def split_transcript(text, chunk_size_minutes=2, words_per_minute=150):
    words = text.split()
    chunk_size = chunk_size_minutes * words_per_minute
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


chunks = split_transcript(transcript)
summaries = [] 

for i, chunk in enumerate(chunks):
    instruction = f"""
        You are an AI summarizer. 
        You are a good communicator who can answer questions from the given text and provide information in clear and concise manner.
        You are given with a audio transcript of a video and visual information about the video.
        Transcript chunk (Minutes {i*2} to {(i+1)*2}): {chunk}

        Utilize the full audio transcript and visual information to generate key points and highlights from the video.
        When generating video highlights or key moments, process the audio transcript in 2-minute chunks and provide a description for each chunk.
        Don't give me additional information like, here is the summary or here is the highlights; just use it to generate the summary. 
    """

    chat = client.chats.create(
        model="gemini-2.0-flash",
        history=[],
        config=types.GenerateContentConfig(
            temperature=0.5,
            max_output_tokens=2048,
            top_p=0.8,
            system_instruction=instruction
        )
    )
    response = chat.send_message("Generate a highlight for this part of the video.")
    summaries.append(f"**{i*2:02d}:{0:02d} - {(i+1)*2:02d}:{0:02d}**\n{response.text}\n")


    full_summary = "\n".join(summaries)
    print(full_summary)

with open("video_summary.txt", "w") as f:
    f.write(full_summary)
