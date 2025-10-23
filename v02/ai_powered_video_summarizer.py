import os
from typing import TypedDict, Annotated
from pathlib import Path
import subprocess
import tempfile
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from groq import Groq

load_dotenv()
os.environ['GROQ_API_KEY']= os.getenv("GROQ_API_KEY")

# State definition for the graph
class VideoSummarizerState(TypedDict):
    video_path: str
    audio_path: str
    transcription: str
    vector_store: Chroma
    summary: str
    error: str


def extract_audio_node(state: VideoSummarizerState) -> VideoSummarizerState:
    """
    Node 1: Extract audio from video file
    """
    try:
        video_path = state["video_path"]
        
        # Validate video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Validate video format
        video_ext = Path(video_path).suffix.lower()
        if video_ext not in ['.mp4', '.mkv']:
            raise ValueError(f"Unsupported video format: {video_ext}. Only .mp4 and .mkv are supported.")
        
        # Create temporary audio file path
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio_path = temp_audio.name
        temp_audio.close()
        
        # Extract audio using ffmpeg
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '16000',  # 16kHz sample rate for Whisper
            '-ac', '1',  # Mono channel
            '-y',  # Overwrite output file
            audio_path
        ]
        
        print(f"Extracting audio from {video_path}...")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")
        
        print(f"Audio extracted successfully to {audio_path}")
        
        return {
            **state,
            "audio_path": audio_path,
            "error": ""
        }
        
    except Exception as e:
        print(f"Error in extract_audio_node: {str(e)}")
        return {
            **state,
            "error": f"Audio extraction failed: {str(e)}"
        }


def transcribe_audio_node(state: VideoSummarizerState) -> VideoSummarizerState:
    """
    Node 2: Transcribe audio using Whisper via Groq
    """
    try:
        # Check for previous errors
        if state.get("error"):
            return state
        
        audio_path = state["audio_path"]
        
        # Initialize Groq client
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        client = Groq(api_key=groq_api_key)
        
        print(f"Transcribing audio using Whisper...")
        
        # Transcribe audio using Whisper
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(audio_path, audio_file.read()),
                model="whisper-large-v3",
                response_format="text",
                language="en"
            )
        
        transcription_text = transcription if isinstance(transcription, str) else transcription.text
        
        print(f"Transcription completed. Length: {len(transcription_text)} characters")
        
        # Save transcription to text file
        transcription_file = "transcription.txt"
        with open(transcription_file, "w", encoding="utf-8") as f:
            f.write(transcription_text)
        print(f"Transcription saved to {transcription_file}")
        
        # Clean up temporary audio file
        try:
            os.remove(audio_path)
            print(f"Temporary audio file removed: {audio_path}")
        except Exception as e:
            print(f"Warning: Could not remove temporary audio file: {e}")
        
        return {
            **state,
            "transcription": transcription_text,
            "error": ""
        }
        
    except Exception as e:
        print(f"Error in transcribe_audio_node: {str(e)}")
        return {
            **state,
            "error": f"Transcription failed: {str(e)}"
        }


def store_in_vectordb_node(state: VideoSummarizerState) -> VideoSummarizerState:
    """
    Node 3: Store transcription in Chroma vector database
    """
    try:
        # Check for previous errors
        if state.get("error"):
            return state
        
        transcription = state["transcription"]
        
        if not transcription:
            raise ValueError("No transcription text available to store")
        
        print("Splitting text into chunks...")
        
        # Split text using RecursiveCharacterTextSplitter
        # Optimal chunk size: 500 characters with 50 overlap for good semantic retention
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(transcription)
        print(f"Text split into {len(chunks)} chunks")
        
        # Create Document objects
        documents = [Document(page_content=chunk, metadata={"chunk_id": i}) 
                     for i, chunk in enumerate(chunks)]
        
        print("Initializing embeddings model...")
        
        # Initialize HuggingFace embeddings (no API token required)
        embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1",
            model_kwargs={'device': 'cpu', 'trust_remote_code': True}
        )
        
        print("Creating Chroma vector store...")
        
        # Create Chroma vector store (in-memory, not persistent)
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="video_transcription"
        )
        
        print(f"Successfully stored {len(documents)} documents in vector database")
        
        return {
            **state,
            "vector_store": vector_store,
            "error": ""
        }
        
    except Exception as e:
        print(f"Error in store_in_vectordb_node: {str(e)}")
        return {
            **state,
            "error": f"Vector database storage failed: {str(e)}"
        }


def generate_summary(vector_store: Chroma, query: str = "Summarize the main points") -> str:
    """
    Generate summary from vector store using Groq
    """
    try:
        # Retrieve top 10 most relevant chunks
        print("Retrieving relevant chunks from vector database...")
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        # relevant_docs = retriever.get_relevant_documents(query)
        relevant_docs = retriever.invoke(query)
        
        print(f"Retrieved {len(relevant_docs)} relevant chunks")
        
        # Combine retrieved chunks
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Initialize Groq LLM
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        llm = ChatGroq(
            api_key=groq_api_key,
            model_name="openai/gpt-oss-20b",
            temperature=0.3
        )
        
        # Create detailed prompt for summary generation
        prompt = f"""Based on the following transcription content, please generate a detailed summary report in simple and easy-to-understand terms.

The summary should:
1. Provide a comprehensive overview of the main topics discussed
2. Highlight key points and important details
3. Organize information in a logical flow
4. Use clear and simple language
5. Include specific examples or details mentioned in the content

Transcription Content:
{context}

Please provide a detailed summary report:"""
        
        print("Generating summary using GPT model...")
        
        # Generate summary
        response = llm.invoke(prompt)
        summary = response.content
        
        print(f"Summary generated successfully. Length: {len(summary)} characters")
        
        # Save summary to markdown file
        summary_file = "summary_report.md"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("# Video Summary Report\n\n")
            f.write(summary)
        
        print(f"Summary saved to {summary_file}")
        
        return summary
        
    except Exception as e:
        raise RuntimeError(f"Summary generation failed: {str(e)}")


# Build the LangGraph workflow
def build_video_summarizer_graph():
    """
    Build the LangGraph workflow for video summarization
    """
    workflow = StateGraph(VideoSummarizerState)
    
    # Add nodes
    workflow.add_node("extract_audio", extract_audio_node)
    workflow.add_node("transcribe_audio", transcribe_audio_node)
    workflow.add_node("store_vectordb", store_in_vectordb_node)
    
    # Define edges
    workflow.set_entry_point("extract_audio")
    workflow.add_edge("extract_audio", "transcribe_audio")
    workflow.add_edge("transcribe_audio", "store_vectordb")
    workflow.add_edge("store_vectordb", END)
    
    return workflow.compile()


def process_video(video_path: str) -> dict:
    """
    Main function to process video and generate summary
    """
    try:
        print(f"\n{'='*60}")
        print(f"Starting Video Summarization Process")
        print(f"{'='*60}\n")
        
        # Initialize state
        initial_state = {
            "video_path": video_path,
            "audio_path": "",
            "transcription": "",
            "vector_store": None,
            "summary": "",
            "error": ""
        }
        
        # Build and run the graph
        graph = build_video_summarizer_graph()
        
        # Execute the graph
        final_state = graph.invoke(initial_state)
        
        # Check for errors
        if final_state.get("error"):
            print(f"\n{'='*60}")
            print(f"ERROR: {final_state['error']}")
            print(f"{'='*60}\n")
            return final_state
        
        # Generate summary from vector store
        print(f"\n{'='*60}")
        print("Generating Summary Report")
        print(f"{'='*60}\n")
        
        summary = generate_summary(final_state["vector_store"])
        final_state["summary"] = summary
        
        print(f"\n{'='*60}")
        print("Process Completed Successfully!")
        print(f"{'='*60}\n")
        print(f"✅ Transcription saved to: transcription.txt")
        print(f"✅ Summary report saved to: summary_report.md")
        
        return final_state
        
    except Exception as e:
        error_msg = f"Process failed: {str(e)}"
        print(f"\n{'='*60}")
        print(f"ERROR: {error_msg}")
        print(f"{'='*60}\n")
        return {"error": error_msg}


# Main execution
if __name__ == "__main__":
    # Example usage
    video_file_path = "E:/Images, Videos and Word-Doc's/AI Powered Internal Chatbot - Made with Clipchamp (1).mp4"  # Replace with your video path
    
    # Check if GROQ_API_KEY is set
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: Please set GROQ_API_KEY environment variable")
        print("Example: export GROQ_API_KEY='your-api-key-here'")
    else:
        result = process_video(video_file_path)
        
        if not result.get("error"):
            print("\n" + "="*60)
            print("SUMMARY PREVIEW:")
            print("="*60)
            print(result["summary"][:500] + "..." if len(result["summary"]) > 500 else result["summary"])