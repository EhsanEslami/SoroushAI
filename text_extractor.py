from transformers import pipeline
from dotenv import load_dotenv
import os
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv("/data/EES/SoroushAI/.env")

# Print to verify
print("Hugging Face cache directory:", os.getenv("HF_HUB_CACHE"))

# Load the Persian Whisper model
pipe = pipeline("automatic-speech-recognition", model="MohammadKhosravi/whisper-large-v3-Persian")

# Define the directories
audio_directory = "/data/EES/SoroushAI/input_audio"
output_directory = "/data/EES/SoroushAI/output_text"

# Ensure the directories exist
if not os.path.exists(audio_directory):
    print(f"Error: Directory {audio_directory} does not exist.")
    exit(1)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Get list of MP3 files
mp3_files = [f for f in os.listdir(audio_directory) if f.endswith(".mp3")]

# Process each MP3 file with a progress bar
for filename in tqdm(mp3_files, desc="Processing audio files"):
    audio_path = os.path.join(audio_directory, filename)
    print(f"Processing {audio_path}...")
    
    try:
        # Run the speech-to-text pipeline
        result = pipe(audio_path, return_timestamps=True)
        transcribed_text = result["text"]
        
        # Define output text file path
        output_file = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_transcription.txt")
        
        # Save the output to a text file
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(transcribed_text)
        
        print(f"Transcription saved to {output_file}")
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        continue

print("Processing complete!")
