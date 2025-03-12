# SoroushAI

SoroushAI is a Large Language Model (LLM) Agent designed to interpret and explain the meanings of poems from the great Persian poet Mevlana Rumi. The interpretations are provided based on the insights of Abdolkarim Soroush, a renowned Iranian Rumi expert.

## Project Overview
This project leverages the Retrieval-Augmented Generation (RAG) technique to retrieve relevant content related to a given poem query. The database consists of transcriptions from audio sessions on Mathnavi by Abdolkarim Soroush. 

To achieve accurate transcription, we employ a fine-tuned version of OpenAI's Whisper v3, optimized for the Persian language. The model used for transcription can be found here:

[Whisper Large v3 - Persian](https://huggingface.co/MohammadKhosravi/whisper-large-v3-Persian)

## Features
- Retrieves relevant interpretations of Rumi’s poems
- Uses a fine-tuned Persian language model for accurate transcriptions
- Provides insights based on Abdolkarim Soroush’s teachings
- Interactive chatbot interface powered by Streamlit

## Installation
To set up and run the project, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/SoroushAI.git
   cd SoroushAI
   ```

2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

Before running the application, configure your API keys:

1. Open the `.env.example` file and add your keys:
   ```sh
   OPENAI_API_KEY=""
   TAVILY_API_KEY=""
   ```
2. Rename the file to `.env`.

1. Start the chatbot application:
   ```sh
   streamlit run chatbot/app.py
   ```
2. Open the provided link in your browser to interact with the AI bot.

