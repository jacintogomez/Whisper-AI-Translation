# audio file input, transcription output

from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")
client = OpenAI()

audio_file=open("recordings/speech.mp3", "rb")
transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file
)
print(transcription.text)