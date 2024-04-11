# text input, audio file output

from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")
client = OpenAI()

speech_file_path = Path(__file__).parent / "span.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input="hola, cómo estás"
)

response.stream_to_file(speech_file_path)