# text input, audio file output

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

speech_file_path = "recordings/span.mp3"

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="我不说英语，你呢？"
)

with open(speech_file_path, "wb") as f:
    f.write(response.content)

print("Speech file saved successfully!")
