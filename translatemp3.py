# translate foreign input to text output

from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")
client = OpenAI()

audio_file= open("recordings/segment.mp3", "rb")
translation = client.audio.translations.create(
    model="whisper-1",
    file=audio_file
)
print(translation.text)