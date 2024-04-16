# translate foreign input to text output

from dotenv import load_dotenv
from openai import OpenAI
import os

import torch
import sounddevice as sd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from scipy.io.wavfile import write

load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")
client=OpenAI()

device="cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
model_id="openai/whisper-base"
model=AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,torch_dtype=torch_dtype,
    use_safetensors=True
)
model.to(device)
processor=AutoProcessor.from_pretrained(model_id)
pipe=pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

audio_file=open("recordings/human.wav", "rb")
translation=client.audio.translations.create(
    model="whisper-1",
    file=audio_file
)
result=pipe(audio_file,generate_kwargs={'language':'en'})
print(result)