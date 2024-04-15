import pygame
import torch
import accelerate
import sounddevice as sd
import ffmpeg
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from scipy.io.wavfile import write
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")
client=OpenAI()

device="cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
model_id="openai/whisper-base"
model=AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype,
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
pygame.init()

def recordaudio(filename,duration=3,fs=44100):
    print('recording...')
    recording=sd.rec(int(duration*fs),samplerate=fs,channels=1)
    sd.wait() #wait for recording to finish
    print('done waiting')
    write(filename,fs,recording) #wave WAV file
    print('written to file')
    result=pipe(filename,generate_kwargs={'language':'en'})
    print(f'finished recording, file saved as {filename}')
    print(result['text'])

def translate_to_english(file):
    audio_file=open(file, "rb")
    translation=client.audio.translations.create(
        model="whisper-1",
        file=audio_file
    )
    print(translation.text)

def machine_turn():
    pass

def human_turn():
    recordaudio('human.wav')
    return translate_to_english('human.wav')

def conversation:
    human_response=''
    while True:
        machine_turn()
        human_response=human_turn()