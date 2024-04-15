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

def recordaudio(filename,duration=5,fs=44100):
    print('recording...')
    recording=sd.rec(int(duration*fs),samplerate=fs,channels=1)
    sd.wait() #wait for recording to finish
    print('done waiting')
    write(filename,fs,recording) #wave WAV file
    print('written to file')
    result=pipe(filename,generate_kwargs={'language':'en'})
    print(f'finished recording, file saved as {filename}')
    print(result['text'])

def play_audio(file):
    sound=pygame.mixer.Sound(file)
    recordlength=int(sound.get_length()*1000)
    sound.play()
    pygame.time.wait(recordlength)

def make_speech_file(text):
    speech_file_path='recordings/machine.wav'
    response=client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    with open(speech_file_path,"wb") as f:
        f.write(response.content)
    print("Speech file saved successfully!")

def translate_to_english(file):
    audio_file=open(file, "rb")
    translation=client.audio.translations.create(
        model="whisper-1",
        file=audio_file
    )
    print(translation.text)

def machine_turn(text):
    machine_file='recordings/machine.wav'
    if text=='':
        talk='Hello, how are you today?'
    # else:
    #     talk=generate_response()
    make_speech_file(talk)
    play_audio(machine_file)
    print(talk)
    return talk

def human_turn():
    file='recordings/human.wav'
    recordaudio(file)
    talk=translate_to_english(file)
    print(talk)
    return talk

def conversation():
    human_response=''
    x=0
    while x!=1:
        machine_turn(human_response)
        human_response=human_turn()
        x=1