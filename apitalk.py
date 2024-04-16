import pygame
import torch
import sounddevice as sd
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from scipy.io.wavfile import write
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")
client=OpenAI()

llm=ChatOpenAI()
prompt=ChatPromptTemplate.from_messages([
    ("system", "Respond conversationally to the given input, in whatever language it is given in"),
    ("user", "{input}")
])
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

translator=ChatPromptTemplate.from_messages([
    ("system", "Translate this to the ISO language specified at the end in parentheses"),
    ("user", "{input}")
])
translate=translator|llm|output_parser

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
pygame.init()

def recordaudio(filename,duration=5,fs=44100):
    print('recording...')
    recording=sd.rec(int(duration*fs),samplerate=fs,channels=1)
    sd.wait()
    #print('done waiting')
    write(filename,fs,recording)
    #print('written to file')
    result=pipe(filename,generate_kwargs={'language':lang})
    #print(f'finished recording, file saved as {filename}')
    #print(result['text'])
    return result['text']

def play_audio(file):
    sound=pygame.mixer.Sound(file)
    recordlength=int(sound.get_length()*1000)
    sound.play()
    pygame.time.wait(recordlength)

def make_speech_file(speech_file_path,text):
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
    return translation.text

def native(lang,text):
    trans=translate.invoke({'input':text+' (language:'+lang+')'})
    print(trans)
    return trans

def generate_response(text):
    generation=chain.invoke({'input':text})
    print(generation)
    return generation

def machine_turn(text):
    machine_file='recordings/machine.wav'
    if text=='':
        eng='Hello, how are you today?'
        talk=native(lang,eng)
    else:
        talk=generate_response(text)
    make_speech_file(machine_file,talk)
    play_audio(machine_file)
    eng=translate_to_english(machine_file)
    print('Computer: '+talk+' ('+eng+')')

def human_turn():
    file='recordings/human.wav'
    talk=recordaudio(file)
    eng=translate_to_english(file)
    print('Me: '+talk+' ('+eng+')')
    return talk

def conversation():
    human_response=''
    x=0
    while x!=5:
        machine_turn(human_response)
        human_response=human_turn()
        x+=1
    pygame.quit()

#begin
lang='zh'
conversation()