# take input directly from microphone

import torch
import accelerate
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
from scipy.io.wavfile import write
import ffmpeg

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-base"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype,
    use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
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

def recordaudio(filename,duration=3,fs=44100):
    print('recording...')
    recording=sd.rec(int(duration*fs),samplerate=fs,channels=1)
    sd.wait() #wait for recording to finish
    print('done waiting')
    write(filename,fs,recording) #wave WAV file
    print('written to file')
    result=pipe(filename,generate_kwargs={'language':'en'})
    print(f'finished recording, file saved as {filename}')
    print(result)
    print(result['text'])

recordaudio('/Users/jacintogomez/PycharmProjects/whisper/recordings/test.wav')