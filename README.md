# Whisper AI Translation

To use: 
1. Change the `lang` variable in `aitalk.py` to the ISO code of whatever language you want to speak in (preferably a common one since not all are supported).
2. Run `aitalk.py` and the conversation will begin.
3. Talk only after the "recording..." text pops up in the terminal; after which you will have 5 seconds to say your response before it stops recording.

This will begin a voice conversation with an AI bot in a language of your choosing. It uses the OpenAI Whisper model to make translations, and LangChain to generate logical responses to user speech input.

