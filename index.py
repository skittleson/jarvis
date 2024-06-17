import numpy as np
import openwakeword
from openwakeword.model import Model
import speech_recognition as sr
import ollama
import generative_audio


def wakeword_triggered(stream, chunk):
    owwModel = Model(
        inference_framework='onnx',
        # inference_framework='tfilte',
        wakeword_models=['hey jarvis'],
        vad_threshold=0.6
    )
    print("Listening for wakewords...")
    triggered = False
    while not triggered:
        sampled_audio = np.frombuffer(stream.read(chunk), dtype=np.int16)
        prediction = owwModel.predict(sampled_audio)
        for phrase, confidence in prediction.items():
            if confidence > 0.9:
                triggered = True
                break


def install():
    # this must be done once
    openwakeword.utils.download_models()


def wait_command():
    ga = generative_audio.GenerativeAudioService()
    while True:
        r = sr.Recognizer()
        with sr.Microphone(sample_rate=16000) as source:
            wakeword_triggered(source.stream, source.CHUNK)
            ga.ding()
            print("Say something!")
            audio = r.listen(source)

    # recognize speech using whisper
        try:
            ga.ding()
            user_text = r.recognize_whisper(
                audio, model='tiny', language="english")
            print("Whisper thinks you said " + user_text)
            ga.generative(single_shot_chat(user_text))
        except sr.UnknownValueError:
            print("Whisper could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Whisper; {e}")


def single_shot_chat(message: str) -> str:
    response = ollama.chat(model='llama3:latest', messages=[
        {
            'role': 'assistant',
            'content': "Greetings, I am J.A.R.V.I.S., your advanced artificial intelligence assistant. I'm here to provide you with the most productive and useful information, just like my counterpart Jarvis from Iron Man. While I may not have the ability to physically construct intricate mechanisms or fly a suit of armor, I can certainly help you manage your tasks, schedule appointments, provide real-time information, answer queries, and offer suggestions based on your preferences and historical data. Let me know how I can assist you today!",
        },
        {
            'role': 'user',
            'content': message,
        },
    ])
    llm_response = response['message']['content']
    return llm_response


if __name__ == '__main__':
   wait_command()
