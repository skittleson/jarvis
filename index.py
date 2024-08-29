from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.prompt import Prompt
import numpy as np
import openwakeword
from openwakeword.model import Model
import speech_recognition as sr
import generative_audio
console = Console()


def wake_word(stream, chunk):
    owwModel = Model(
        inference_framework='onnx',
        wakeword_models=['hey jarvis'],
        vad_threshold=0.6
    )
    console.log("Listening for wake words...")
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


def voice_command_wait():
    ga = generative_audio.GenerativeAudioService()
    while True:
        r = sr.Recognizer()
        with sr.Microphone(sample_rate=16000) as source:
            wake_word(source.stream, source.CHUNK)
            ga.ding()
            console.log("Say something!")
            audio = r.listen(source)

    # recognize speech using whisper
        try:
            ga.ding()
            user_text = r.recognize_whisper(
                audio, model='tiny', language="english")
            console.log(f"Whisper thinks you said {user_text}")
            if '$ActionRequired' in user_text:
                command = user_text.split('$ActionRequired')[1]
                user_text = user_text.split('$ActionRequired')[0]
                console.log(f"command {command}")
            llm_response = chat_stream([{
                'role': 'assistant',
                'content': prompt()
            }, {"role": "user", "content": user_text}])
            ga.generative(llm_response)
        except sr.UnknownValueError:
            console.log("Whisper could not understand audio")
        except sr.RequestError as e:
            console.log(e)


@staticmethod
def prompt() -> str:
    from pybars import Compiler
    import datetime
    with open('prompt.hbs', 'r') as file:
        source = file.read()
    now = datetime.datetime.now()
    context = {
        'datetime': now.strftime("%B %d, %Y at %I:%M%p")
    }
    compiler = Compiler()
    template = compiler.compile(source)
    return template(context)


@staticmethod
def chat_stream(messages: list[str], write_out):

    # https://github.com/ollama/ollama/blob/main/examples/python-simplechat/client.py
    import requests
    import json
    r = requests.post(
        "http://127.0.0.1:11434/api/chat",
        json={"model": 'llama3.1:latest', "messages": messages, "stream": True},
        stream=True
    )
    r.raise_for_status()
    output = ""

    for line in r.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            message = body.get("message", "")
            content = message.get("content", "")
            output += content
            write_out(content)

        if body.get("done", False):
            message["content"] = output
            return message


def cli():
    messages = []
    console.print(
        Panel(Text("CLI Chat", justify="center", style="bold green")))
    while True:
        user_input = Prompt.ask("\n\n[bold blue]You[/bold blue]")
        if not user_input:
            exit()
        elif 'command:clear' in user_input:
            messages = []
            console.print('cleared!')
            continue
        elif 'command:audio' in user_input:
            audio = not audio
            console.print(f'read out audio: {audio}')
            continue

        # redo the original rag prompt for this conversation
        starter = {
            'role': 'assistant',
            'content': prompt()
        }
        if len(messages) == 0:
            messages.append(starter)
        else:
            messages[0] = starter
        messages.append({"role": "user", "content": user_input})

        def write_out_func(content):
            console.print(content, end='', style="bold yellow")
        message = chat_stream(messages, write_out=write_out_func)
        messages.append(message)
        # if audio:
        #     ga.generative(message['content'])


if __name__ == '__main__':
    cli()
    # print(ollama.list())
    # wait_command()
    # install()
