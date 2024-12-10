"""Entrypoint to entire Jarvis app"""

import sys
import subprocess
import shlex
import json
import datetime
import requests
from pybars import Compiler
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.prompt import Prompt
import numpy as np
import openwakeword
from openwakeword.model import Model
import speech_recognition as sr
import generative_audio
from dotenv import load_dotenv
import os

load_dotenv()

console = Console()

whisper_model = os.getenv("WHISPER_MODEL")
ollama_model = os.getenv("OLLAMA_MODEL")
ollama_url = os.getenv("OLLAMA_URL")
piper_model = os.getenv("PIPER_MODEL")
audio_response_enabled = os.getenv("AUDIO_RESPONSE_ENABLED").lower() == "true"


def wake_word(stream, chunk) -> bool:
    """upon wake word, break out of endless loop"""

    own_model = Model(
        inference_framework="onnx", wakeword_models=["hey jarvis"], vad_threshold=0.6
    )
    console.log("Listening for wake words...")
    triggered = False
    while not triggered:
        sampled_audio = np.frombuffer(stream.read(chunk), dtype=np.int16)
        prediction = own_model.predict(sampled_audio)
        for _, confidence in prediction.items():
            if confidence > 0.9:
                triggered = True
                break
    return triggered


def install():
    """install assets"""

    # this must be done once
    openwakeword.utils.download_models()


def strip_action_required(user_text: str) -> str | None:
    """Get the action out of text if provided."""

    if "$ActionRequired" in user_text:
        
        # TODO error checking required here.
        command = user_text.split("$ActionRequired")[1].strip()
        if command[::-1][0] == "}":
            return command
    return None


def voice_command_wait():
    """wait for the wakeword before doing analyzing the text"""

    ga = generative_audio.GenerativeAudioService()
    while True:
        r = sr.Recognizer()
        with sr.Microphone(sample_rate=16000) as source:
            wake_word(source.stream, source.CHUNK)
            ga.ding()
            console.log("Say something!")
            audio = r.listen(source)

        try:
            ga.ding()
            user_text = r.recognize_whisper(
                audio, language="english", model=whisper_model
            )
            console.log(f"Whisper thinks you said {user_text}")
            if "$ActionRequired" in user_text:
                command = strip_action_required(user_text)
                user_text = user_text.split("$ActionRequired")[0]
                console.log(f"command {command}")
            llm_response = chat_stream(
                [
                    {"role": "assistant", "content": prompt()},
                    {"role": "user", "content": user_text},
                ],
                console.print,
            )
            ga.generative(llm_response["content"])
        except sr.UnknownValueError:
            console.log("Whisper could not understand audio")
        except sr.RequestError as e:
            console.log(e)


@staticmethod
def prompt() -> str:
    """Generalize prompt to get a persona going."""

    with open("prompt.hbs", "r", encoding="utf-8") as file:
        source = file.read()
    now = datetime.datetime.now()
    context = {"datetime": now.strftime("%B %d, %Y at %I:%M%p")}
    compiler = Compiler()
    template = compiler.compile(source)
    return template(context)


@staticmethod
def chat_stream(messages: list[str], write_out) -> str | None:
    """a list of messages"""

    # https://github.com/ollama/ollama/blob/main/examples/python-simplechat/client.py
    r = requests.post(
        f"{ollama_url}/api/chat",
        json={
            "model": ollama_model,
            "messages": messages,
            "stream": True,
            "keep_alive": 3600,
            "options": {"num_predict": 100}
        },
        stream=True,
        timeout=120,
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
    return None


def is_command_available(command: str) -> bool:
    """ensure command exists before procedding"""
    try:
        # Try to run the command with '--version' (or similar) to check if it's available
        subprocess.run(
            [command, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def cli():
    """cli for this app"""

    audio = audio_response_enabled
    messages = []
    console.print(Panel(Text("CLI Chat", justify="center", style="bold green")))
    while True:
        user_input = Prompt.ask("\n\n[bold blue]You[/bold blue]")
        if not user_input:
            sys.exit()
        elif "clear" in user_input:
            messages = []
            console.print("cleared!")
            continue
        elif "toggle audio" in user_input:
            audio = not audio
            console.print(f"read out audio: {audio}")
            continue

        # redo the original rag prompt for this conversation
        starter = {"role": "assistant", "content": prompt()}
        if len(messages) == 0:
            messages.append(starter)
            messages.append({"role": "user", "content": "set a 5 minute timer"})
            messages.append(
                {
                    "assistant": "user",
                    "content": 'Setting up your countdown clock now... and here is to hoping you can finish whatever it is within five minutes, or I might have to start dingling. $ActionRequired {"service": "timer", "minutes": 5}',
                }
            )
        else:
            messages[0] = starter
        messages.append({"role": "user", "content": user_input})

        def write_out_func(content):
            console.print(content, end="", style="bold yellow")

        message = chat_stream(messages, write_out=write_out_func)
        command = strip_action_required(message["content"])
        if command is not None:
            message["content"] = message["content"].split("$ActionRequired")[0].strip()
        messages.append(message)
        speak(message["content"])


def speak(text: str):
    """Generalized method to send text to tts engine then play it over default speakers.xx"""

    command = [
        "bash",
        "-c",
        f"echo {shlex.quote(text)} | "
        f"piper --model {piper_model}.onnx --output-raw | "
        "ffplay -autoexit -nodisp -hide_banner -loglevel error -f s16le -ar 22050 -ac 1 -i -",
    ]
    subprocess.Popen(command)


if __name__ == "__main__":
    install()
    for command_type in ["ffmpeg", "ffplay", "piper"]:
        is_command_available(command_type)

    cli()
    # print(ollama.list())
    # wait_command()
