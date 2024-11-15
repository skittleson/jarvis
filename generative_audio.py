import simpleaudio as sa
import os
import subprocess
import psutil
import os
import signal
from threading import Thread
import subprocess

class GenerativeAudioService:
    """
    Create and manage playback with text to speech audio files
    """

    def __init__(self) -> None:
        # todo. check if piper is installed
        pass

    @staticmethod
    def play_stream(stream: bytes) -> None:
        """play wav stream using ffplay"""
        for proc in psutil.process_iter():
            if "ffplay" in proc.name():
                print(f"kill process {proc.pid}")
                os.kill(proc.pid, signal.SIGTERM)
                break
        command = "ffplay -autoexit -nodisp -hide_banner -loglevel error -fs -".split(
            ' ')

        def run(c, s): return subprocess.run(c, input=s, shell=True)
        (Thread(target=run, args=(command, stream))).start()

    @staticmethod
    def shell(command: str, distribution: str = "Ubuntu-22.04"):
        """
        Command to start an interactive shell in WSL
        """
        
        shell = ["wsl", "-d", distribution, "/bin/bash"]
        p = subprocess.run(shell, input=command, capture_output=True, text=True, timeout=10)
        return {'stdout': p.stdout, 'stderr': p.stderr, 'returncode': p.returncode}


    def generative(self, text: str):
        import time
        import shlex
        temp_filename = 'generative.wav'
        text = text.replace('\r\n','').replace('\n','')

        # en_US-lessac-medium
        # en_US-libritts-high
        # piper can stream to stdout using output_raw
        command = f"echo {shlex.quote(text)} | piper --model en_GB-northern_english_male-medium.onnx --output_file {temp_filename}"
        if os.name == 'posix':
            # pip install piper-tts
            # echo 'This sentence is spoken first. This sentence is synthesized while the first sentence is spoken.' |   piper --model en_GB-northern_english_male-medium.onnx --output-raw |   aplay -r 22050 -f S16_LE -t raw -
            print('install piper-tts')
        elif os.name == 'nt':
            GenerativeAudioService.shell(command)
        while not os.path.isfile(temp_filename):
            time.sleep(1)
        GenerativeAudioService.play(temp_filename)
        os.remove(temp_filename)

    @staticmethod
    def play(filename: str, blocking: bool = True):
        wave_obj = sa.WaveObject.from_wave_file(filename)
        play_obj = wave_obj.play()
        if blocking:
            play_obj.wait_done()

    @staticmethod
    def ding():
        GenerativeAudioService.play('ding.wav', False)