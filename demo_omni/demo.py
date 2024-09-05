import flask
import base64
import tempfile
import traceback
import threading
import pyaudio
import wave
import queue
from flask import Flask, Response, stream_with_context
from inference import OmniInference

class OmniChatServer(object):
    def __init__(self, ip='0.0.0.0', port=60808, run_app=True,
                 ckpt_dir='./checkpoint', device='cuda:0') -> None:
        server = Flask(__name__)
        self.client = OmniInference(ckpt_dir, device)
        self.client.warm_up()
        server.route("/chat", methods=["POST"])(self.chat)
        
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        if run_app:
            threading.Thread(target=self.record_audio, daemon=True).start()
            server.run(host=ip, port=port, threaded=True)
        else:
            self.server = server

    def record_audio(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        while True:
            if self.is_recording:
                data = stream.read(CHUNK)
                self.audio_queue.put(data)

    def chat(self) -> Response:
        req_data = flask.request.get_json()
        try:
            if req_data.get("start_recording"):
                self.is_recording = True
                return Response("Recording started", status=200)
            
            if req_data.get("stop_recording"):
                self.is_recording = False
                audio_data = b''.join(list(self.audio_queue.queue))
                self.audio_queue.queue.clear()
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    wf = wave.open(f.name, 'wb')
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(audio_data)
                    wf.close()
                    
                    stream_stride = req_data.get("stream_stride", 4)
                    max_tokens = req_data.get("max_tokens", 2048)
                    
                    def generate_audio():
                        for audio_chunk in self.client.run_AT_batch_stream(f.name, stream_stride, max_tokens):
                            yield audio_chunk

                    return Response(stream_with_context(generate_audio()), mimetype="audio/wav")
            
            return Response("Invalid request", status=400)
        except Exception as e:
            print(traceback.format_exc())
            return Response(str(e), status=500)

def create_app():
    server = OmniChatServer(run_app=False)
    return server.server

def serve(ip='0.0.0.0', port=60808):
    OmniChatServer(ip, port=port, run_app=True)

if __name__ == "__main__":
    import fire
    fire.Fire(serve)