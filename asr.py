import modal
import pathlib
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import whisperx
import torch
from whisperx.utils import get_writer
import os

from modal import (
    App,
    Dict,
    Image,
    Mount,
    NetworkFileSystem,
    Secret,
    asgi_app,
    web_endpoint,
    build,
    enter,
    method
)


app_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "ctranslate2",
        "torch==2.0.0",
        "torchvision==0.15.1",
        "torchaudio==2.0.1",
        "git+https://github.com/m-bain/whisperx.git",
        "uvicorn",
        "fastapi",
        "ffmpeg-python"
    )
)

app = App(
    "whisperx-transcriber",
    image=app_image,
)

# FastAPI app
fastapi_app = FastAPI()

# @app.cls(
#     image=app_image,
# )
# class WhisperXTranscriber:
#     @build()
#     def download_model(self):
#         # Download the model at build time
#         # whisperx.load_model("large-v2", "cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
#         whisperx.load_model("medium", "cuda" if torch.cuda.is_available() else "cpu", compute_type="float32", asr_options={"initial_prompt": "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking. Now, umm.. watch uhh.. this video"})
#
#     @enter()
#     def load_model(self):
#         # Load the model once per container start
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         whisperx.load_model("medium", self.device, compute_type="float32", asr_options={"initial_prompt": "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking. Now, umm.. watch uhh.. this video"})
#         self.align_model, self.metadata = whisperx.load_align_model(language_code="en", device=self.device)
#
#     @method()
#     def transcribe(self, audio_file: str):
#         audio = whisperx.load_audio(audio_file)
#         result = self.model.transcribe(audio, batch_size=16, language="en")
#         result = whisperx.align(result["segments"], self.align_model, self.metadata, audio, self.device, return_char_alignments=False)
#         return result["segments"]
#
# def save_as_vtt(result, audio_file, output_path):
#     result["language"] = "en"
#     vtt_writer = get_writer("vtt", output_path)
#     vtt_writer(
#         result,
#         audio_file,
#         {"max_line_width": None, "max_line_count": None, "highlight_words": False},
#     )
#
# def save_as_txt(result, audio_file, output_path):
#     result["language"] = "en"
#     txt_writer = get_writer("txt", output_path)
#     txt_writer(result, audio_file, {})
#
#
# @fastapi_app.post("/transcribe/")
# async def transcribe(file: UploadFile = File(...)):
#     print(os.path.abspath(".") + "\n")
#     #audio_file = f"/tmp/{file.filename}"
#     #audio_file = f"{file.filename}"
#     audio_file = f"/tmp/{file.filename}"
#     with open(audio_file, "wb") as f:
#         f.write(await file.read())
#
#     transcriber = WhisperXTranscriber()
#     result = await transcriber.transcribe.remote(audio_file)
#     #segments = WhisperXTranscriber().transcribe(audio_file)
#     output_txt_file = "/tmp/transcription.txt"
#     save_as_txt(result, audio_file, output_txt_file)
#
#     return FileResponse(output_txt_file, media_type='text/txt', filename='transcription.txt')
#     #return {"transcription": segments}
#
# @app.function(gpu="T4", image=app_image)
# @asgi_app()
# def fastapi_endpoint():
#     return fastapi_app
#
# # Local entry point for testing
# @app.local_entrypoint()
# def main():
#     import uvicorn
#     uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)


# Load WhisperX model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisperx.load_model("medium", device, compute_type="float32", asr_options={"initial_prompt": "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking. Now, umm.. watch uhh.. this video"}, language="en")
align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
print("Model loaded")

in_progress = Dict.from_name("transcriber-in-progress", create_if_missing=True)

@fastapi_app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    print("Transcribing...")
    audio_file = f"/tmp/{file.filename}"
    with open(audio_file, "wb") as f:
        f.write(await file.read())

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=16,  language="en")
    result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)
    #segments = WhisperXTranscriber().transcribe(audio_file)
    #output_txt_file = "/tmp/transcription.txt"
    output_vtt_file = "/tmp/transcription.vtt"
    f = open("/tmp/transcription.vtt", "x")
    await save_as_vtt(result, "transcription.vtt")
   # save_as_txt(result, audio_file, output_txt_file)

    return FileResponse(output_vtt_file, media_type='text/vtt', filename='transcription.vtt')
    #return {"transcription": result["segments"]}


async def save_as_vtt(result, audio_file, output_path):
    result["language"] = "en"
    vtt_writer = get_writer("vtt", output_path)
    vtt_writer(
        result,
        audio_file,
        {"max_line_width": None, "max_line_count": None, "highlight_words": False},
    )
async def save_as_vtt(result, file_name):
    result["language"] = "en"
    vtt_writer = get_writer("vtt", "/tmp/")
    txt_writer = get_writer("txt", "/tmp/")
    vtt_writer(
        result,
        file_name,
        {"max_line_width": None, "max_line_count": None, "highlight_words": False},
    )
async def save_as_txt(result, audio_file, output_path):
    result["language"] = "en"
    txt_writer = get_writer("txt", output_path)
    txt_writer(result, audio_file, {})

async def save_as_txt(result, file_name):
    result["language"] = "en"
    txt_writer = get_writer("txt", "/tmp/")
    txt_writer(result, file_name, {})

@app.function(gpu="T4", image=app_image)
@asgi_app()
def fastapi_endpoint():
    return fastapi_app

# Local entry point for testing
@app.local_entrypoint()
def main():
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

# FastAPI ASGI app

# @app.function(image=app_image)
# @asgi_app()
# def fastapi_app():
#
#
#     app = FastAPI()
#
#     # Load WhisperX model
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = whisperx.load_model("large-v2", device, compute_type="float16", asr_options={"initial_prompt": "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking. Now, umm.. watch uhh.. this video"})
#     align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
#
#     @web_app.post("/transcribe/")
#     async def transcribe(file: UploadFile = File(...)):
#         audio_file = f"/tmp/{file.filename}"
#         with open(audio_file, "wb") as f:
#             f.write(await file.read())
#
#
#
#         audio = whisperx.load_audio(audio_file)
#         result = model.transcribe(audio, batch_size=16)
#         result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)
#
#         return {"transcription": result["segments"]}
#
#     return app

#jmlaffbdapwhgwovikel

#@app.local_entrypoint()
#def main():
    #import uvicorn
    #uvicorn.run(fastapi_app(), host="0.0.0.0", port=8000)



