import requests
from PIL import Image
import base64
import io
from decord import VideoReader, cpu
from moviepy.editor import VideoFileClip
import time

system_prompt = """Given unstructured noisy text as an amazon product details carefully extract detailed answer strictly in form of 
{'title':"", 'description':'', 'attributes':'', 'price':''} 
where title is product name, 
description is short description of product, 
attributes is a dictionary containing attributes of product. 
if price is not mentioned return price None.
"""

def extract_audio(mp4_path, output_audio_path):
    clip = VideoFileClip(mp4_path)
    clip.audio.write_audiofile(output_audio_path)

def encode_video(video_path):
    MAX_NUM_FRAMES = 16
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    return frames

def process_text(text, question=system_prompt):
    payload = {
        "text": text,
        "question": question
    }
    s = time.time()
    response = requests.post("http://localhost:8000/", json=payload)
    print("text", time.time() - s)

    if response.status_code == 200:
        result = response.json()
        return result['answer']
    return None

def process_img(path, question):
    image = Image.open(path)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")   
    payload = {
        "image": image_base64,
        "question": question
    }
    s = time.time()
    response = requests.post("http://localhost:8000/", json=payload)
    print("image", time.time()-s, response.text)
    if response.status_code == 200:
        result = response.json()
        return result['answer']
    return None

def process_audio(path):
    with open(path, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    payload = {
        "audio": audio_base64
    }
    response = requests.post("http://localhost:8000/", json=payload)
    if response.status_code == 200:
        result = response.json()
        return result['answer']
    return None

def process_video(path, question="Describe all content on this image highly detailed."):
    video_frames = encode_video(path)
    video_b64 = []
    for frame in video_frames:
        buffer = io.BytesIO()
        frame.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        video_b64.append(image_base64)

    payload = {
        "video": video_b64,
        "question": question
    }
    response = requests.post("http://localhost:8000/", json=payload)
    if response.status_code == 200:
        result = response.json()
        answer = process_text(result['answer'])
        images = result['images']
        return answer, images
    return None