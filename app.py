import time
import subprocess

import base64
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
from main import process_audio, process_img, process_text, process_video, extract_audio, encode_video, extract_audio
import uvicorn
import ast

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

class MediaRequest(BaseModel):
    caption: str
    media_url: str
    media_type: str

class LitAPI:
    def setup(self):
        self.system_prompt = """Given unstructured noisy text as an amazon product details carefully extract detailed answer strictly in form of 
                            {'title':"", 'description':'', 'attributes':'', 'price':''} 
                            where title is product name, 
                            description is short description of product, 
                            attributes is a dictionary containing attributes of product. 
                            if price is not mentioned return price None.
                            """
    def download_media(self, media_url, save_path):
        response = requests.get(media_url, stream=True)
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        return save_path

    def decode_request(self, data):
        caption = data.caption
        media_url = data.media_url
        media_type = data.media_type

        if media_type == 'IMAGE':
            path = self.download_media(media_url, 'image.png')
            return path, caption
        elif media_type == 'VIDEO':
            path = self.download_media(media_url, 'video.mp4')
            return path, caption
        return None, caption

    def predict(self, path, caption):
        if path.endswith('.png') or path.endswith('.jpg'):
            raw_text = process_img(path, "describe highly detailed.")
            structured_dict = process_text(caption + raw_text, self.system_prompt)
            # payload = {
            #     "text": raw_text,
            #     "question": self.system_prompt
            # }
            # response = requests.post("http://localhost:8000/", json=payload)
            # if response.status_code == 200:
            #     result = response.json()
            #     structured_dict = result['answer']
            print(structured_dict)
            try:
                structured_dict = ast.literal_eval(structured_dict)
            except:
                return None
            with open(path, "rb") as img_file:
                structured_dict["image"]  = base64.b64encode(img_file.read()).decode('utf-8')
            return structured_dict

        elif path.endswith('.mp4'):
            audio_path = path.rsplit(".", 1)[0] + ".mp3"
            extract_audio(path, audio_path)
            raw_text = process_audio(audio_path)
            video_data = process_video(path)
            raw_text = raw_text + video_data[0] if video_data is not None else raw_text
            structured_dict = process_text(caption + raw_text, self.question)
            try:
                structured_dict = ast.literal_eval(structured_dict)
            except:
                return None
            frames = encode_video(path)
            similar_imgs = video_data[1] 
            return structured_dict
        return None

api = LitAPI()
api.setup()

@app.post("/predict")
async def predict(request: MediaRequest):
    path, caption = api.decode_request(request)
    print(path, caption)
    if path:
        result = api.predict(path, caption)
        if result:
            return {"result": result}
        return {"error": "Failed to process media"}, 500
    return {"error": "Invalid media type"}, 400

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8190)
