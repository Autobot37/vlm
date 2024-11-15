from starlette.requests import Request
from ray import serve
from ray.serve.handle import DeploymentHandle
from io import BytesIO
import base64
import os
import requests
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM
import open_clip
from PIL import Image
import torch
from moviepy.editor import VideoFileClip
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from faster_whisper import WhisperModel
import ollama

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 4, "num_gpus": 1})
class App:
    def __init__(self):
 
        self.llm_tokenizer = AutoTokenizer.from_pretrained("singhshiva/qwen3bi-AWQ")
        self.sampling_params = SamplingParams(temperature=0.3, top_p=0.8, repetition_penalty=1.05, max_tokens=1024)
        self.llm_model = LLM(model="singhshiva/qwen3bi-AWQ", gpu_memory_utilization=0.6)
        print("LLM Loaded")

        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k_augreg')
        self.clip_model.eval().cuda()
        print("Clip loaded")

        self.whisper_model = WhisperModel("medium", device="cuda", compute_type="float16")
        print("Whisper Loaded")

    def pil_image_to_base64(self, img):
        buffered = BytesIO()
        img.save(buffered, format="PNG")  
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_base64

    def image_search(self, text, images, topk=3):
        image_tensors = torch.stack([self.preprocess(img).to("cuda") for img in images])
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensors).cpu()
            text_tensor = self.clip_model.encode_text(open_clip.tokenize([text]).to("cuda")).cpu()
        similarities = (text_tensor @ image_features.T).squeeze(0)
        topk_indices = similarities.topk(topk).indices.tolist()
        return [images[i] for i in topk_indices]

    def process_img(self, image, question="Describe this image detailed."):
        decoded_bytes = base64.b64decode(image)
        os.makedirs("processing", exist_ok=True)
        image_path = os.path.join("processing", "decoded_image.png")
        with open(image_path, "wb") as f:
            f.write(decoded_bytes)
        response = ollama.chat(
            model="moondream",
            messages=[{"role": "user", "content": "Describe this image.", "images": [image_path]}],
        )
        return response["message"]["content"]

    def process_audio(self, audio):
        audio_bytes = base64.b64decode(audio)
        os.makedirs("processing", exist_ok=True)
        audio_path = os.path.join("processing", "temp_audio.wav")
        with open(audio_path, 'wb') as f:
            f.write(audio_bytes)
        segments, info = self.whisper_model.transcribe(audio_path, beam_size=5)
        text = ""
        for segment in segments:
            text += segment.text
        return text

    def process_video(self, video_base64, question="Describe all content on this image highly detailed."):
        frames = []
        os.makedirs("processing", exist_ok=True)
        for i, frame_base64 in enumerate(video_base64):
            frame_bytes = base64.b64decode(frame_base64)
            frame_path = os.path.join("processing", f"frame_{i}.png")
            with open(frame_path, "wb") as f:
                f.write(frame_bytes)
            frame = Image.open(frame_path)
            frames.append(frame)
        prompts = [question] * len(frames)
        answers = ""
        for i in range(0, len(frames), 4):
            answer = self.vlm_model.batch_answer(
                images=frames[i:min(i+4, len(frames))],
                prompts=prompts[i:min(i+4, len(frames))],
                tokenizer=self.vlm_tokenizer,
            )
            answers += ''.join(answer)

        similar_imgs = self.image_search(answers, frames)
        return answers, similar_imgs

    def process_text(self, raw_text, question):
        messages = [
            {"role": "system", "content": question},
            {"role": "user", "content": raw_text}
        ]
        text = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        outputs = self.llm_model.generate([text], self.sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
        return generated_text

    async def __call__(self, http_request: Request) -> str:
        json = await http_request.json()
        return_json = {"answer": ""}
        if "image" in json and json["image"] is not None:
            return_json["answer"] = self.process_img(json["image"], json["question"])
            return return_json

        if "text" in json and json["text"] is not None:
            return_json["answer"] = self.process_text(json["text"], json["question"])
            return return_json

        if "audio" in json and json["audio"] is not None:
            return_json["answer"] = self.process_audio(json["audio"])
            return return_json

        if "video" in json and json["video"] is not None:
            out = self.process_video(json["video"], json["question"])
            return_json["answer"] = out[0]
            return_json["images"] = [self.pil_image_to_base64(img) for img in out[1]]  # Convert all images to base64
            return return_json

        return None


app = App.bind()