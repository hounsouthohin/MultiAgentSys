from PIL import Image
import requests
from io import BytesIO

img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
response = requests.get(img_url)
pil_img = Image.open(BytesIO(response.content)).convert("RGB")


import torch
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

llava_pipe = pipeline(
    "image-text-to-text",
    model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": pil_img},
        {"type": "text", "text": "What is shown in this image?"}
    ]
}]

res = llava_pipe(messages)
print(res)
