from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download

from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# 🔑 Replace this with your actual token
#hf_token = "your_token"

# ✅ Download and cache the GPT-Neo model
print("Downloading GPT-Neo 2.7B...")
AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", token=hf_token)
AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B", token=hf_token)

# ✅ Download and cache Stable Diffusion
print("Downloading Stable Diffusion v1-4...")
StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", token=hf_token)

# ✅ Download and cache Coqui TTS
print("Downloading Coqui TTS model...")
hf_hub_download(repo_id="coqui-ai/en_ljspeech_tts", filename="model.pth", token=hf_token)

print("✅ All models downloaded and cached locally.")
