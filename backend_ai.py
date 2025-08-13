import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from diffusers import StableDiffusionPipeline
import os
from TTS.api import TTS
from dotenv import load_dotenv

# Load .env (optional)
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Connect to Tor for deep web scraping
def connect_to_tor():
    session = requests.Session()
    session.proxies = {
        'http': 'socks5h://127.0.0.1:9050',
        'https': 'socks5h://127.0.0.1:9050'
    }
    return session

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model from local path or fallback to download
def load_model(model_name, local_dir=None):
    try:
        if local_dir and os.path.isdir(local_dir):
            print(f"🔍 Loading model locally: {local_dir}")
            model = AutoModelForCausalLM.from_pretrained(local_dir)
            tokenizer = AutoTokenizer.from_pretrained(local_dir)
        else:
            print(f"🌐 Downloading model: {model_name}")
            if hf_token:
                model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

# === Load GPT-Neo model locally with fallback ===
gpt_local_path = "models/gpt-neo-2.7B"  # Ensure this folder has your model files
model_name = "EleutherAI/gpt-neo-2.7B"
model, tokenizer = load_model(model_name, local_dir=gpt_local_path)

if not model:
    print("🔁 Falling back to GPT-2...")
    model_name = "gpt2"
    model, tokenizer = load_model(model_name)

# Prepare text generation pipeline
if model:
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device.type == 'cuda' else -1
    )
else:
    text_generator = None
    print("⚠️ Text generation model not loaded!")

# === Load Stable Diffusion remotely (no local path) ===
try:
    print("🌐 Loading Stable Diffusion model remotely...")
    image_generator = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")  # or another model ID
    image_generator.to(device)
except Exception as e:
    print(f"❌ Error loading Stable Diffusion model: {e}")
    image_generator = None

def generate_image(query):
    if not image_generator:
        print("⚠️ Image generator not available.")
        return
    print(f"🖼️ Generating image for: {query}")
    image = image_generator(query).images[0]
    image.show()

# === Load Coqui TTS from local or fallback ===
coqui_model_path = "models/coqui/en_ljspeech_tts/model.pth"
coqui_config_path = "models/coqui/en_ljspeech_tts/config.json"

if os.path.exists(coqui_model_path) and os.path.exists(coqui_config_path):
    coqui_tts = TTS(model_path=coqui_model_path, config_path=coqui_config_path, gpu=device.type == "cuda")
else:
    print("⚠️ Coqui TTS model files not found locally. Attempting to load default model...")
    try:
        coqui_tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")  # loads remote TTS model
    except Exception as e:
        print(f"❌ Error loading Coqui TTS model: {e}")
        coqui_tts = None

def generate_coqui_voice(text, output_path="coqui_output.wav"):
    if not coqui_tts:
        print("⚠️ Coqui TTS not loaded, cannot generate voice.")
        return None
    print(f"🔊 Generating voice for: {text}")
    coqui_tts.tts_to_file(text, output_path)
    print(f"✅ Voice saved at: {output_path}")
    return output_path

# === Surface web scraping ===
def scrape_surface_web(query, search_engine='duckduckgo'):
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }
    results = []

    if search_engine == 'duckduckgo':
        url = f'https://duckduckgo.com/html/?q={query}'
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, 'html.parser')
        links = soup.find_all('a', {'class': 'result__a'})
        results = [link.get_text() for link in links]

    elif search_engine == 'bing':
        url = f'https://www.bing.com/search?q={query}'
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, 'html.parser')
        blocks = soup.find_all('li', {'class': 'b_algo'})
        results = [b.get_text() for b in blocks]

    return results

# === .onion scraping ===
def scrape_onion_site(url):
    try:
        session = connect_to_tor()
        r = session.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        snippets = soup.find_all('pre')
        return [s.get_text() for s in snippets]
    except Exception as e:
        print(f"❌ Onion scrape error: {e}")
        return []

def scrape_ahmia(query):
    try:
        url = f'https://ahmia.fi/search/?q={query}'
        session = connect_to_tor()
        r = session.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        links = soup.find_all('a', {'class': 'result__title'})
        return [link.get_text() for link in links]
    except Exception as e:
        print(f"❌ Ahmia error: {e}")
        return []

# === AI Agent ===
def ai_agent(query):
    if not text_generator:
        print("⚠️ Text generator is not loaded, cannot process queries.")
        return

    q = query.lower()

    if "generate code" in q:
        print("🧠 Generating code...")
        output = text_generator(query)
        print("🧾 Output:", output[0]['generated_text'])

    elif "generate text" in q:
        print("🧠 Generating text...")
        output = text_generator(query, max_length=200)
        print("📝 Output:", output[0]['generated_text'])

    elif "generate image" in q:
        print("🧠 Generating image...")
        generate_image(query)

    elif "generate voice" in q:
        print("🧠 Generating voice...")
        audio = generate_coqui_voice(query)
        if audio:
            print(f"🔉 Voice saved to: {audio}")

    elif "scrape surface" in q:
        print("🌐 Scraping surface web...")
        results = scrape_surface_web(query)
        print("🔎 Results:", results)

    elif "scrape deep web" in q:
        print("🌐 Scraping .onion site...")
        results = scrape_onion_site(query)
        print("🔎 Deep Web Results:", results)

    else:
        print("❓ Query not recognized.")

# === Example usage ===
ai_agent("Generate Python code for a weather prediction system using machine learning.")
ai_agent("A futuristic city skyline during sunset")
ai_agent("Generate a speech about AI and its impact on society.")
ai_agent("Generate voice for a speech about AI and its impact on society.")

# Wrap text generation
def generate_text_func(query, max_length=150):
    if model and text_generator:
        output = text_generator(query, max_length=max_length)
        return output[0]['generated_text']
    else:
        return "Model not loaded."

# Wrap image generation - return PIL Image object instead of showing
def generate_image_func(query):
    if image_generator:
        image = image_generator(query).images[0]
        return image
    else:
        return None

def generate_coqui_voice(text, output_path="coqui_output.wav"):
    print(f"🔊 Generating voice for: {text}")
    try:
        coqui_tts.tts_to_file(text, output_path)
        print(f"✅ Voice saved at: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Voice generation error: {e}")
        raise


# Wrap surface web scraping
def scrape_surface_web_func(query, engine='duckduckgo'):
    return scrape_surface_web(query, search_engine=engine)
