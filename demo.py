# Following pip packages need to be installed:
# !pip install git+https://github.com/huggingface/transformers sentencepiece datasets
import transformers
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset
proxies={'http': '127.0.0.1:1080', 'https': '127.0.0.1:1080'}

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts", proxies=proxies)
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts", proxies=proxies)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", proxies=proxies)

inputs = processor(text="Hello, my dog is cute", return_tensors="pt")

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

sf.write("speech.wav", speech.numpy(), samplerate=16000)
