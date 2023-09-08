# Following pip packages need to be installed:
# !pip install git+https://github.com/huggingface/transformers sentencepiece datasets
import transformers
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

def readInput():
    f=open("input.txt", "r", encoding="utf8")
    list = f.readlines()  # 直接将文件中按行读到list里，效果与方法2一样
    f.close()  # 关
    ret = ",".join(list)
    print(f'xxx {ret}')
    return ret

def run():
        
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts", )
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts", )
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", )
    input = readInput()
    print(f"input=>{input}")
    inputs = processor(text=input, return_tensors="pt")

    # load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    sf.write("speech.wav", speech.numpy(), samplerate=16000)


run()