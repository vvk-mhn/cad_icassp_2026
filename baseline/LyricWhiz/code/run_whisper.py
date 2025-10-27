import argparse
import os
import json
from tqdm import tqdm
import librosa
import numpy as np
import torch
from shared_predict_utils import load_mixture
from omegaconf import OmegaConf
from pathlib import Path
from panns_inference import AudioTagging, labels
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer

###############################
# CONFIG FOR DEEPSEEK
###############################
PROMPT_PATH = "/scratch/work/mohanv1/cadenza_stuff/Cadenza/clarity/recipes/cad_icassp_2026/baseline/LyricWhiz/code/prompt_files/pick_best_prompt.txt"
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1"

###############################
# AUDIO HELPERS
###############################
def find_audios_from_metadata(metadata_path, dataroot, cfg_path):
    cfg = OmegaConf.load(cfg_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    audio_files = []
    for record in records:
        signal_name = record["signal"]
        flac_path = Path(dataroot) / cfg.split / "signals" / f"{signal_name}.flac"
        if flac_path.exists():
            audio_files.append(str(flac_path))
        else:
            print(f"⚠️ Missing: {flac_path}")
    return audio_files

###############################
# PANNs
###############################
def load_panns(device='cuda'):
    from panns_inference import AudioTagging
    return AudioTagging(checkpoint_path=None, device=device)

@torch.no_grad()
def tag_audio(model, audio_path):
    (audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
    audio = audio[None, :30*32000]
    (clipwise_output, _) = model.inference(audio)
    return get_audio_tagging_result(clipwise_output[0])

def get_audio_tagging_result(clipwise_output):
    sorted_indexes = np.argsort(clipwise_output)[::-1]
    tags, probs = [], []
    for k in range(10):
        tag = np.array(labels)[sorted_indexes[k]]
        prob = clipwise_output[sorted_indexes[k]]
        tags.append(tag)
        probs.append(float(prob))
    return tags, probs

def is_vocal(tags, probs, threshold=0.08):
    pos_tags = {'Speech', 'Singing', 'Rapping'}
    for tag, prob in zip(tags, probs):
        if tag in pos_tags and prob > threshold:
            return True
    return False

###############################
# WHISPER + DEEPSEEK
###############################
def load_whisper(model="large"):
    return whisper.load_model(model, in_memory=True)

def load_deepseek():
    print("🚀 Loading DeepSeek model...")
    tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL)
    model = AutoModelForCausalLM.from_pretrained(DEEPSEEK_MODEL, torch_dtype="auto", device_map="auto")
    with open(PROMPT_PATH, "r") as f:
        base_prompt = f.read().strip()
    return model, tokenizer, base_prompt

def refine_with_deepseek(model, tokenizer, base_prompt, text):
    full_prompt = f"{base_prompt}\n\nRaw lyrics from Whisper:\n{text}\n\nPlease clean, structure, and correct these lyrics."
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024)
    cleaned_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return cleaned_text.strip()

###############################
# MAIN PIPELINE
###############################
def transcribe_and_save(whisper_model, panns_model, deepseek_model, tokenizer, base_prompt, args):
    audio_files = find_audios_from_metadata(
        metadata_path="/scratch/work/mohanv1/cadenza_stuff/Cadenza/cadenza_data/metadata/train_metadata.json",
        dataroot="/scratch/work/mohanv1/cadenza_stuff/Cadenza/cadenza_data",
        cfg_path="/scratch/work/mohanv1/cadenza_stuff/Cadenza/clarity/recipes/cad_icassp_2026/baseline/configs/config.yaml"
    )

    if args.n_shard > 1:
        print(f'Processing shard {args.shard_rank} of {args.n_shard}')
        audio_files.sort()
        audio_files = audio_files[
            args.shard_rank * len(audio_files) // args.n_shard :
            (args.shard_rank + 1) * len(audio_files) // args.n_shard
        ]

    for file in tqdm(audio_files):
        output_file = os.path.join(args.output_dir, os.path.relpath(file, args.input_dir))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        try:
            tags, probs = tag_audio(panns_model, file)
            if args.threshold == 0.0 or is_vocal(tags, probs, threshold=args.threshold):
                res = whisper.transcribe(whisper_model, file, language=args.language, initial_prompt=args.prompt)
                whisper_text = res.get("text", "")
                cleaned_text = refine_with_deepseek(deepseek_model, tokenizer, base_prompt, whisper_text)

                result = {
                    "raw_whisper": whisper_text,
                    "refined_lyrics": cleaned_text,
                    "tags_with_probs": [{"tag": t, "prob": p} for t, p in zip(tags, probs)]
                }

                with open(output_file + '.json', 'w') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
            else:
                print(f'No vocal in {file}')
        except Exception as e:
            print(e)
            continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='large')
    parser.add_argument('--prompt', type=str, default='lyrics: ')
    parser.add_argument('--language', type=str, default='en')
    parser.add_argument('--input_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--n_shard', type=int, default=1)
    parser.add_argument('--shard_rank', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.00)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    whisper_model = load_whisper(args.model)
    panns_model = load_panns()
    deepseek_model, tokenizer, base_prompt = load_deepseek()

    transcribe_and_save(whisper_model, panns_model, deepseek_model, tokenizer, base_prompt, args)

if __name__ == '__main__':
    main()
