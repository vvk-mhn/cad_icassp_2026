import logging
import json
import re
import os
from pathlib import Path
from difflib import SequenceMatcher

import hydra
import torch
import numpy as np
from omegaconf import DictConfig, open_dict
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

# Import Cadenza utilities
try:
    from clarity.utils.file_io import read_jsonl, write_jsonl, write_signal
    from clarity.utils.signal_processing import resample
    from recipes.cad_icassp_2026.baseline.shared_predict_utils import load_mixture
except ImportError:
    print("Could not import Clarity/Cadenza tools. Check your path.")

log = logging.getLogger(__name__)

def calculate_similarity(hypothesis: str, reference: str) -> float:
    """Computes text similarity (0.0 to 1.0) using SequenceMatcher."""
    if not hypothesis or not reference:
        return 0.0
    return SequenceMatcher(None, hypothesis.lower(), reference.lower()).ratio()

def extract_score_from_reasoning(text: str) -> float:
    """
    Extracts a float score (0.0 to 1.0) from the LLM's chat response.
    """
    matches = re.findall(r"\b0\.\d+\b|\b1\.0\b|\b0\b", text)
    if matches:
        return float(matches[-1])
    
    matches_int = re.findall(r"\b([0-9]|10)\s*\/", text)
    if matches_int:
        return float(matches_int[0]) / 10.0
        
    return 0.5 

def convert_inputs_to_half(inputs, device):
    """
    Recursively convert all floating point tensors in a dict to half precision.
    This ensures complete dtype consistency.
    """
    converted = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if v.dtype in [torch.float32, torch.float64]:
                converted[k] = v.to(device=device, dtype=torch.float16)
            else:
                converted[k] = v.to(device=device)
        elif isinstance(v, dict):
            converted[k] = convert_inputs_to_half(v, device)
        elif isinstance(v, list):
            converted[k] = [
                item.to(device=device, dtype=torch.float16) if isinstance(item, torch.Tensor) and item.dtype in [torch.float32, torch.float64]
                else item.to(device=device) if isinstance(item, torch.Tensor)
                else item
                for item in v
            ]
        else:
            converted[k] = v
    return converted

@hydra.main(config_path="configs", config_name="config", version_base=None)
def compute_audio_flamingo(cfg: DictConfig):
    with open_dict(cfg):
        if 'baseline' in cfg and 'reference' not in cfg.baseline:
            log.warning("Config missing 'baseline.reference', injecting default 'processed'")
            cfg.baseline.reference = "processed"

    log.info(f"Initializing Audio Flamingo 3: {cfg.baseline.model_id}")

    # 1. Load Model with explicit dtype handling
    processor = AutoProcessor.from_pretrained(cfg.baseline.model_id)
    
    # SOLUTION 1: Try loading with bfloat16 if available (better stability)
    # Check if your GPU supports bfloat16
    try:
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            cfg.baseline.model_id, 
            device_map="auto",
            torch_dtype=torch.bfloat16,  # Try bfloat16 first
        )
        log.info("Model loaded with bfloat16")
        target_dtype = torch.bfloat16
    except:
        # Fallback to float16
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            cfg.baseline.model_id, 
            device_map="auto",
            torch_dtype=torch.float16,
        )
        log.info("Model loaded with float16")
        target_dtype = torch.float16
    
    model.eval()
    
    # Get the actual device the model is on
    model_device = next(model.parameters()).device
    log.info(f"Model device: {model_device}, dtype: {target_dtype}")

    # 2. Setup Paths
    dataroot = Path(cfg.data.cadenza_data_root) / cfg.data.dataset
    dataset_filename = dataroot / "metadata" / f"{cfg.split}_metadata.json"
    
    with open(dataset_filename, "r") as f:
        records = json.load(f)

    output_file = f"{cfg.data.dataset}.{cfg.split}.{cfg.baseline.system}_dual.jsonl"
    
    existing_results = read_jsonl(output_file) if os.path.exists(output_file) else []
    processed_ids = {r["signal"] for r in existing_results}
    
    records_to_process = [r for r in records if r["signal"] not in processed_ids]
    log.info(f"Processing {len(records_to_process)} signals...")

    for record in records_to_process:
        signal_id = record["signal"]
        reference_text = record["prompt"]
        temp_wav_path = Path(f"temp_{signal_id}.wav")

        try:
            # 3. Load Audio & Save to Temp WAV
            mixture_signal, sr = load_mixture(dataroot, record, cfg)
            
            if sr != cfg.data.sample_rate:
                mixture_signal = resample(mixture_signal, sr, cfg.data.sample_rate)

            # Convert stereo to mono
            if mixture_signal.ndim == 2:
                mixture_signal = np.mean(mixture_signal, axis=1)
                
            write_signal(temp_wav_path, mixture_signal, cfg.data.sample_rate, floating_point=True)
            abs_audio_path = str(temp_wav_path.resolve())

            device_type = 'cuda' if model_device.type == 'cuda' else 'cpu'
            
            with torch.no_grad():
                # Create autocast context based on device and model dtype
                if device_type == 'cuda':
                    autocast_ctx = torch.autocast(device_type=device_type, dtype=target_dtype)
                else:
                    autocast_ctx = torch.autocast(device_type=device_type, enabled=False)
                
                with autocast_ctx:
                    inputs_trans = processor.apply_transcription_request(
                        audio=abs_audio_path
                    )
                    
                    # SOLUTION 3: Deep conversion of all inputs
                    inputs_trans = convert_inputs_to_half(inputs_trans, model_device)
                    
                    out_trans = model.generate(**inputs_trans, max_new_tokens=500)
                    transcription = processor.batch_decode(
                        out_trans[:, inputs_trans["input_ids"].shape[1]:], 
                        skip_special_tokens=True, 
                        strip_prefix=True
                    )[0]
                    
                    score_correctness = calculate_similarity(transcription, reference_text)

                    prompt_text = (
                        "You will hear an audio clip of accompanied singing. "
                        "Rate the intelligibility of the lyrics on a continuous scale from 0.0 to 1.0, where: "
                        "0.0 = no words can be understood; "
                        "0.2 = only occasional isolated words are understood; "
                        "0.5 = about half of the words are understandable; "
                        "0.8 = most words are clear, only brief sections are hard to parse; "
                        "1.0 = all words are perfectly clear throughout. "
                        "Focus only on the lead vocal (ignore backing vocals unless they carry the main lyric). "
                        "Rate the entire clip, not just the easiest sections. "
                        "If there are no discernible lyrics (instrumental, scat, non-lexical vocalizations), output 0.0. "
                        "Use these anchors to calibrate your rating; avoid random or constant values. "
                        "Reply with a single number only (a float between 0.0 and 1.0) with two decimals and no extra text."
                    )

                    
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {"type": "audio", "path": abs_audio_path},
                            ],
                        }
                    ]

                    inputs_chat = processor.apply_chat_template(
                        conversation,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True
                    )
                    
                    # Deep conversion of all inputs
                    inputs_chat = convert_inputs_to_half(inputs_chat, model_device)

                    out_chat = model.generate(**inputs_chat, max_new_tokens=50)
                    reasoning_text = processor.batch_decode(
                        out_chat[:, inputs_chat["input_ids"].shape[1]:], 
                        skip_special_tokens=True
                    )[0]
                    
                    score_reasoning = extract_score_from_reasoning(reasoning_text)

            log.info(f"Signal {signal_id}: Trans={score_correctness:.2f}, Reason={score_reasoning:.2f}")

            # 4. Save Result
            result_entry = {
                "signal": signal_id,
                "af_transcription_score": score_correctness,
                "af_reasoning_score": score_reasoning,
                "af_raw_reasoning": reasoning_text 
            }
            
            write_jsonl(output_file, [result_entry])

        except Exception as e:
            log.error(f"Failed on {signal_id}: {e}")
            import traceback
            log.error(traceback.format_exc())
            write_jsonl(output_file, [{
                "signal": signal_id, 
                "af_transcription_score": 0.0,
                "af_reasoning_score": 0.0,
                "error": str(e)
            }])

        finally:
            if temp_wav_path.exists():
                temp_wav_path.unlink()

if __name__ == "__main__":
    compute_audio_flamingo()