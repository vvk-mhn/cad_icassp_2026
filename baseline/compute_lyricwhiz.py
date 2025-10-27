import importlib.util
import json
import logging
import tempfile
from pathlib import Path
import hydra
import numpy as np
import torch
import whisper
from omegaconf import DictConfig
from tqdm import tqdm
from clarity.utils.file_io import read_jsonl, write_jsonl, write_signal
from clarity.utils.signal_processing import resample
from recipes.cad_icassp_2026.baseline.transcription_scorer import SentenceScorer
from recipes.cad_icassp_2026.baseline.shared_predict_utils import load_mixture
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

PROMPT_PATH = "/scratch/work/mohanv1/cadenza_stuff/Cadenza/clarity/recipes/cad_icassp_2026/baseline/LyricWhiz/code/prompts_files/pick_best_prompt.txt"
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

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

def _import_run_whisper_module(lyricwhiz_code_dir: Path):
    runner_path = lyricwhiz_code_dir / "run_whisper.py"
    if not runner_path.exists():
        logger.warning(f"run_whisper.py not found at {runner_path}")
        return None
    spec = importlib.util.spec_from_file_location("lyricwhiz_runner", str(runner_path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def compute_correctness_with_models(signal, sr, reference, whisper_model, panns_model, lyricwhiz_cfg, runner_module=None, deepseek_model=None, tokenizer=None, base_prompt=None):
    scorer = SentenceScorer(contractions_file=str(Path(__file__).parent / "contractions.csv"))

    if signal.ndim == 2 and signal.shape[0] == 2 and signal.shape[1] != 2:
        signal = signal.T
    mono = np.mean(signal, axis=1) if signal.ndim == 2 else signal

    tmp = tempfile.NamedTemporaryFile(prefix="lyricwhiz_", suffix=".flac", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    write_signal(tmp_path, mono, sr, floating_point=False)

    hypothesis = ""
    try:
        if runner_module is not None:
            try:
                tags, probs = runner_module.tag_audio(panns_model, str(tmp_path))
                if runner_module.is_vocal(tags, probs, threshold=float(lyricwhiz_cfg.baseline.get("threshold", 0.0))):
                    res = whisper.transcribe(whisper_model, str(tmp_path), language=lyricwhiz_cfg.baseline.get("language", "en"))
                    hypothesis = res.get("text", "")
                    if deepseek_model:
                        hypothesis = refine_with_deepseek(deepseek_model, tokenizer, base_prompt, hypothesis)
            except Exception as e:
                logger.warning(f"Runner failed for {tmp_path.name}: {e}")
        else:
            res = whisper.transcribe(whisper_model, str(tmp_path), language="en")
            hypothesis = res.get("text", "")
            if deepseek_model:
                hypothesis = refine_with_deepseek(deepseek_model, tokenizer, base_prompt, hypothesis)
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass

    results = scorer.score([reference], [hypothesis])
    total_words = results.substitutions + results.deletions + results.hits
    return 0.0 if total_words == 0 else float(results.hits) / float(total_words)

@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_compute_lyricwhiz(cfg: DictConfig):
    assert cfg.baseline.name == "lyricwhiz"
    dataroot = Path(cfg.data.cadenza_data_root) / cfg.data.dataset
    dataset_filename = dataroot / "metadata" / f"{cfg.split}_metadata.json"
    with dataset_filename.open("r", encoding="utf-8") as fp:
        records = json.load(fp)

    lyricwhiz_code_dir = Path("/scratch/work/mohanv1/cadenza_stuff/Cadenza/clarity/recipes/cad_icassp_2026/baseline/LyricWhiz/code")
    runner_module = _import_run_whisper_module(lyricwhiz_code_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper.load_model(cfg.baseline.get("lyricwhiz_model", "base"), device=device)
    panns_model = runner_module.load_panns(device=device) if runner_module else None

    deepseek_model, tokenizer, base_prompt = load_deepseek()

    results_file = Path(f"{cfg.data.dataset}.{cfg.split}.{cfg.baseline.system}.jsonl")
    for record in tqdm(records):
        signal_name = record["signal"]
        signal_arr, sr = load_mixture(dataroot, record, cfg)
        if sr != cfg.data.sample_rate:
            signal_arr = resample(signal_arr, sr, cfg.data.sample_rate)
        sr = cfg.data.sample_rate

        try:
            score = compute_correctness_with_models(signal_arr, sr, record.get("prompt", ""), whisper_model, panns_model, cfg, runner_module, deepseek_model, tokenizer, base_prompt)
        except Exception as e:
            logger.exception(f"Failed for {signal_name}: {e}")
            score = 0.0

        score = float(max(0.0, min(1.0, score)))
        write_jsonl(str(results_file), [{"signal": signal_name, "lyricwhiz": score}])

    logger.info("✅ Finished LyricWhiz scoring with DeepSeek enhancement.")

if __name__ == "__main__":
    run_compute_lyricwhiz()
