import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_TEST = REPO_ROOT / "results" / "raw" / "test"
RESULTS = REPO_ROOT / "results"

LLAMA = "Llama-3.1-8B-Instruct"
TRACKS = {"t1": "t1", "t2": "t2", "t3": "t3/de"}
TRACK_LABELS = {"t1": "Track 1", "t2": "Track 2", "t3": "Track 3"}

METHODS = {
    "prompting": "english",
    "diffmean_l20": "diffmean-source-L20-c1",
    "diffmean_l16": "diffmean-source-L16-c0p75",
    "tuned": "tuned",
    "adaptive": "adaptive-gate-w32-t0-l0-source-L16-c0p75",
    "tuned_adaptive": "tuned-adaptive",
}
METHOD_LABELS = {
    "prompting": "Prompting",
    "diffmean_l20": "DiffMean (L20, c=1)",
    "diffmean_l16": "DiffMean (L16, c=0.75)",
    "tuned": "Tuned",
    "adaptive": "Adaptive",
    "tuned_adaptive": "Tuned + Adaptive",
}

MODEL_STEERING = {
    "Llama-3.1-8B-Instruct": "diffmean-source-L16-c0p75",
    "tiny-aya-global": "diffmean-source-L12-c1",
    "Qwen2.5-7B-Instruct": "diffmean-source-L21-c0p5",
    "Qwen2.5-14B-Instruct": "diffmean-source-L36-c1",
}
MODEL_LABELS = {
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "tiny-aya-global": "tiny-aya-global",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen2.5-14B-Instruct": "Qwen2.5-14B",
}

LANG_NAMES = {"ar": "Arabic", "de": "German", "en": "English", "es": "Spanish", "fr": "French",
              "ja": "Japanese", "zh": "Chinese", "ru": "Russian", "hi": "Hindi", "ko": "Korean",
              "cs": "Czech", "el": "Greek", "fa": "Persian", "vi": "Vietnamese", "uk": "Ukrainian",
              "mt": "Maltese", "no": "Norwegian", "sw": "Swahili", "th": "Thai", "tr": "Turkish"}

LOOP_FILES = {
    ("t1", "prompting"): ("results/inspect_llama_prompting.txt", None),
    ("t1", "diffmean_l20"): ("results/inspect_llama_diffmean.txt", None),
    ("t2", "diffmean_l20"): ("results/inspect_Llama-3.1-8B-Instruct_t2_diffmean.txt", "920a1ea"),
    ("t3", "diffmean_l20"): ("results/inspect_Llama-3.1-8B-Instruct_t3-de_diffmean.txt", "920a1ea"),
    ("t1", "diffmean_l16"): ("results/inspect_Llama-3.1-8B-Instruct_t1_diffmean.txt", None),
    ("t2", "diffmean_l16"): ("results/inspect_Llama-3.1-8B-Instruct_t2_diffmean.txt", None),
    ("t3", "diffmean_l16"): ("results/inspect_Llama-3.1-8B-Instruct_t3-de_diffmean.txt", None),
    ("t1", "tuned"): ("results/inspect_Llama-3.1-8B-Instruct_t1_tuned.txt", None),
    ("t2", "tuned"): ("results/inspect_Llama-3.1-8B-Instruct_t2_tuned.txt", None),
    ("t3", "tuned"): ("results/inspect_Llama-3.1-8B-Instruct_t3-de_tuned.txt", None),
    ("t1", "adaptive"): ("results/inspect_Llama-3.1-8B-Instruct_t1_adaptive.txt", None),
    ("t2", "adaptive"): ("results/inspect_Llama-3.1-8B-Instruct_t2_adaptive.txt", None),
    ("t3", "adaptive"): ("results/inspect_Llama-3.1-8B-Instruct_t3-de_adaptive.txt", None),
    ("t2", "tuned_adaptive"): ("results/inspect_Llama-3.1-8B-Instruct_t2_tuned-adaptive.txt", None),
}


def scores_path(model: str, track: str, method_dir: str) -> Path:
    return RAW_TEST / model / TRACKS[track] / method_dir / "scores.json"


def load_scores(model: str, track: str, method_dir: str):
    p = scores_path(model, track, method_dir)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def harmonic(a: float, b: float) -> float:
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def loop_rate(track: str, method: str):
    entry = LOOP_FILES.get((track, method))
    if entry is None:
        return None
    rel, commit = entry
    if commit is None:
        p = REPO_ROOT / rel
        if not p.exists():
            return None
        text = p.read_text(encoding="utf-8")
    else:
        try:
            text = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{commit}:{rel}"],
                                  capture_output=True, text=True, encoding="utf-8",
                                  check=True).stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    for line in reversed(text.strip().splitlines()):
        if "look repetitive" in line:
            return float(line.split("(")[1].split("%")[0])
    return None
