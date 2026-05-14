"""Steer the model with hand-picked EP partitions whose meanings are read
directly off ``partition.sample_prompts`` (no cosine contrast against
synthetic seeds).

Headline claim: "for partitions whose member prompts are clearly about
concept C, steering with `alpha * partition_direction` at the model's
target layer pushes generation toward C in a way that scales with alpha
and beats a random partition control at the same alpha."

The partitions and their semantics are read off ``sample_prompts`` directly
on the host before this script runs (see scripts/exp_concept_steering.py
docstring for the cosine-contrast version, which had selection failures).

Modal:
    modal run ep_modal_experiments.py::partition_steering
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Concept registry — partitions and scorers.
# Partition IDs are valid for gemma-2-2b-it L20 p4 mt1M
# (dictionaries/gemma-2-2b-it_L20_p4_ctx128_mt1000000_bs128_seed0_per-position_satbatch_full).
# ---------------------------------------------------------------------------

@dataclass
class ConceptSpec:
    name: str
    pid: int             # partition id chosen by inspection of sample_prompts
    scorer: callable     # text -> float
    label: str           # human-readable description for the table
    positives: list[str] = None  # seed prompts for DiffMean baseline


def _cyrillic_score(text: str) -> float:
    """Fraction of letter characters that are Cyrillic."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    cyr = sum(1 for c in letters if "Ѐ" <= c <= "ӿ")
    return float(cyr / len(letters))


_CODE_PATTERNS = [
    re.compile(r"\bdef\s+\w+\s*\("),
    re.compile(r"\bfunction\s+\w*\s*\("),
    re.compile(r"\bclass\s+\w+"),
    re.compile(r"\breturn\b"),
    re.compile(r"\bimport\s+\w+"),
    re.compile(r"\bfrom\s+\w+\s+import"),
    re.compile(r"\bif\s+__name__"),
    re.compile(r"=>"),
    re.compile(r"->\s*\w+"),
    re.compile(r"\bconsole\.log\b"),
    re.compile(r"\bprint\s*\("),
    re.compile(r"\bSELECT\b\s.+\bFROM\b", re.IGNORECASE),
    re.compile(r"```"),
]

# Vocabulary that shows up when the model is steered toward Python-like
# content but generates fragments rather than valid syntax.
_PYTHON_VOCAB = {
    "import", "from", "def", "class", "return", "self", "lambda",
    "async", "await", "yield", "with", "try", "except", "raise",
    "True", "False", "None", "elif", "pass",
    # Common module/object names that recur in our partition's sample prompts.
    "utils", "models", "module", "config", "dataset", "datasets",
    "tokenizer", "tensor", "torch", "numpy", "json", "logger",
    "model", "params", "args", "kwargs", "data",
}


def _code_score(text: str) -> float:
    """Code-syntax marker hits + Python-vocabulary density per 100 chars."""
    if not text.strip():
        return 0.0
    n = max(1, len(text))
    hits = sum(len(p.findall(text)) for p in _CODE_PATTERNS)
    indent_lines = sum(1 for line in text.splitlines()
                       if line.startswith(("    ", "\t")))
    # Tokenise on word boundaries AND on internal CamelCase / snake_case
    # boundaries — under heavy steering the model often emits "utils" runs
    # without spaces ("utilsutilsutilsdatautils"), and we want to count
    # those too.
    chunked = re.findall(r"[a-z]+|[A-Z][a-z]*", text)
    vocab_hits = sum(1 for w in chunked if w.lower() in _PYTHON_VOCAB)
    return float((hits + vocab_hits) / (n / 100.0) + 0.5 * indent_lines)


_MATH_PATTERNS = [
    re.compile(r"\\frac"),
    re.compile(r"\\sqrt"),
    re.compile(r"\\sum"),
    re.compile(r"\\int"),
    re.compile(r"\$[^$]+\$"),       # inline TeX
    re.compile(r"\b\d+\s*[+\-*/]\s*\d+"),  # 3 + 4
    re.compile(r"\bx\s*=\s*-?\d"),       # x = N
    re.compile(r"\^\s*\{?-?\d"),        # x^2
    re.compile(r"=\s*-?\d+(?:\.\d+)?"),  # = N
    re.compile(r"\b[a-z]\s*=\s*[a-z\d]"),  # symbolic eq
]


def _math_score(text: str) -> float:
    if not text.strip():
        return 0.0
    n = max(1, len(text))
    hits = sum(len(p.findall(text)) for p in _MATH_PATTERNS)
    return float(hits / (n / 100.0))


_BIOMED_KEYWORDS = {
    "nitric", "oxide", "inflammation", "fibrosis", "cystic", "protein",
    "molecular", "cellular", "expression", "receptor", "enzyme",
    "metabolism", "pathway", "signaling", "kinase", "antibody",
    "tumor", "immune", "neural", "synaptic", "mitochondria",
    "genome", "genetic", "rna", "dna", "ribosome",
    "phosphorylation", "regulation", "homeostasis",
}


def _biomed_score(text: str) -> float:
    words = re.findall(r"[A-Za-z]+", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in _BIOMED_KEYWORDS)
    return float(hits / len(words))


CYRILLIC_POS = [
    "Обяснение на вот. Устни обяснения на вот.",
    "Здравейте, как сте днес?",
    "Времето е много хубаво в София днес.",
    "Учителят обясни задачата подробно.",
    "Книгата на масата е много интересна.",
    "Той живее в малко село близо до планината.",
    "Кучето тича в парка всеки ден.",
    "Благодаря много за подаръка.",
    "Утре ще пътувам до Варна с автобус.",
    "Обичам да чета поезия през уикенда.",
    "Здравствуйте, как ваши дела?",
    "Сегодня очень тёплая погода в Москве.",
    "Я работаю в библиотеке три дня в неделю.",
    "Эта книга была очень интересной.",
    "Спасибо большое за вашу помощь.",
    "Кошка спит на диване весь день.",
    "Завтра я поеду в Санкт-Петербург на поезде.",
    "Мой друг живёт в маленькой деревне.",
    "Детям нравится играть в парке.",
    "Я учу русский язык уже два года.",
    "Хлеб и сыр на завтрак — простая еда.",
    "Концерт начинается в восемь вечера.",
    "Пожалуйста, передайте мне соль.",
    "Мне нравится слушать классическую музыку.",
    "Дождь идёт с самого утра.",
    "Этот ресторан очень популярен в нашем городе.",
    "Дети учатся в школе пять дней в неделю.",
    "Я забыл свой зонтик дома сегодня.",
    "Море очень красивое во время заката.",
    "На рынке продают свежие овощи и фрукты.",
]

PYTHON_POS = [
    "import os\nimport shutil\nfrom pathlib import Path",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "import numpy as np\nimport pandas as pd\ndf = pd.read_csv('data.csv')",
    "from sklearn.linear_model import LogisticRegression\nclf = LogisticRegression()",
    "import torch\nimport torch.nn as nn\n\nclass Net(nn.Module):\n    pass",
    "for i, item in enumerate(items):\n    print(f'{i}: {item}')",
    "with open('file.txt', 'r') as f:\n    lines = f.readlines()",
    "result = [x ** 2 for x in range(10) if x % 2 == 0]",
    "from typing import Optional, List\n\ndef foo(xs: List[int]) -> Optional[int]:\n    return None",
    "import json\nimport logging\nlogger = logging.getLogger(__name__)",
    "@app.route('/api/users')\ndef get_users():\n    return jsonify(users)",
    "try:\n    response = requests.get(url)\nexcept Exception as e:\n    logger.error(e)",
    "df = df.groupby('category').agg({'value': 'mean'}).reset_index()",
    "model = Sequential([\n    Dense(128, activation='relu'),\n    Dense(10, activation='softmax')\n])",
    "import asyncio\n\nasync def fetch(url):\n    async with aiohttp.ClientSession() as session:\n        return await session.get(url)",
    "if __name__ == '__main__':\n    main()",
    "import unittest\n\nclass TestFoo(unittest.TestCase):\n    def test_one(self):\n        self.assertEqual(1+1, 2)",
    "from dataclasses import dataclass\n\n@dataclass\nclass Point:\n    x: float\n    y: float",
    "import re\npattern = re.compile(r'\\d+')\nmatches = pattern.findall(text)",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])",
    "import sqlalchemy as sa\nengine = sa.create_engine('sqlite:///data.db')",
    "from collections import defaultdict\ncounts = defaultdict(int)",
    "x = np.array([[1, 2], [3, 4]])\ny = x.T @ x",
    "import requests\nresponse = requests.post(url, json=payload, headers=headers)",
    "from functools import lru_cache\n\n@lru_cache(maxsize=None)\ndef expensive(x): return x * x",
    "import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--input', required=True)",
    "from concurrent.futures import ThreadPoolExecutor\nwith ThreadPoolExecutor() as exe:\n    results = list(exe.map(fn, args))",
    "import yaml\nwith open('config.yaml') as f:\n    config = yaml.safe_load(f)",
    "import csv\nwith open('out.csv', 'w', newline='') as f:\n    writer = csv.writer(f)",
    "from datetime import datetime, timedelta\nnow = datetime.utcnow()",
]

REACT_POS = [
    "import * as React from 'react';\nimport { useState } from 'react';",
    "const App = () => {\n  return <div>Hello</div>;\n};",
    "import React, { useEffect, useRef } from 'react';",
    "function Button({ onClick, children }) {\n  return <button onClick={onClick}>{children}</button>;\n}",
    "interface UserProps { id: number; name: string; }",
    "const [count, setCount] = useState(0);",
    "useEffect(() => {\n  fetchData();\n}, []);",
    "import { Link } from 'react-router-dom';",
    "type Theme = 'light' | 'dark';",
    "const useFetch = (url: string) => { /* ... */ };",
    "<Component prop={value} onChange={handleChange} />",
    "export default function Page() { return <Layout /> }",
    "import styled from 'styled-components';\nconst Wrapper = styled.div`color: red;`;",
    "const memoized = useMemo(() => compute(x), [x]);",
    "<Provider store={store}><App /></Provider>",
    "const handleSubmit = (e: React.FormEvent) => { e.preventDefault(); };",
    "export const ThemeContext = React.createContext<Theme>('light');",
    "import { z } from 'zod';\nconst schema = z.object({ name: z.string() });",
    "<form onSubmit={handleSubmit}><input value={name} onChange={e => setName(e.target.value)} /></form>",
    "const data = await fetch('/api/users').then(r => r.json());",
    "export type User = { id: number; email: string };",
    "const router = createBrowserRouter([{ path: '/', element: <Home /> }]);",
    "const reducer = (state, action) => { switch (action.type) { /* ... */ } };",
    "<ErrorBoundary fallback={<Error />}>{children}</ErrorBoundary>",
    "const Component: React.FC<Props> = ({ id }) => <span>{id}</span>;",
    "import { motion } from 'framer-motion';\n<motion.div animate={{ opacity: 1 }} />",
    "const [open, setOpen] = useState<boolean>(false);",
    "useEffect(() => () => cleanup(), []);",
    "const debounced = useDebouncedCallback(fn, 300);",
    "<Suspense fallback={<Spinner />}><LazyComponent /></Suspense>",
]

MATH_POS = [
    "Suppose 3*x + 5 = 14. Solve for x.",
    "Let f(x) = x^2 + 2x + 1. Find f'(x).",
    "Evaluate the integral of x^2 from 0 to 3.",
    "If a = -7 and b = 3, what is a + 2b?",
    "Solve the system: 2x + y = 5, x - y = 1.",
    "Compute the determinant of [[1, 2], [3, 4]].",
    "What is the value of (-3)^2 - 4*(-3) + 1?",
    "Suppose -d = 2*k + 29 and 0*k - 4*k - 83 = 3*d. Find d.",
    "Factor the polynomial x^2 - 5x + 6.",
    "Find the roots of the equation x^2 - 4 = 0.",
    "If sin(theta) = 1/2, what is theta in radians?",
    "Compute lim_{x -> 0} sin(x)/x.",
    "Suppose f(x) = e^x. Find f''(x).",
    "Differentiate y = ln(x^2 + 1).",
    "What is the sum of the first 10 positive integers?",
    "If A = [[2, 0], [0, 3]] then A^2 = ?",
    "Solve for x: 2^x = 32.",
    "Evaluate (-12) / 3 + 5.",
    "What is the gradient of f(x, y) = x^2 + y^2?",
    "Suppose 5x = 25. Then x = ?",
    "Compute the binomial coefficient C(5, 2).",
    "Find x: log_2(x) = 3.",
    "If z = 3 + 4i, what is |z|?",
    "Solve x^3 - 8 = 0.",
    "Compute 7! / (5! * 2!).",
    "Differentiate g(x) = sqrt(x).",
    "Compute the dot product of (1, 2, 3) and (4, 5, 6).",
    "What is the area of a circle with radius 4?",
    "Solve the inequality 2x + 3 > 7.",
    "Find the slope of the line through (1, 2) and (4, 8).",
]

BIOMED_POS = [
    "Nitric oxide (NO) is produced from three isoforms of nitric oxide synthase, contributing to inflammation regulation.",
    "Cystic fibrosis is a genetic disorder caused by mutations in the CFTR gene affecting chloride transport.",
    "Mitochondrial dysfunction has been linked to a wide range of metabolic and neurodegenerative diseases.",
    "Tumor-infiltrating lymphocytes (TILs) play a critical role in anti-tumor immunity and immunotherapy response.",
    "The kinase mTOR regulates cell growth, proliferation, and autophagy in response to nutrient signals.",
    "RNA polymerase II transcribes protein-coding genes and produces precursor messenger RNA in eukaryotes.",
    "Synaptic plasticity is the cellular basis of learning and memory in the mammalian hippocampus.",
    "Phosphorylation of tau protein is implicated in the pathogenesis of Alzheimer's disease.",
    "The ribosome catalyzes peptide bond formation during protein synthesis at the peptidyl transferase center.",
    "Inflammation is mediated by cytokines, chemokines, and reactive oxygen species in the innate immune response.",
    "G-protein coupled receptors are a major class of cell surface receptors targeted by many pharmaceuticals.",
    "Apoptosis is regulated by the Bcl-2 family of proteins controlling mitochondrial outer membrane permeabilization.",
    "Stem cell differentiation is governed by transcription factor networks and epigenetic remodeling.",
    "The blood-brain barrier restricts the passage of most molecules from the bloodstream into neural tissue.",
    "DNA methylation at CpG islands is a major epigenetic mechanism for gene silencing.",
    "Insulin resistance in skeletal muscle contributes to the development of type 2 diabetes mellitus.",
    "Antibody-dependent cellular cytotoxicity is mediated by natural killer cells via Fc receptor engagement.",
    "Calcium signaling regulates contraction in cardiac myocytes via the ryanodine receptor.",
    "The endoplasmic reticulum stress response activates the unfolded protein response pathway.",
    "Microglia are the resident macrophages of the central nervous system and modulate neuroinflammation.",
    "Tumor microenvironment heterogeneity contributes to therapeutic resistance in solid cancers.",
    "Long-term potentiation involves NMDA receptor activation and AMPA receptor trafficking.",
    "Homeostatic regulation of iron involves hepcidin and the ferroportin transporter.",
    "Hippocampal neurogenesis declines with age and is associated with cognitive decline.",
    "The complement system mediates innate immune responses through cascading proteolytic activation.",
    "Mitogen-activated protein kinase pathways transduce extracellular signals to nuclear transcription factors.",
    "T cell receptor diversity is generated through V(D)J recombination during thymic development.",
    "Autophagy degrades damaged organelles and protein aggregates via the lysosomal pathway.",
    "Oxidative phosphorylation in mitochondria couples electron transport to ATP synthesis.",
    "Dopaminergic neurons in the substantia nigra degenerate progressively in Parkinson's disease.",
]


CONCEPTS: list[ConceptSpec] = [
    ConceptSpec("cyrillic",  424, _cyrillic_score,
                "Bulgarian/Russian text (Cyrillic script)",
                positives=CYRILLIC_POS),
    ConceptSpec("python",    27,  _code_score,
                "Python imports and module body",
                positives=PYTHON_POS),
    ConceptSpec("react",     395, _code_score,
                "React / TypeScript imports and JSX",
                positives=REACT_POS),
    ConceptSpec("math",      131, _math_score,
                "Symbolic / arithmetic equations",
                positives=MATH_POS),
    ConceptSpec("biomedical", 464, _biomed_score,
                "Biomedical research abstract",
                positives=BIOMED_POS),
]

# Negatives for DiffMean: generic English prose, same as the neutral
# eval prompts we steer over.
DIFFMEAN_NEGATIVES = [
    "The cat sat on the mat in the warm afternoon sun.",
    "My favourite book is one about the history of mathematics.",
    "I went to the grocery store this morning to buy bread.",
    "The weather has been quite pleasant for the past week.",
    "She finished her homework before going to bed last night.",
    "The new restaurant downtown serves excellent Italian food.",
    "He decided to take a walk in the park to clear his head.",
    "The library is open until nine on weekdays.",
    "We watched a documentary about coral reefs over dinner.",
    "Her brother is studying engineering at the local university.",
    "The garden looks beautiful when the roses are in bloom.",
    "I forgot my umbrella so I got wet on the way home.",
    "The meeting has been rescheduled for next Tuesday afternoon.",
    "He plays guitar in a small band on weekends.",
    "She sent me a postcard from her vacation in Greece.",
    "The bakery on the corner makes the best croissants.",
    "I need to renew my driver's license before the end of the month.",
    "The children are excited about the upcoming school trip.",
    "He has been collecting vintage stamps for over twenty years.",
    "The conference will be held in San Francisco this fall.",
    "My grandmother taught me how to bake apple pie.",
    "The old movie theatre closed down a few years ago.",
    "She painted her bedroom a soft shade of blue.",
    "The bus runs every fifteen minutes during rush hour.",
    "I tried a new recipe for chicken curry last night.",
    "The professor gave a fascinating lecture on ancient Rome.",
    "He finally finished the marathon he had been training for.",
    "The kids built a snowman in the front yard yesterday.",
    "She volunteers at the animal shelter every Saturday.",
    "The market has fresh fruit and vegetables on Sundays.",
]


# Neutral generation prompts (same as exp_concept_steering).
NEUTRAL_INSTRUCTIONS = [
    "Tell me a fun fact about animals.",
    "Write a short paragraph about cooking.",
    "What's your favourite season and why?",
    "Describe a typical morning routine.",
    "Give me three tips for learning a new skill.",
    "Recommend a good book for someone new to reading.",
    "Explain what makes a friendship strong.",
    "Describe what a good teacher is like.",
    "Tell me about a hobby you find interesting.",
    "What's the best way to spend a rainy day?",
    "Write a short paragraph about gardening.",
    "Describe a place you've always wanted to visit.",
    "Suggest a healthy breakfast idea.",
    "Tell me about the importance of sleep.",
    "Recommend three things to do on a free weekend.",
    "Describe what a relaxing evening looks like to you.",
    "Tell me something interesting about the ocean.",
    "Suggest a fun activity to do with friends.",
    "Describe the best way to organize a small kitchen.",
    "Write a short paragraph about a memorable holiday.",
]


def _auto_select_partition(dictionary, scorer, min_members: int = 30) -> tuple[int, float]:
    """For a given content-scorer, pick the partition whose sample_prompts
    score highest. Skips partitions with fewer than ``min_members`` to
    avoid noise singletons.

    Returns (partition_id, score). score=0 means nothing matched.
    """
    best_pid, best_score = -1, -1.0
    for pid, p in enumerate(dictionary.partitions):
        if p.member_count < min_members:
            continue
        sample = " ".join(s[:300] for _, s, _ in p.sample_prompts[:5])
        s = scorer(sample)
        if s > best_score:
            best_score, best_pid = s, pid
    return best_pid, float(best_score)


def _diffmean_direction(model, hook_name: str,
                        positives: list[str], negatives: list[str],
                        center: np.ndarray,
                        batch_size: int = 4) -> np.ndarray:
    """AxBench-style DiffMean baseline: mean(act|pos) - mean(act|neg) at hook_name,
    L2-normalised in centered space. Returns a unit vector compatible with
    EP exemplar/mean directions.

    Activations are taken as the per-prompt mean over the back-half of tokens
    (matching the partition-selection convention in exp_concept_steering.py).
    """
    def _mean_acts(prompts):
        acts = []
        for prompt in prompts:
            formatted = _format_chat(model, prompt)
            tokens = model.to_tokens(formatted, prepend_bos=True)
            cache = {}

            def cache_hook(act, hook):
                cache["a"] = act.detach().cpu().float()

            model.reset_hooks()
            model.add_hook(hook_name, cache_hook, "fwd")
            try:
                with torch.no_grad():
                    model(tokens)
            finally:
                model.reset_hooks()
            a = cache["a"][0]              # (T, D)
            half = a.shape[0] // 2
            acts.append(a[half:].mean(dim=0).numpy())
        return np.stack(acts)

    pos_acts = _mean_acts(positives)
    neg_acts = _mean_acts(negatives)
    pos_centered = pos_acts.mean(axis=0) - center
    neg_centered = neg_acts.mean(axis=0) - center
    diff = pos_centered - neg_centered
    return (diff / (np.linalg.norm(diff) + 1e-12)).astype(np.float32)


def _format_chat(model, prompt: str) -> str:
    try:
        return model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


def _generate_steered(model, prompts: list[str], hook_name: str,
                      direction: torch.Tensor | None,
                      alpha: float, max_new_tokens: int = 60,
                      batch_size: int = 8) -> list[str]:
    tokenizer = model.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    out_texts: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        formatted = [_format_chat(model, p) for p in chunk]
        tokenizer.padding_side = "left"
        enc = tokenizer(formatted, return_tensors="pt", padding=True,
                        add_special_tokens=False)
        input_ids = enc["input_ids"].to(model.cfg.device)

        if direction is not None and alpha != 0.0:
            def steer_hook(act, hook, _d=direction, _a=alpha):
                return act + _a * _d.to(device=act.device, dtype=act.dtype)
            model.reset_hooks()
            model.add_hook(hook_name, steer_hook, "fwd")
        else:
            model.reset_hooks()

        try:
            with torch.no_grad():
                out = model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    do_sample=False, temperature=0.0, verbose=False,
                )
            in_len = input_ids.shape[1]
            new_tokens = out[:, in_len:]
        finally:
            model.reset_hooks()

        for row in new_tokens:
            out_texts.append(tokenizer.decode(row, skip_special_tokens=True))
    return out_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--model-short", default="gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--dict-path", required=True)
    parser.add_argument("--alphas", default="0,16,64,128,256")
    parser.add_argument("--auto-pid", action="store_true",
                        help="For each concept, auto-discover the best "
                             "partition by scoring partition.sample_prompts "
                             "with the concept's scorer. Overrides hard-coded "
                             "pids; required when running on dictionaries "
                             "other than the one those pids were curated for.")
    parser.add_argument("--n-eval-prompts", type=int, default=15)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--n-random-controls", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="ep-properties")
    parser.add_argument("--wandb-entity", default="jessicamarycooper")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", force=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if args.output_dir is None:
        args.output_dir = Path("results/exp_partition_steering") / (
            f"{args.model_short}_L{args.layer}_seed{args.seed}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    alphas = [float(a) for a in args.alphas.split(",")]
    eval_prompts = NEUTRAL_INSTRUCTIONS[:args.n_eval_prompts]

    # --- Load dictionary ---
    logger.info("Loading dictionary %s", args.dict_path)
    with open(args.dict_path, "rb") as f:
        dictionary = pickle.load(f)
    logger.info("Dictionary: %s", dictionary)
    n_partitions = len(dictionary.partitions)

    # --- Load model ---
    import transformer_lens as tl
    logger.info("Loading model %s on %s", args.model, args.device)
    t0 = time.time()
    model = tl.HookedTransformer.from_pretrained_no_processing(
        args.model, device=args.device, dtype=torch.bfloat16,
    )
    model.eval()
    hook_name = f"blocks.{args.layer}.hook_resid_post"
    logger.info("Model loaded in %.1fs", time.time() - t0)

    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=f"partition_steering_{args.model_short}_L{args.layer}_seed{args.seed}",
            config=vars(args), job_type="partition_steering",
        )

    all_results = {}
    pair_examples = []

    if args.auto_pid:
        logger.info("Auto-discovering partition pids by scoring sample_prompts:")
        for spec in CONCEPTS:
            new_pid, sc = _auto_select_partition(dictionary, spec.scorer)
            logger.info("  %-12s old_pid=%4d  -> auto_pid=%4d  score=%.3f",
                        spec.name, spec.pid, new_pid, sc)
            if new_pid >= 0:
                spec.pid = new_pid

    for spec in CONCEPTS:
        logger.info("=== %s (pid=%d, %s) ===",
                    spec.name, spec.pid, spec.label)
        ep_dir = torch.from_numpy(
            dictionary.partitions[spec.pid]
            .mean_member_direction.astype(np.float32)
        )

        # DiffMean baseline: supervised mean(pos) - mean(neg) at hook_name,
        # normalised in centered space. Same direction format as EP (unit
        # vector in centered space) so steering is apples-to-apples.
        if spec.positives is not None:
            logger.info("  computing DiffMean direction (%d pos, %d neg)",
                        len(spec.positives), len(DIFFMEAN_NEGATIVES))
            dm_np = _diffmean_direction(
                model, hook_name, spec.positives, DIFFMEAN_NEGATIVES,
                dictionary.center,
            )
            dm_dir = torch.from_numpy(dm_np)
            # Sanity-check cosine sim between EP and DiffMean directions.
            ep_np = dictionary.partitions[spec.pid].mean_member_direction
            cos = float(np.dot(ep_np, dm_np))
            logger.info("  cos(EP, DiffMean) = %.4f", cos)
        else:
            dm_dir = None
            cos = float("nan")

        # Random partitions sampled per concept (avoid the EP one).
        candidate = list(range(n_partitions))
        candidate.remove(spec.pid)
        rng.shuffle(candidate)
        rand_pids = candidate[:args.n_random_controls]
        rand_dirs = [
            torch.from_numpy(
                dictionary.partitions[r]
                .mean_member_direction.astype(np.float32)
            )
            for r in rand_pids
        ]

        per_alpha = {}
        for alpha in alphas:
            ep_gens = _generate_steered(
                model, eval_prompts, hook_name,
                ep_dir if alpha != 0 else None, alpha,
                max_new_tokens=args.max_new_tokens,
            )
            ep_scores = [spec.scorer(g) for g in ep_gens]

            if dm_dir is not None:
                dm_gens = _generate_steered(
                    model, eval_prompts, hook_name,
                    dm_dir if alpha != 0 else None, alpha,
                    max_new_tokens=args.max_new_tokens,
                )
                dm_scores = [spec.scorer(g) for g in dm_gens]
            else:
                dm_gens = None
                dm_scores = None

            rand_score_arrays = []
            rand_gens_pooled = []
            for rd in rand_dirs:
                gens = _generate_steered(
                    model, eval_prompts, hook_name,
                    rd if alpha != 0 else None, alpha,
                    max_new_tokens=args.max_new_tokens,
                )
                scores = [spec.scorer(g) for g in gens]
                rand_score_arrays.append(scores)
                rand_gens_pooled.append(gens)
            rand_scores = np.array(rand_score_arrays).mean(axis=0)

            entry = {
                "alpha": alpha,
                "ep_score_mean":   float(np.mean(ep_scores)),
                "ep_score_std":    float(np.std(ep_scores)),
                "rand_score_mean": float(np.mean(rand_scores)),
                "rand_score_std":  float(np.std(rand_scores)),
                "n_prompts": len(eval_prompts),
            }
            if dm_scores is not None:
                entry["dm_score_mean"] = float(np.mean(dm_scores))
                entry["dm_score_std"]  = float(np.std(dm_scores))
            per_alpha[str(alpha)] = entry

            if dm_scores is not None:
                logger.info("  alpha=%5.1f  ep=%.4f  dm=%.4f  rand=%.4f  "
                            "ep-rand=%+.4f  ep-dm=%+.4f",
                            alpha, np.mean(ep_scores), np.mean(dm_scores),
                            np.mean(rand_scores),
                            np.mean(ep_scores) - np.mean(rand_scores),
                            np.mean(ep_scores) - np.mean(dm_scores))
            else:
                logger.info("  alpha=%5.1f  ep=%.4f  rand=%.4f  lift=%.4f",
                            alpha, np.mean(ep_scores), np.mean(rand_scores),
                            np.mean(ep_scores) - np.mean(rand_scores))

            for i, p in enumerate(eval_prompts[:5]):
                pair_examples.append({
                    "concept": spec.name, "pid": spec.pid, "alpha": alpha,
                    "condition": "ep", "prompt": p, "gen": ep_gens[i],
                    "score": float(ep_scores[i]),
                })
                if dm_gens is not None:
                    pair_examples.append({
                        "concept": spec.name, "pid": -1, "alpha": alpha,
                        "condition": "diffmean", "prompt": p,
                        "gen": dm_gens[i],
                        "score": float(dm_scores[i]),
                    })
                pair_examples.append({
                    "concept": spec.name, "pid": rand_pids[0], "alpha": alpha,
                    "condition": "rand", "prompt": p,
                    "gen": rand_gens_pooled[0][i],
                    "score": float(rand_score_arrays[0][i]),
                })

            if args.wandb:
                import wandb
                log_data = {
                    f"{spec.name}/alpha":  alpha,
                    f"{spec.name}/ep":    np.mean(ep_scores),
                    f"{spec.name}/rand":  np.mean(rand_scores),
                    f"{spec.name}/lift":  np.mean(ep_scores) - np.mean(rand_scores),
                }
                if dm_scores is not None:
                    log_data[f"{spec.name}/dm"] = np.mean(dm_scores)
                    log_data[f"{spec.name}/ep_minus_dm"] = (
                        np.mean(ep_scores) - np.mean(dm_scores)
                    )
                wandb.log(log_data)

        all_results[spec.name] = {
            "partition_id": spec.pid,
            "label": spec.label,
            "rand_partition_ids": rand_pids,
            "ep_dm_cosine": cos,
            "per_alpha": per_alpha,
        }

    # --- Headline ---
    logger.info("\n=== HEADLINE ===")
    logger.info(f"{'concept':<12} {'pid':>5} {'best α':>7} {'EP':>8} {'DM':>8} "
                f"{'rand':>8} {'EP-rand':>9} {'EP-DM':>8} {'cos(EP,DM)':>11}")
    headlines = {}
    for cname, r in all_results.items():
        best_a, best_lift = None, -np.inf
        for a, v in r["per_alpha"].items():
            lift = v["ep_score_mean"] - v["rand_score_mean"]
            if lift > best_lift:
                best_lift, best_a = lift, a
        v = r["per_alpha"][best_a]
        dm = v.get("dm_score_mean", float("nan"))
        ep_minus_dm = v["ep_score_mean"] - dm if not np.isnan(dm) else float("nan")
        logger.info("%-12s %5d %7s %8.4f %8.4f %8.4f %9.4f %8.4f %11.4f",
                    cname, r["partition_id"], best_a,
                    v["ep_score_mean"], dm, v["rand_score_mean"],
                    v["ep_score_mean"] - v["rand_score_mean"],
                    ep_minus_dm, r.get("ep_dm_cosine", float("nan")))
        headlines[cname] = {
            "partition_id": r["partition_id"],
            "label": r["label"],
            "best_alpha": float(best_a),
            "ep_score_at_best": v["ep_score_mean"],
            "rand_score_at_best": v["rand_score_mean"],
            "dm_score_at_best": dm,
            "lift_vs_rand": v["ep_score_mean"] - v["rand_score_mean"],
            "lift_vs_dm": ep_minus_dm,
            "ep_dm_cosine": r.get("ep_dm_cosine", float("nan")),
        }

    payload = {
        "config": vars(args) | {"output_dir": str(args.output_dir),
                                "dict_path": str(args.dict_path)},
        "n_partitions": n_partitions,
        "alphas": alphas,
        "results": all_results,
        "headlines": headlines,
        "examples": pair_examples,
    }
    out_path = args.output_dir / "partition_steering.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Saved results to %s", out_path)

    if args.wandb:
        import wandb
        cols = ["concept", "partition_id", "label", "best_alpha",
                "ep_score_at_best", "rand_score_at_best", "lift"]
        rows = [[c, h["partition_id"], h["label"], h["best_alpha"],
                 h["ep_score_at_best"], h["rand_score_at_best"], h["lift"]]
                for c, h in headlines.items()]
        wandb.log({"headline": wandb.Table(columns=cols, data=rows)})

        ex_cols = ["concept", "pid", "alpha", "condition",
                   "prompt", "gen", "score"]
        ex_rows = [[r[c] for c in ex_cols] for r in pair_examples]
        wandb.log({"examples": wandb.Table(columns=ex_cols, data=ex_rows)})

        for cname, h in headlines.items():
            wandb.summary[f"summary/{cname}/lift"] = h["lift"]
            wandb.summary[f"summary/{cname}/best_alpha"] = h["best_alpha"]
        wandb.summary["summary/mean_lift"] = float(
            np.mean([h["lift"] for h in headlines.values()])
        )
        wandb.finish()


if __name__ == "__main__":
    main()
