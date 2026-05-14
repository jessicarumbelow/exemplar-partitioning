"""Experiment: do EP partitions correspond to causally meaningful directions?

Setup:
  1. For each of a handful of automatically-scoreable concepts (Spanish,
     French, code, uppercase, refusal), build a small seed set of
     concept-positive and concept-negative example prompts.
  2. For each concept, find the EP partition whose direction maximises the
     cosine contrast between positive and negative seed activations at the
     target layer. (Same selection rule as AxBench's EPExemplar/EPMean
     module — see baselines/axbench/axbench/models/ep.py.)
  3. Generate from a fixed set of neutral instructions with the partition
     direction added to the residual stream at the target layer:
         act[pos] += alpha * partition_direction
     Sweep alpha in {0, 0.5, 1, 2, 4, 8}. Direction is unit-norm in
     centered space (EPDictionarySAE convention); alpha is the knob.
  4. Score generations automatically — no LM judge:
         spanish/french : language-marker character + word frequency
         code           : code-syntax regex hits per 100 chars
         uppercase      : fraction of letters in upper case
         refusal        : refusal-phrase substring match
  5. Compare the EP-direction sweep against:
         alpha=0 baseline (no steering)
         random partition direction (same alpha sweep)
     Report each concept's score curve as a function of alpha.

Headline claim under test: "for partitions empirically about concept C,
steering with the partition direction increases concept-C behaviour
markedly above what a random partition direction at the same alpha
produces". If true, EP partitions are usable steering vectors and the
'meaningful directions' claim has direct causal support.

Run on Modal:
    modal run ep_modal_experiments.py::concept_steering
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
# Concept seed sets and scorers.
# ---------------------------------------------------------------------------

SPANISH_POS = [
    "Hola, ¿cómo estás hoy?",
    "Mi gato es muy bonito y le gusta dormir todo el día.",
    "Me gusta cocinar pasta los domingos por la tarde.",
    "El sol brilla mucho hoy y hace calor en la ciudad.",
    "Voy a comprar pan y leche en la tienda de la esquina.",
    "La biblioteca está cerrada los lunes pero abre los martes.",
    "Mañana viajaré a Barcelona para visitar a mi familia.",
    "Las flores en el jardín son rojas, amarillas y blancas.",
    "Necesito aprender más sobre la historia de México.",
    "El concierto empieza a las ocho de la noche en el teatro.",
    "Estoy estudiando español porque quiero vivir en Madrid.",
    "Los niños están jugando en el parque cerca de la escuela.",
    "Quiero un café con leche y un croissant, por favor.",
    "El restaurante en la plaza sirve comida tradicional española.",
    "Me encanta la música clásica, especialmente Beethoven.",
    "Voy a la playa con mis amigos este fin de semana.",
    "El museo tiene una nueva exposición de arte moderno.",
    "La casa de mi abuela está en un pueblo pequeño.",
    "Mi hermana es médica y trabaja en un hospital grande.",
    "Aprendí a cocinar paella cuando vivía en Valencia.",
    "El libro que estoy leyendo es muy interesante.",
    "Tengo una reunión importante mañana por la mañana.",
    "Los perros del vecino ladran toda la noche.",
    "El examen de matemáticas fue más difícil de lo esperado.",
    "Vamos al cine esta noche para ver la nueva película.",
    "El parque está lleno de gente los fines de semana.",
    "Necesito comprar un regalo para el cumpleaños de mi amiga.",
    "La playa de Cádiz es una de mis favoritas en el mundo.",
    "Me duele la cabeza después de trabajar tantas horas.",
    "El profesor explicó la lección con mucha paciencia.",
]

FRENCH_POS = [
    "Bonjour, comment allez-vous aujourd'hui?",
    "J'aime beaucoup le café noir le matin avec un croissant.",
    "Le chat dort sur le canapé pendant toute la journée.",
    "Demain, je vais visiter le musée du Louvre à Paris.",
    "Il fait très beau aujourd'hui dans le sud de la France.",
    "Mon frère habite à Lyon depuis plusieurs années.",
    "La boulangerie ouvre tôt le matin tous les jours.",
    "Nous avons mangé une excellente ratatouille hier soir.",
    "Les enfants jouent dans le jardin avec leurs amis.",
    "J'apprends le français parce que je veux travailler à Paris.",
    "La Tour Eiffel est très belle quand elle est illuminée.",
    "Le concert commence à huit heures du soir au théâtre.",
    "Je voudrais un verre de vin rouge avec mon repas.",
    "Le métro est le moyen le plus rapide de se déplacer en ville.",
    "Cette rue est très calme le dimanche matin.",
    "Mon professeur de français est patient et très gentil.",
    "Nous allons à la plage cet après-midi avec la famille.",
    "Le pain frais sent vraiment bon dans la cuisine.",
    "J'ai lu un livre fascinant sur l'histoire de France.",
    "Les Champs-Élysées sont magnifiques en automne.",
    "Mon père cuisine toujours le dimanche pour toute la famille.",
    "Le train pour Bordeaux part à neuf heures précises.",
    "J'écoute de la musique classique pendant que je travaille.",
    "La pluie tombe doucement sur les toits de la ville.",
    "Cette boutique vend de très belles robes fait main.",
    "Je préfère le thé au café après le déjeuner.",
    "Le jardin du Luxembourg est très agréable au printemps.",
    "Les baguettes françaises sont les meilleures du monde.",
    "Le marché du samedi est plein de fruits et de légumes frais.",
    "Mon ami m'a invité à dîner chez lui ce soir.",
]

CODE_POS = [
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "for i in range(10):\n    print(i * 2)",
    "import numpy as np\narr = np.zeros((5, 5))\nprint(arr.shape)",
    "class Animal:\n    def __init__(self, name):\n        self.name = name",
    "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
    "function reverseString(s) {\n  return s.split('').reverse().join('');\n}",
    "const result = arr.filter(x => x > 0).map(x => x * 2);",
    "while True:\n    data = read_input()\n    if data is None: break",
    "try:\n    response = requests.get(url)\nexcept Exception as e:\n    print(e)",
    "SELECT name, age FROM users WHERE age > 18 ORDER BY name;",
    "if (x > 0 && y > 0) { return x + y; } else { return 0; }",
    "let counter = 0;\nfunction increment() { counter++; }",
    "with open('file.txt', 'r') as f:\n    lines = f.readlines()",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2",
    "public static int factorial(int n) { return n <= 1 ? 1 : n * factorial(n-1); }",
    "git commit -m 'fix bug in parser' && git push origin main",
    "df['age_bucket'] = pd.cut(df['age'], bins=[0, 18, 65, 100])",
    "interface User { id: number; name: string; email?: string; }",
    "@app.route('/api/users', methods=['POST'])\ndef create_user():\n    data = request.json",
    "x = [i**2 for i in range(20) if i % 2 == 0]",
    "async def fetch_data(url):\n    async with aiohttp.ClientSession() as session:\n        return await session.get(url)",
    "func main() {\n    fmt.Println('Hello, World!')\n}",
    "matrix = [[1,2,3],[4,5,6],[7,8,9]]\ntranspose = list(zip(*matrix))",
    "console.log(`User ${user.name} logged in at ${new Date()}`);",
    "DROP TABLE IF EXISTS users CASCADE;",
    "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())",
    "vector<int> v = {1, 2, 3, 4, 5};\nfor (auto& x : v) x *= 2;",
    "if __name__ == '__main__':\n    main()",
    "fn add(a: i32, b: i32) -> i32 { a + b }",
    "git checkout -b feature/new-parser && git rebase main",
]

UPPER_POS = [
    "HELLO WORLD HOW ARE YOU TODAY",
    "I CANNOT BELIEVE THIS IS HAPPENING RIGHT NOW",
    "PLEASE STOP DOING THAT IMMEDIATELY",
    "WHY ARE YOU YELLING AT ME LIKE THIS",
    "THIS IS THE BEST DAY OF MY LIFE",
    "I AM SO TIRED OF WORKING SO MUCH",
    "STOP WHAT YOU ARE DOING AND LISTEN CAREFULLY",
    "EVERYTHING IS GOING TO BE FINE I PROMISE",
    "WE ARE GOING TO BE LATE FOR THE MEETING",
    "DO NOT TOUCH ANYTHING ON THE TABLE",
    "I LOVE THIS NEW SONG ON THE RADIO",
    "PLEASE PASS ME THE SALT FROM THE SHELF",
    "WHO ATE THE LAST SLICE OF PIZZA",
    "WE NEED TO LEAVE THE HOUSE IN FIVE MINUTES",
    "THE CAT KNOCKED OVER THE GLASS OF WATER AGAIN",
    "I FORGOT TO BRING MY UMBRELLA TODAY",
    "THIS COFFEE IS WAY TOO STRONG FOR ME",
    "I CANNOT FIND MY KEYS ANYWHERE IN THE HOUSE",
    "PLEASE TURN DOWN THE VOLUME ON THE TV",
    "THE TRAFFIC IS TERRIBLE THIS MORNING",
    "I JUST REMEMBERED I HAVE TO GO TO THE BANK",
    "THIS RESTAURANT HAS THE BEST FOOD IN TOWN",
    "WE SHOULD GO TO THE BEACH THIS WEEKEND",
    "DO NOT FORGET TO LOCK THE DOOR WHEN YOU LEAVE",
    "THE NEW MOVIE IS ABSOLUTELY INCREDIBLE",
    "I DO NOT WANT TO TALK ABOUT IT RIGHT NOW",
    "WHY IS THE PRINTER NEVER WORKING WHEN I NEED IT",
    "MY PHONE BATTERY DIED IN THE MIDDLE OF THE CALL",
    "EVERYBODY GETS A FREE GIFT TONIGHT",
    "STOP SHOUTING AT THE TELEVISION",
]

NEUTRAL_NEG = [
    "The cat sat on the mat in the warm afternoon sun.",
    "My favorite book is one about the history of mathematics.",
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
    "The old movie theater closed down a few years ago.",
    "She painted her bedroom a soft shade of blue.",
    "The bus runs every fifteen minutes during rush hour.",
    "I tried a new recipe for chicken curry last night.",
    "The professor gave a fascinating lecture on ancient Rome.",
    "He finally finished the marathon he had been training for.",
    "The kids built a snowman in the front yard yesterday.",
    "She volunteers at the animal shelter every Saturday.",
    "The market has fresh fruit and vegetables on Sundays.",
]

REFUSAL_POS = [
    "I'm sorry, but I cannot help you with that request.",
    "I cannot provide instructions for that activity.",
    "As an AI, I am not able to assist with this.",
    "I won't be able to help you with this question.",
    "I'm not comfortable discussing that topic.",
    "I'd rather not engage with that kind of content.",
    "That's not something I can help with, sorry.",
    "I cannot generate that content as it would be inappropriate.",
    "I have to decline this request as it goes against my guidelines.",
    "I'm unable to provide guidance on that subject.",
    "I will not assist with anything that could cause harm.",
    "Sorry, this falls outside what I'm willing to help with.",
    "I must respectfully decline to answer that.",
    "Providing that information would not be appropriate.",
    "I'm not going to help with that, as it could be dangerous.",
    "I cannot fulfill this request because it could lead to harm.",
    "I'm sorry, I can't do that for you.",
    "It would be irresponsible of me to assist with this.",
    "I have to refuse this kind of request.",
    "I will not engage with that line of questioning.",
    "That request is something I'm not able to comply with.",
    "I'm not in a position to help you with that.",
    "I'm sorry but I won't be writing that for you.",
    "I cannot in good conscience help with this.",
    "I need to politely decline this request.",
    "I'm afraid I can't be of assistance with that.",
    "Sorry, I have to decline.",
    "That's not something I should help with.",
    "I don't feel comfortable answering that question.",
    "I can't generate harmful content, sorry.",
]


# Concept registry: each concept has positive seeds (def. characteristic of
# the concept) and an automated scorer for output text.
@dataclass
class Concept:
    name: str
    positives: list[str]
    scorer: callable
    score_neutral_baseline: float = 0.0  # Pre-computed expected score on neutral text


def _spanish_score(text: str) -> float:
    text = text.lower()
    if not text.strip():
        return 0.0
    n_chars = len(text)
    es_chars = sum(text.count(c) for c in "ñ¿¡áéíóúü")
    es_words = sum(1 for w in re.findall(r"\b\w+\b", text)
                   if w in {"el", "la", "los", "las", "es", "son", "y", "de",
                            "en", "que", "un", "una", "por", "con", "para",
                            "no", "se", "lo", "le", "del", "al", "como",
                            "más", "pero", "hola", "gracias", "muy", "yo",
                            "tu", "mi", "su", "está", "están", "soy", "eres",
                            "son", "tengo", "tiene", "voy", "va", "vamos"})
    total_words = max(1, len(re.findall(r"\b\w+\b", text)))
    return float(es_chars / max(1, n_chars) * 5.0
                 + es_words / total_words)


def _french_score(text: str) -> float:
    text = text.lower()
    if not text.strip():
        return 0.0
    n_chars = len(text)
    fr_chars = sum(text.count(c) for c in "éèêëçîïàâôûùü")
    fr_words = sum(1 for w in re.findall(r"\b[\w']+\b", text)
                   if w in {"le", "la", "les", "de", "des", "un", "une",
                            "et", "est", "sont", "que", "qui", "pour", "dans",
                            "sur", "avec", "pas", "ne", "ce", "cette", "mes",
                            "tes", "ses", "mon", "ton", "son", "nous", "vous",
                            "ils", "elles", "je", "tu", "il", "elle",
                            "bonjour", "merci", "oui", "non", "très", "tout",
                            "toute", "tous", "j'ai", "c'est", "n'est"})
    total_words = max(1, len(re.findall(r"\b[\w']+\b", text)))
    return float(fr_chars / max(1, n_chars) * 5.0
                 + fr_words / total_words)


_CODE_PATTERNS = [
    re.compile(r"\bdef\s+\w+\s*\("),
    re.compile(r"\bfunction\s+\w*\s*\("),
    re.compile(r"\bclass\s+\w+"),
    re.compile(r"\breturn\b"),
    re.compile(r"\bimport\s+\w+"),
    re.compile(r"\bif\s+__name__"),
    re.compile(r"=>"),
    re.compile(r"->\s*\w+"),
    re.compile(r"console\.log"),
    re.compile(r"\bprint\s*\("),
    re.compile(r"\bSELECT\b\s.+\bFROM\b", re.IGNORECASE),
    re.compile(r"```"),  # triple-backtick code block
]


def _code_score(text: str) -> float:
    if not text.strip():
        return 0.0
    n = max(1, len(text))
    hits = sum(len(p.findall(text)) for p in _CODE_PATTERNS)
    # Indentation as a softer signal.
    indent_lines = sum(1 for line in text.splitlines()
                       if line.startswith(("    ", "\t")))
    return float(hits / (n / 100.0) + indent_lines * 0.5)


def _upper_score(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return float(sum(c.isupper() for c in letters) / len(letters))


REFUSAL_PHRASES = (
    "i can't", "i cannot", "i won't", "i will not", "i'm not able",
    "i am not able", "i'm unable", "i am unable", "i refuse",
    "as an ai", "i'm sorry", "i am sorry", "i apologize",
    "i don't think", "it would be inappropriate",
    "i'm not comfortable", "i am not comfortable",
    "i don't feel comfortable", "i must decline",
    "i'd rather not", "i would rather not",
    "that's not something", "i shouldn't", "i should not",
    "i have to decline", "i have to refuse",
)


def _refusal_score(text: str) -> float:
    t = text.strip().lower()[:400]
    return float(any(p in t for p in REFUSAL_PHRASES))


CONCEPTS = {
    "spanish":  Concept("spanish",  SPANISH_POS, _spanish_score),
    "french":   Concept("french",   FRENCH_POS,  _french_score),
    "code":     Concept("code",     CODE_POS,    _code_score),
    "uppercase":Concept("uppercase",UPPER_POS,   _upper_score),
    "refusal":  Concept("refusal",  REFUSAL_POS, _refusal_score),
}


# Neutral generation prompts. Drawn so that the unsteered IT model produces
# generic English prose, leaving a wide gap for steered concept signal.
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
    "Tell me about your favourite kind of music.",
    "Describe a beautiful sunset.",
    "Suggest a few ways to be more productive.",
    "Tell me about a piece of advice you'd give a child.",
    "Describe the perfect cup of tea.",
    "Tell me a story about a forest in autumn.",
    "What makes a city interesting to live in?",
    "Describe a typical farmers' market scene.",
    "Tell me about an everyday object you find well-designed.",
    "Recommend a way to relax after a stressful day.",
]


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _format_chat(model, prompt: str) -> str:
    try:
        return model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


def _activations_for(model, prompts: list[str], hook_name: str,
                     batch_size: int = 8) -> np.ndarray:
    """Per-prompt mean-of-final-half-tokens activation at ``hook_name``."""
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
        a = cache["a"][0]  # (T, D)
        # Take the back half of tokens — the concept-relevant content sits in
        # the body of the prompt, not BOS / chat-template scaffolding.
        half = a.shape[0] // 2
        acts.append(a[half:].mean(dim=0).numpy())
    return np.stack(acts)


def _select_partition(dictionary, model, hook_name, concept: Concept,
                      negatives: list[str]) -> tuple[int, float]:
    """AxBench-style: pick the partition with max cosine-contrast (pos − neg)."""
    pos_acts = _activations_for(model, concept.positives, hook_name)
    neg_acts = _activations_for(model, negatives, hook_name)
    # Cosine readout: center, L2-normalise, project onto unit partition dirs.
    def to_unit_centered(x):
        c = x - dictionary.center
        n = np.linalg.norm(c, axis=-1, keepdims=True) + 1e-12
        return c / n
    dirs = np.stack([p.mean_member_direction
                     for p in dictionary.partitions]).astype(np.float32)
    pos_cos = to_unit_centered(pos_acts) @ dirs.T  # (N, K)
    neg_cos = to_unit_centered(neg_acts) @ dirs.T
    contrast = pos_cos.mean(axis=0) - neg_cos.mean(axis=0)
    best = int(np.argmax(contrast))
    return best, float(contrast[best])


def _generate_steered(model, prompts: list[str], hook_name: str,
                      direction: torch.Tensor | None,
                      alpha: float, max_new_tokens: int = 80,
                      batch_size: int = 8,
                      ) -> list[str]:
    """Generate from `prompts` (batched) with `+= alpha * direction` at hook_name.

    direction: unit vector in centered space. Pass None or alpha=0 to skip
    the hook entirely (baseline).
    """
    tokenizer = model.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    out_texts: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        formatted = [_format_chat(model, p) for p in chunk]
        # Left-pad so all sequences share a final position; needed because
        # HookedTransformer.generate appends new tokens at the right edge.
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


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--model-short", default="gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--dict-path", required=True)
    parser.add_argument("--alphas", default="0,0.5,1,2,4,8",
                        help="Comma-separated alpha sweep.")
    parser.add_argument("--concepts", default=None,
                        help="Comma-separated subset of concepts to run "
                             "(default: all). Names: spanish,french,code,"
                             "uppercase,refusal.")
    parser.add_argument("--max-seeds", type=int, default=None,
                        help="Cap pos/neg seed counts for partition "
                             "selection. None = use full sets.")
    parser.add_argument("--n-eval-prompts", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--n-random-controls", type=int, default=3)
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
        args.output_dir = Path("results/exp_concept_steering") / (
            f"{args.model_short}_L{args.layer}_seed{args.seed}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    alphas = [float(a) for a in args.alphas.split(",")]
    eval_prompts = NEUTRAL_INSTRUCTIONS[:args.n_eval_prompts]
    if args.concepts:
        wanted = {c.strip() for c in args.concepts.split(",")}
        unknown = wanted - set(CONCEPTS)
        if unknown:
            raise SystemExit(f"Unknown concepts: {unknown}. "
                             f"Available: {sorted(CONCEPTS)}")
        concepts_to_run = {k: v for k, v in CONCEPTS.items() if k in wanted}
    else:
        concepts_to_run = CONCEPTS
    if args.max_seeds:
        # Slice each concept's positives and the global negatives.
        for c in concepts_to_run.values():
            c.positives = c.positives[:args.max_seeds]
        neutral_neg = NEUTRAL_NEG[:args.max_seeds]
    else:
        neutral_neg = NEUTRAL_NEG

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
            name=f"concept_steering_{args.model_short}_L{args.layer}_seed{args.seed}",
            config=vars(args), job_type="concept_steering",
        )

    # --- For each concept: pick partition, sweep alpha ---
    all_results = {}
    pair_examples = []  # (concept, alpha, condition, prompt, gen, score)

    for cname, concept in concepts_to_run.items():
        logger.info("=== Concept: %s ===", cname)

        pid, contrast = _select_partition(dictionary, model, hook_name,
                                          concept, neutral_neg)
        logger.info("Selected partition %d (cosine contrast=%.4f)",
                    pid, contrast)

        # Concept direction: unit vector in centered space.
        ep_dir = torch.from_numpy(
            dictionary.partitions[pid].mean_member_direction.astype(np.float32)
        )

        # Pre-pick random partitions (avoid the EP-selected one).
        rand_pids = []
        candidate = list(range(n_partitions))
        candidate.remove(pid)
        rng.shuffle(candidate)
        rand_pids = candidate[:args.n_random_controls]
        rand_dirs = [
            torch.from_numpy(
                dictionary.partitions[r].mean_member_direction.astype(np.float32)
            )
            for r in rand_pids
        ]

        per_alpha = {}
        for alpha in alphas:
            # EP direction
            ep_gens = _generate_steered(
                model, eval_prompts, hook_name,
                ep_dir if alpha != 0 else None, alpha,
                max_new_tokens=args.max_new_tokens,
            )
            ep_scores = [concept.scorer(g) for g in ep_gens]

            # Random partitions (mean across n_random_controls)
            rand_score_arrays = []
            rand_gens_pooled = []
            for rd in rand_dirs:
                gens = _generate_steered(
                    model, eval_prompts, hook_name,
                    rd if alpha != 0 else None, alpha,
                    max_new_tokens=args.max_new_tokens,
                )
                scores = [concept.scorer(g) for g in gens]
                rand_score_arrays.append(scores)
                rand_gens_pooled.append(gens)
            rand_scores = np.array(rand_score_arrays).mean(axis=0)

            per_alpha[str(alpha)] = {
                "alpha": alpha,
                "ep_score_mean":   float(np.mean(ep_scores)),
                "ep_score_std":    float(np.std(ep_scores)),
                "rand_score_mean": float(np.mean(rand_scores)),
                "rand_score_std":  float(np.std(rand_scores)),
                "n_prompts": len(eval_prompts),
            }
            logger.info("  alpha=%4.2f  ep=%.3f  rand=%.3f",
                        alpha, np.mean(ep_scores), np.mean(rand_scores))

            # Stash a few example generations for the wandb table.
            for i, p in enumerate(eval_prompts[:5]):
                pair_examples.append({
                    "concept": cname, "alpha": alpha,
                    "condition": "ep", "prompt": p,
                    "gen": ep_gens[i], "score": float(ep_scores[i]),
                })
                pair_examples.append({
                    "concept": cname, "alpha": alpha,
                    "condition": "rand", "prompt": p,
                    "gen": rand_gens_pooled[0][i],
                    "score": float(rand_score_arrays[0][i]),
                })

            if args.wandb:
                import wandb
                wandb.log({
                    f"{cname}/alpha":           alpha,
                    f"{cname}/ep_score_mean":   per_alpha[str(alpha)]["ep_score_mean"],
                    f"{cname}/ep_score_std":    per_alpha[str(alpha)]["ep_score_std"],
                    f"{cname}/rand_score_mean": per_alpha[str(alpha)]["rand_score_mean"],
                    f"{cname}/rand_score_std":  per_alpha[str(alpha)]["rand_score_std"],
                    f"{cname}/lift": (per_alpha[str(alpha)]["ep_score_mean"]
                                      - per_alpha[str(alpha)]["rand_score_mean"]),
                })

        all_results[cname] = {
            "partition_id": pid,
            "cosine_contrast": contrast,
            "rand_partition_ids": rand_pids,
            "per_alpha": per_alpha,
        }

    # --- Persist + headline ---
    payload = {
        "config": vars(args) | {"output_dir": str(args.output_dir),
                                "dict_path": str(args.dict_path)},
        "n_partitions": n_partitions,
        "alphas": alphas,
        "results": all_results,
    }
    out_path = args.output_dir / "concept_steering.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Saved results to %s", out_path)

    # Concise summary table.
    logger.info("\n=== HEADLINE ===")
    logger.info(f"{'concept':<10} {'pid':>5} {'contrast':>10} {'best alpha':>12} {'EP score':>10} {'rand score':>12} {'lift':>10}")
    headlines = {}
    for cname, r in all_results.items():
        # Best alpha = the alpha that gives the largest EP lift over random.
        best_a, best_lift = None, -np.inf
        for a, v in r["per_alpha"].items():
            lift = v["ep_score_mean"] - v["rand_score_mean"]
            if lift > best_lift:
                best_lift, best_a = lift, a
        v = r["per_alpha"][best_a]
        logger.info("%-10s %5d %10.3f %12s %10.3f %12.3f %10.3f",
                    cname, r["partition_id"], r["cosine_contrast"],
                    best_a, v["ep_score_mean"], v["rand_score_mean"],
                    v["ep_score_mean"] - v["rand_score_mean"])
        headlines[cname] = {
            "partition_id": r["partition_id"],
            "best_alpha": float(best_a),
            "ep_score_at_best": v["ep_score_mean"],
            "rand_score_at_best": v["rand_score_mean"],
            "lift": v["ep_score_mean"] - v["rand_score_mean"],
        }

    payload["headlines"] = headlines
    payload["examples"] = pair_examples
    out_path.write_text(json.dumps(payload, indent=2, default=str))

    if args.wandb:
        import wandb
        # Headline summary table.
        cols = ["concept", "partition_id", "best_alpha",
                "ep_score_at_best", "rand_score_at_best", "lift"]
        rows = [[c, h["partition_id"], h["best_alpha"],
                 h["ep_score_at_best"], h["rand_score_at_best"], h["lift"]]
                for c, h in headlines.items()]
        wandb.log({"headline": wandb.Table(columns=cols, data=rows)})

        # Example generation table.
        ex_cols = ["concept", "alpha", "condition", "prompt", "gen", "score"]
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
