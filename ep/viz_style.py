"""Shared visualization style for EP plots. Matches the Disco dashboard aesthetic."""

import warnings

import numpy as np

warnings.filterwarnings("ignore", message="n_jobs value.*overridden.*random_state")

# -- Brand colors --------------------------------------------------------------

CREAM = "#fbf8f4"
SAND = "#f8f3ed"
STONE = "#f4eee6"
CHARCOAL = "#252525"
BRAND_BLUE = "#8e96c9"
BRAND_WARM = "#e48d56"

SAGE = "#8fc4a3"
WHEAT = "#dcc591"
TEAL = "#7fb8ba"
MAUVE = "#bda0bd"
PLUM = "#9b8bb5"
AMBER = "#d9b07a"
SLATE = "#9db5c9"
TAUPE = "#c9ae96"
BRICK = "#c97272"

# -- Categorical palette (10 colors, ordered) ----------------------------------

CATEGORICAL = [
    BRAND_BLUE, BRAND_WARM, TEAL, PLUM, AMBER,
    SAGE, MAUVE, WHEAT, SLATE, TAUPE,
]

# Expanded palette for more categories (adds brick + darkened variants)
CATEGORICAL_EXTENDED = CATEGORICAL + [
    BRICK, "#6b7db5", "#c46a2e", "#5a9a9c", "#7a6b9a",
    "#b8943e", "#6a9a7a", "#9a7a9a", "#b89a5a", "#7a9ab5",
]

# -- Word-level colors for the toy (animals = cool, vehicles = warm) -----------

WORD_COLORS = {
    "cat": BRAND_BLUE, "dog": PLUM, "horse": TEAL, "fish": SLATE, "bird": SAGE,
    "car": BRAND_WARM, "boat": AMBER, "train": TAUPE, "plane": WHEAT, "bike": BRICK,
    "the": "#d4cec6", "and": "#c4beb6", "or": "#c4beb6",
}

# -- Plotly layout constants ---------------------------------------------------

FONT_FAMILY = "Suisse Intl, ui-sans-serif, system-ui, sans-serif"
FONT_MONO = "Suisse Intl Mono, ui-monospace, monospace"

GRID_COLOR = "rgba(244, 238, 230, 0.3)"
TEXT_COLOR = CHARCOAL

OPACITY_DEFAULT = 0.7
OPACITY_EMPHASIZED = 0.9
OPACITY_DIMMED = 0.3

# Diverging: brand_blue → sand → brand_warm
DIVERGING_COLORSCALE = [
    [0.0, BRAND_BLUE],
    [0.5, SAND],
    [1.0, BRAND_WARM],
]

# Sequential: brand_blue → teal → darker teal
SEQUENTIAL_COLORSCALE = [
    [0.0, SAND],
    [0.3, SLATE],
    [0.6, BRAND_BLUE],
    [1.0, PLUM],
]


def base_layout(**overrides):
    """Standard Plotly layout matching the Disco dashboard."""
    layout = dict(
        font=dict(family=FONT_FAMILY, size=11, color=TEXT_COLOR),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=40, t=50, b=50),
        hovermode="closest",
    )
    layout.update(overrides)
    return layout


def axis_style(**overrides):
    """Standard axis configuration."""
    style = dict(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        title_font=dict(family=FONT_FAMILY, size=11, color=TEXT_COLOR),
        tickfont=dict(family=FONT_FAMILY, size=9, color=TEXT_COLOR),
        automargin=True,
    )
    style.update(overrides)
    return style


def token_attributions(model, text, concept_direction, hook_name, device="cpu"):
    """Compute per-token importance as cosine similarity of each position's
    activation with the concept's mean attribution direction.

    Returns list of (token_str, score) pairs. Score in [0, 1].
    """
    import torch

    tokens = model.to_tokens(text, prepend_bos=False)
    if tokens.shape[1] < 1:
        return []

    acts_ref: dict = {}

    def fwd(act, **kwargs):
        acts_ref["a"] = act

    model.reset_hooks()
    model.add_hook(hook_name, fwd, "fwd")
    with torch.no_grad():
        model(tokens)
    model.reset_hooks()

    act = acts_ref["a"][0].detach().cpu().numpy()  # (L, D)
    direction = concept_direction / (np.linalg.norm(concept_direction) + 1e-12)

    scores = []
    token_strs = [model.tokenizer.decode([t.item()]) for t in tokens[0]]
    for i in range(len(token_strs)):
        v = act[i]
        sim = np.dot(v, direction) / (np.linalg.norm(v) + 1e-12)
        scores.append(max(0, sim))

    if scores:
        max_s = max(scores) or 1.0
        scores = [s / max_s for s in scores]

    return list(zip(token_strs, scores))


def marker_style(size=7, opacity=OPACITY_DEFAULT, **overrides):
    """Standard marker configuration."""
    style = dict(
        size=size,
        opacity=opacity,
        line=dict(width=0.4, color="rgba(37,37,37,0.3)"),
    )
    style.update(overrides)
    return style
