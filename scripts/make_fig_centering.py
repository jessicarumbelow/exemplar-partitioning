import matplotlib.pyplot as plt
import numpy as np

# Three vectors, all clustered near a shared mean μ along x.
# Two of them (u, v) deviate upward; one (w) deviates slightly along μ.
mu = np.array([10.0, 0.0])
u = mu + np.array([0.0, 1.0])
v = mu + np.array([0.0, 0.5])
w = mu + np.array([0.1, 0.05])


def cos(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


# Uncentered cosines (vectors from origin)
cuv, cuw, cvw = cos(u, v), cos(u, w), cos(v, w)

# Centered: subtract μ, then cosine
du, dv, dw = u - mu, v - mu, w - mu
dcuv, dcuw, dcvw = cos(du, dv), cos(du, dw), cos(dv, dw)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# -------- Left panel: uncentered --------
ax = axes[0]
ax.set_title("Uncentered cosine\n(vectors from origin)", fontsize=13)

# μ as a thin dashed arrow
ax.annotate("", xy=mu, xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2, ls="--"))
ax.text(5.0, -0.45, "μ (shared mean)", color="gray", fontsize=10)

for vec, name, color in [(u, "u", "C0"), (v, "v", "C1"), (w, "w", "C2")]:
    ax.annotate("", xy=vec, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=color, lw=2))
    ax.text(vec[0] + 0.3, vec[1] + 0.05, name, color=color,
            fontsize=14, fontweight="bold")

ax.set_xlim(-1, 13)
ax.set_ylim(-3, 3)
ax.axhline(0, color="black", linewidth=0.3)
ax.axvline(0, color="black", linewidth=0.3)
ax.set_aspect("equal")
ax.grid(alpha=0.2)
ax.text(0.2, -2.4,
        f"cos(u,v) = {cuv:.4f}\ncos(u,w) = {cuw:.4f}\ncos(v,w) = {cvw:.4f}  ← largest",
        fontsize=10, family="monospace")
ax.text(0.2, 2.2, "v's nearest neighbour: w",
        fontsize=12, color="darkred", fontweight="bold")

# -------- Right panel: centered, zoomed in --------
ax = axes[1]
ax.set_title("Centered cosine\n(vectors from μ as new origin)", fontsize=13)

for vec, name, color in [(du, "u − μ", "C0"), (dv, "v − μ", "C1"), (dw, "w − μ", "C2")]:
    ax.annotate("", xy=vec, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=color, lw=2))
    ax.text(vec[0] + 0.04, vec[1] + 0.02, name, color=color,
            fontsize=14, fontweight="bold")

ax.set_xlim(-0.4, 0.6)
ax.set_ylim(-0.4, 1.3)
ax.axhline(0, color="black", linewidth=0.3)
ax.axvline(0, color="black", linewidth=0.3)
ax.set_aspect("equal")
ax.grid(alpha=0.2)
ax.text(-0.35, -0.32,
        f"cos(u,v) = {dcuv:.4f}  ← largest\ncos(u,w) = {dcuw:.4f}\ncos(v,w) = {dcvw:.4f}",
        fontsize=10, family="monospace")
ax.text(-0.35, 1.18, "v's nearest neighbour: u",
        fontsize=12, color="darkgreen", fontweight="bold")

plt.suptitle(
    "Same three vectors, two metrics. Centering reorders who's nearest to whom.",
    fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("/tmp/centering_diagram.png", dpi=130, bbox_inches="tight")
print(f"uncentered: cos(u,v)={cuv:.4f}  cos(u,w)={cuw:.4f}  cos(v,w)={cvw:.4f}")
print(f"centered:   cos(u,v)={dcuv:.4f}  cos(u,w)={dcuw:.4f}  cos(v,w)={dcvw:.4f}")
