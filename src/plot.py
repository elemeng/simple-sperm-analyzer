import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ==================== USER CONFIGURATION ====================
PLOT_TITLE = "Sperm Motility Analysis: Mean ± SD with Significance"
REFERENCE_CONDITION = "DMSO"  # Reference group for statistical comparisons
ALPHA = 0.05  # Significance threshold
MULTIPLE_TESTING_CORRECTION = False  # Set to True for Bonferroni correction
FIG_SIZE = (9, 6)  # Figure dimensions (width, height) in inches
DPI = 300  # Resolution for PNG output
# ==========================================================

# ==================== EXPERIMENTAL DATA =====================
# Data structure: 9 replicates per condition
# [bio1-1, bio1-2, bio1-3, bio2-1, bio2-2, bio2-3, bio3-1, bio3-2, bio3-3]

data = {
    "Mock": [78.82, 79.73, 80.88, 59.12, 58.14, 56.86, 80.88, 79.73, 78.82],
    "DMSO": [77.69, 79.19, 81.58, 55.42, 51.12, 47.57, 81.58, 79.19, 77.69],
    "2.5 mM": [62.5, 64.91, 58.33, 51.67, 45.57, 44.83, 65.29, 64.91, 62.5],
    "5 mM": [59.01, 53.33, 50, 50, 44.9, 43.06, 59.01, 53.33, 50],
    "7.5 mM": [45.57, 53.62, 64.29, 36.14, 35.21, 26.32, 45.57, 39.68, 38.1],
    "10 mM": [34.38, 35.44, 37.88, 16.67, 16.28, 15.58, 34.38, 27.27, 23.96],
}
# ==========================================================
# ==================== STATISTICAL ANALYSIS ==================

# Verify reference condition exists
if REFERENCE_CONDITION not in data:
    raise ValueError(f"Reference condition '{REFERENCE_CONDITION}' not found in data!")

# Calculate descriptive statistics
conditions = list(data.keys())
means = [np.mean(data[c]) for c in conditions]
stds = [np.std(data[c], ddof=1) for c in conditions]  # Sample standard deviation

# Perform Welch's t-tests (unequal variances) vs reference
ref_values = data[REFERENCE_CONDITION]
p_values = {}

for c in conditions:
    if c == REFERENCE_CONDITION:
        p_values[c] = np.nan
    else:
        _, p_val = stats.ttest_ind(ref_values, data[c], equal_var=False)
        p_values[c] = p_val

# Apply Bonferroni correction if requested
if MULTIPLE_TESTING_CORRECTION:
    n_tests = len([p for p in p_values.values() if not np.isnan(p)])
    for c in p_values:
        if not np.isnan(p_values[c]):
            p_values[c] = min(p_values[c] * n_tests, 1.0)


# Convert p-values to star notation
def get_stars(p):
    """Convert p-value to significance stars."""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < ALPHA:
        return "*"
    return "ns"


star_labels = [get_stars(p_values[c]) for c in conditions]

# ==================== PLOT CREATION ========================

# Create figure and axis
fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
x_positions = np.arange(len(conditions))

# Color scheme: reference in red, treatments in blue
bar_colors = ["#e74c3c" if c == REFERENCE_CONDITION else "#3498db" for c in conditions]

# Plot bars with error bars (SD)
bars = ax.bar(
    x_positions,
    means,
    yerr=stds,
    capsize=6,
    color=bar_colors,
    alpha=0.8,
    edgecolor="black",
    linewidth=1,
    error_kw={"elinewidth": 1.2, "capthick": 1.2},
)

# Labels and title
ax.set_xlabel("Treatment Conditions", fontweight="bold", fontsize=12)
ax.set_ylabel("Mean ± SD", fontweight="bold", fontsize=12)
ax.set_title(PLOT_TITLE, fontweight="bold", fontsize=14, pad=15)
ax.set_xticks(x_positions)
ax.set_xticklabels(conditions, rotation=15, ha="right", fontsize=10)

# Grid and spines
ax.grid(axis="y", linestyle="--", alpha=0.5, linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)

# Add significance stars above bars
max_bar_height = max(means[i] + stds[i] for i in range(len(conditions)))
star_y_position = max_bar_height * 1.12

for i, (cond, stars) in enumerate(zip(conditions, star_labels)):
    if stars and stars != "ns":
        # Add star text (editable in SVG)
        ax.text(
            i,
            star_y_position,
            stars,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color="red",
            fontfamily="sans-serif",
        )

        # Add connecting line
        line_bottom = means[i] + stds[i] + max_bar_height * 0.02
        line_top = star_y_position * 0.98
        ax.plot([i, i], [line_bottom, line_top], color="red", linewidth=0.8, alpha=0.7)

# Add reference line for DMSO
ref_mean = means[conditions.index(REFERENCE_CONDITION)]
ax.axhline(y=ref_mean, color="gray", linestyle=":", alpha=0.5, linewidth=2)

# Add significance legend
legend_text = "Significance vs {}:\n* p<0.05, ** p<0.01, *** p<0.001".format(
    REFERENCE_CONDITION
)
if MULTIPLE_TESTING_CORRECTION:
    legend_text += "\n(Bonferroni corrected)"
ax.text(
    0.02,
    0.98,
    legend_text,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)

# Adjust layout and save
plt.tight_layout()

# Save as editable SVG (text remains as selectable text objects)
plt.savefig(
    "bar_plot_with_significance.svg",
    format="svg",
    dpi=DPI,
    bbox_inches="tight",
    transparent=False,
)

# Also save as PNG for preview
plt.savefig(
    "bar_plot_with_significance.png", format="png", dpi=DPI, bbox_inches="tight"
)

# ==================== STATISTICAL SUMMARY ==================

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY (Reference: {})".format(REFERENCE_CONDITION))
if MULTIPLE_TESTING_CORRECTION:
    print("Multiple testing correction: Bonferroni applied")
print("=" * 60)
for i, c in enumerate(conditions):
    print(
        f"{c:<8} | Mean={means[i]:6.2f} | SD={stds[i]:5.2f} | p={p_values[c]:.4f} {star_labels[i]}"
    )

plt.show()
