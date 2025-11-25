import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ==================== USER CONFIGURATION ====================
ALPHA = 0.05  # Significance threshold
MULTIPLE_TESTING_CORRECTION = True  # Bonferroni correction for multiple comparisons
FIG_SIZE = (9, 6)  # Figure dimensions (width, height) in inches
DPI = 300  # Resolution for PNG output
SHOW_INDIVIDUAL_POINTS = True  # Show individual biological replicates
ERROR_BAR_TYPE = "SEM"  # 'SEM' or 'SD'
# ==========================================================

# ==================== EXPERIMENTAL DATA =====================
# Data structure: 9 values per condition = 3 experiments × 3 technical replicates
# Order: [bio1_tech1, bio1_tech2, bio1_tech3, bio2_tech1, bio2_tech2, bio2_tech3, bio3_tech1, bio3_tech2, bio3_tech3]

data = {
    "Mock": [78.82, 79.73, 80.88, 59.12, 58.14, 56.86, 80.88, 79.73, 78.82],
    "DMSO": [77.69, 79.19, 81.58, 55.42, 51.12, 47.57, 81.58, 79.19, 77.69],
    "2.5 mM": [62.5, 64.91, 58.33, 51.67, 45.57, 44.83, 65.29, 64.91, 62.5],
    "5 mM": [59.01, 53.33, 50, 50, 44.9, 43.06, 59.01, 53.33, 50],
    "7.5 mM": [45.57, 53.62, 64.29, 36.14, 35.21, 26.32, 45.57, 39.68, 38.1],
    "10 mM": [34.38, 35.44, 37.88, 16.67, 16.28, 15.58, 34.38, 27.27, 23.96],
}
# ==========================================================


def reshape_to_experiments(data_dict):
    """Reshape flattened data into experiments (3 experiments × 3 replicates)."""
    reshaped = {}
    for condition, values in data_dict.items():
        reshaped[condition] = [
            values[0:3],  # bio1: replicates 1-3
            values[3:6],  # bio2: replicates 4-6
            values[6:9],  # bio3: replicates 7-9
        ]
    return reshaped


def calculate_normalized_ratios(data_dict):
    """
    Normalize each treatment to its paired DMSO control within each experiment.
    Returns: ratios_dict where each condition has 3 biological ratio values (% of DMSO)
    """
    ratios = {}

    # Calculate ratios for each condition (including Mock)
    for condition in data_dict:
        condition_ratios = []
        for i in range(3):  # 3 experiments
            if condition == "DMSO":
                # DMSO is always 100% relative to itself
                ratio = 100.0
            else:
                # Calculate mean of treatment and paired DMSO for this experiment
                treatment_mean = np.mean(data_dict[condition][i])
                dmso_mean = np.mean(data_dict["DMSO"][i])
                ratio = treatment_mean / dmso_mean * 100
            condition_ratios.append(ratio)
        ratios[condition] = condition_ratios

    return ratios


# Reshape data and calculate normalized ratios
data_reshaped = reshape_to_experiments(data)
ratios_dict = calculate_normalized_ratios(data_reshaped)

# Calculate statistics for plotting
conditions = ["Mock", "DMSO", "2.5 mM", "5 mM", "7.5 mM", "10 mM"]
means = [np.mean(ratios_dict[c]) for c in conditions]
stds = [np.std(ratios_dict[c], ddof=1) for c in conditions]
sem_values = [stats.sem(ratios_dict[c]) for c in conditions]

# Perform one-sample t-tests vs 100% (DMSO baseline)
p_values = {}
n_comparisons = len([c for c in conditions if c != "DMSO"])  # 5 comparisons

for c in conditions:
    if c == "DMSO":
        p_values[c] = np.nan
    else:
        t_stat, p_val = stats.ttest_1samp(ratios_dict[c], 100.0)
        # Apply Bonferroni correction if enabled
        if MULTIPLE_TESTING_CORRECTION:
            p_val = min(p_val * n_comparisons, 1.0)
        p_values[c] = p_val


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

fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
x_positions = np.arange(len(conditions))

# Color scheme: controls in gray/red, treatments in blue gradient
control_colors = {"Mock": "#7f8c8d", "DMSO": "#e74c3c"}
treatment_colors = ["#3498db", "#2980b9", "#1f618d", "#154360"]

# FIXED: Proper color assignment
bar_colors = []
treatment_counter = 0  # Counter specifically for treatment colors
for c in conditions:
    if c in control_colors:
        bar_colors.append(control_colors[c])
    else:
        bar_colors.append(treatment_colors[treatment_counter])
        treatment_counter += 1  # Only increment for treatments

# Select error bar values
yerr_values = sem_values if ERROR_BAR_TYPE == "SEM" else stds

# Plot bars with error bars
bars = ax.bar(
    x_positions,
    means,
    yerr=yerr_values,
    capsize=6,
    color=bar_colors,
    alpha=0.8,
    edgecolor="black",
    linewidth=1,
    error_kw={"elinewidth": 1.2, "capthick": 1.2},
)

# Add individual data points if requested
if SHOW_INDIVIDUAL_POINTS:
    for i, c in enumerate(conditions):
        ax.plot(
            [i] * len(ratios_dict[c]),
            ratios_dict[c],
            "o",
            color="white",
            markeredgecolor="black",
            markersize=7,
            markeredgewidth=1.2,
            alpha=0.9,
        )

# Labels and title
ax.set_xlabel("Treatment Conditions", fontweight="bold", fontsize=12)
ax.set_ylabel("Sperm Motility (% of DMSO Control)", fontweight="bold", fontsize=12)
ax.set_title(
    "TRiCi Inhibits Sperm Motility in a Dose-Dependent Manner",
    fontweight="bold",
    fontsize=14,
    pad=20,
)
ax.set_xticks(x_positions)
ax.set_xticklabels(conditions, rotation=15, ha="right", fontsize=10)

# Grid and spines
ax.grid(axis="y", linestyle="--", alpha=0.5, linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)

# Add horizontal reference line at 100%
ax.axhline(y=100, color="gray", linestyle=":", alpha=0.5, linewidth=2, zorder=0)

# Add significance stars above bars
max_bar_height = max(means[i] + yerr_values[i] for i in range(len(conditions)))
star_y_position = max_bar_height * 1.12

for i, (cond, stars) in enumerate(zip(conditions, star_labels)):
    if stars and stars != "ns":
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

        line_bottom = means[i] + yerr_values[i] + max_bar_height * 0.02
        line_top = star_y_position * 0.98
        ax.plot([i, i], [line_bottom, line_top], color="red", linewidth=0.8, alpha=0.7)

# Add significance legend
legend_text = f"Statistical significance vs. DMSO (one-sample t-test):\n* p<0.05, **p<0.01, ***p<0.001"
if MULTIPLE_TESTING_CORRECTION:
    legend_text += "\n(Bonferroni-corrected for multiple comparisons)"
ax.text(
    0.02,
    0.98,
    legend_text,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)

# Adjust y-axis to make room for stars
current_ylim = ax.get_ylim()
ax.set_ylim(current_ylim[0], current_ylim[1] * 1.18)

# Adjust layout and save
plt.tight_layout()

# Save as editable SVG
plt.savefig(
    "bar_plot_TRiCi_normalized.svg",
    format="svg",
    dpi=DPI,
    bbox_inches="tight",
    transparent=False,
)

# Also save as PNG for preview
plt.savefig("bar_plot_TRiCi_normalized.png", format="png", dpi=DPI, bbox_inches="tight")

# ==================== STATISTICAL SUMMARY ==================

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY (Normalized to Paired DMSO, n=3 biological replicates)")
if MULTIPLE_TESTING_CORRECTION:
    print("Multiple testing correction: Bonferroni applied")
print("=" * 70)
print(
    f"{'Condition':<10} {'Mean(%)':<10} {'SD(%)':<10} {'SEM(%)':<10} {'p-value':<12} {'Stars'}"
)
print("-" * 70)
for i, c in enumerate(conditions):
    print(
        f"{c:<10} {means[i]:<10.2f} {stds[i]:<10.2f} {sem_values[i]:<10.2f} {p_values[c]:<12.4f} {star_labels[i]}"
    )

plt.show()
