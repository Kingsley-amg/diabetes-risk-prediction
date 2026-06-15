"""
Build a polished, LinkedIn-ready PDF carousel report of the diabetes project.
Each PDF page is a square "slide". Run:  python make_report_pdf.py
Output: report/Diabetes_Risk_Prediction_Report.pdf
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path("report"); OUT.mkdir(exist_ok=True)
FIG = Path("outputs")
metrics = json.load(open(FIG / "metrics.json"))

INK = "#1a2330"; ACCENT = "#e31a1c"; MUTED = "#5b6573"; LIGHT = "#eef1f4"

def new_slide(footer=True):
    fig = plt.figure(figsize=(10, 10), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if footer:
        ax.add_patch(Rectangle((0, 0), 1, 0.045, color=INK))
        ax.text(0.04, 0.022, "Diabetes Risk Prediction  ·  Kingsley Amegah",
                color="white", fontsize=10, va="center")
        ax.text(0.96, 0.022, "github.com/Kingsley-amg/diabetes-risk-prediction",
                color="#aab4c2", fontsize=9, va="center", ha="right")
    return fig, ax

def heading(ax, kicker, title):
    ax.add_patch(Rectangle((0.04, 0.88), 0.06, 0.012, color=ACCENT))
    ax.text(0.04, 0.83, kicker, color=ACCENT, fontsize=13, fontweight="bold")
    ax.text(0.04, 0.75, title, color=INK, fontsize=27, fontweight="bold",
            va="top", wrap=True)

def bullets(ax, items, y=0.66, dy=0.085, size=15):
    for i, t in enumerate(items):
        ax.text(0.06, y - i * dy, "•", color=ACCENT, fontsize=size, fontweight="bold")
        ax.text(0.10, y - i * dy, t, color="#2a2f37", fontsize=size, va="top", wrap=True)

def place_image(fig, path, box):
    """Place image in [l,b,w,h] preserving aspect ratio, centered."""
    import matplotlib.image as mpimg
    img = mpimg.imread(path)
    ih, iw = img.shape[0], img.shape[1]
    l, b, w, h = box
    box_ar, img_ar = w / h, iw / ih
    if img_ar > box_ar:        # image wider -> fit width
        nw = w; nh = w / img_ar; nl = l; nb = b + (h - nh) / 2
    else:                      # image taller -> fit height
        nh = h; nw = h * img_ar; nb = b; nl = l + (w - nw) / 2
    a = fig.add_axes([nl, nb, nw, nh]); a.axis("off"); a.imshow(img)

pdf_path = OUT / "Diabetes_Risk_Prediction_Report.pdf"
with PdfPages(pdf_path) as pdf:

    # --- Slide 1: Title ---
    fig, ax = new_slide(footer=False)
    ax.add_patch(Rectangle((0, 0), 1, 1, color=INK))
    ax.add_patch(Rectangle((0.08, 0.62), 0.10, 0.018, color=ACCENT))
    ax.text(0.08, 0.50, "Predicting Diabetes Risk\nwith Explainable AI",
            color="white", fontsize=40, fontweight="bold", va="center")
    ax.text(0.08, 0.34, "An end-to-end machine-learning project on 253,680 US adults",
            color="#aab4c2", fontsize=17)
    ax.text(0.08, 0.14, "Kingsley Amegah", color="white", fontsize=18, fontweight="bold")
    ax.text(0.08, 0.10, "Health Data Scientist  ·  Python · scikit-learn · SHAP",
            color="#aab4c2", fontsize=13)
    pdf.savefig(fig); plt.close(fig)

    # --- Slide 2: The problem ---
    fig, ax = new_slide()
    heading(ax, "THE PROBLEM", "Can we flag diabetes risk early\nfrom routine health data?")
    bullets(ax, [
        "Goal: predict whether an adult has diabetes from everyday health and\n   lifestyle indicators (blood pressure, BMI, general health, age...).",
        "Data: CDC BRFSS 2015 - a survey of 253,680 US adults, 21 features.",
        "The catch: only ~14% have diabetes. On imbalanced data, 'accuracy'\n   is misleading - a model guessing 'no' for everyone scores 86%.",
        "So the real challenge is catching true cases without crying wolf."],
        y=0.61, dy=0.115)
    pdf.savefig(fig); plt.close(fig)

    # --- Slide 3: Approach ---
    fig, ax = new_slide()
    heading(ax, "APPROACH", "An honest, end-to-end\nML workflow")
    bullets(ax, [
        "Exploratory analysis - distributions and correlations with the target.",
        "Models - logistic regression (baseline) and a random forest.",
        "Imbalance-aware evaluation - PR-AUC, calibration, and a recall-tuned\n   decision threshold instead of plain accuracy.",
        "Explainability - SHAP, to show what drives each prediction.",
        "Responsible framing - a screening aid, not a diagnostic tool."],
        y=0.61, dy=0.105)
    pdf.savefig(fig); plt.close(fig)

    # --- Slide 4: Results ---
    fig, ax = new_slide()
    rf = metrics["models"]["Random Forest"]; lr = metrics["models"]["Logistic Regression"]
    heading(ax, "RESULTS", "Strong discrimination on\nimbalanced data")
    ax.text(0.06, 0.60, f"Random Forest:  ROC-AUC {rf['roc_auc']:.2f}   ·   PR-AUC {rf['pr_auc']:.2f}",
            fontsize=16, color=INK, fontweight="bold")
    ax.text(0.06, 0.555, "(PR-AUC vs. a 0.14 random baseline; ~62% of true cases flagged "
            "at the chosen threshold)", fontsize=12, color=MUTED)
    place_image(fig, FIG / "04_roc_pr_curves.png", [0.06, 0.09, 0.88, 0.43])
    pdf.savefig(fig); plt.close(fig)

    # --- Slide 5: SHAP ---
    fig, ax = new_slide()
    heading(ax, "EXPLAINABILITY (SHAP)", "Why the model predicts risk")
    ax.text(0.06, 0.625, "Top drivers - general health, high blood pressure, BMI, cholesterol, "
            "and age -\nmatch the clinical literature: the model learned real signal.",
            fontsize=13, color=MUTED, va="top")
    place_image(fig, FIG / "08_shap_beeswarm.png", [0.15, 0.06, 0.70, 0.50])
    pdf.savefig(fig); plt.close(fig)

    # --- Slide 6: Takeaways ---
    fig, ax = new_slide()
    heading(ax, "TAKEAWAYS", "What this project shows")
    bullets(ax, [
        "End-to-end ML: from raw survey data to an explained, evaluated model.",
        "Rigor on imbalanced health data - the metrics that actually matter.",
        "Explainable AI that a clinician could interrogate, not a black box.",
        "Built responsibly, with clear limits on how it should be used."],
        y=0.66, dy=0.10)
    ax.text(0.06, 0.20, "Full code, figures & write-up:", fontsize=14, color=INK, fontweight="bold")
    ax.text(0.06, 0.155, "github.com/Kingsley-amg/diabetes-risk-prediction",
            fontsize=15, color=ACCENT)
    pdf.savefig(fig); plt.close(fig)

print("Wrote", pdf_path)
