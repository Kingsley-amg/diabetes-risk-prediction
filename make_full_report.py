"""
Build a comprehensive, multi-page technical report (PDF) for the
diabetes risk prediction project, with interpretation of every result.
Run:  python make_full_report.py  ->  report/Diabetes_Risk_Prediction_Full_Report.pdf
"""
import json
from pathlib import Path
from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak, ListFlowable, ListItem)

FIG = Path("outputs")
OUT = Path("report"); OUT.mkdir(exist_ok=True)
m = json.load(open(FIG / "metrics.json"))
rf = m["models"]["Random Forest"]; lr = m["models"]["Logistic Regression"]
cr = m["classification_report"]
AUTHOR = "Kingsley Amegah"
INK = colors.HexColor("#1a2330"); ACCENT = colors.HexColor("#e31a1c")
MUTED = colors.HexColor("#5b6573")

# ---------- styles ----------
ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=10.5, leading=15.5,
                      alignment=TA_JUSTIFY, spaceAfter=8, textColor=colors.HexColor("#23292f"))
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, leading=19,
                    textColor=INK, spaceBefore=14, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, leading=16,
                    textColor=ACCENT, spaceBefore=10, spaceAfter=4)
cap = ParagraphStyle("cap", parent=ss["BodyText"], fontSize=8.5, leading=11,
                     textColor=MUTED, alignment=TA_CENTER, spaceBefore=3, spaceAfter=12)
kicker = ParagraphStyle("kicker", parent=ss["BodyText"], fontSize=10, textColor=ACCENT)

def P(t): return Paragraph(t, body)
def H1(t): return Paragraph(t, h1)
def H2(t): return Paragraph(t, h2)
def bullets(items):
    return ListFlowable([ListItem(Paragraph(t, body), leftIndent=6, value="•")
                         for t in items], bulletType="bullet", start="•",
                        leftIndent=14, spaceAfter=8)

def figure(path, caption, max_w=6.6*inch, max_h=4.5*inch):
    iw, ih = PILImage.open(path).size
    ar = iw / ih
    w = max_w; h = w / ar
    if h > max_h:
        h = max_h; w = h * ar
    img = Image(str(path), width=w, height=h); img.hAlign = "CENTER"
    return [img, Paragraph(caption, cap)]

story = []

# ---------- COVER ----------
story += [Spacer(1, 1.4*inch)]
story.append(Paragraph("TECHNICAL PROJECT REPORT", ParagraphStyle(
    "kc", parent=kicker, alignment=TA_CENTER, fontSize=11)))
story.append(Spacer(1, 0.15*inch))
story.append(Paragraph("Predicting Diabetes Risk with Explainable Machine Learning",
    ParagraphStyle("title", parent=ss["Title"], fontSize=26, leading=32, textColor=INK)))
story.append(Spacer(1, 0.1*inch))
story.append(Paragraph("An end-to-end analysis of 253,680 adults from the CDC BRFSS survey",
    ParagraphStyle("sub", parent=ss["Title"], fontSize=13, leading=18, textColor=MUTED,
                   fontName="Helvetica")))
story.append(Spacer(1, 1.6*inch))
story.append(Paragraph(f"<b>{AUTHOR}</b>", ParagraphStyle(
    "auth", parent=ss["Title"], fontSize=15, textColor=INK)))
story.append(Paragraph("Health Data Scientist", ParagraphStyle(
    "role", parent=ss["Title"], fontSize=11, textColor=MUTED, fontName="Helvetica")))
story.append(Spacer(1, 0.3*inch))
story.append(Paragraph("Python · scikit-learn · SHAP &nbsp;|&nbsp; "
    "github.com/Kingsley-amg/diabetes-risk-prediction", ParagraphStyle(
    "foot", parent=ss["Title"], fontSize=9.5, textColor=MUTED, fontName="Helvetica")))
story.append(PageBreak())

# ---------- EXECUTIVE SUMMARY ----------
story.append(H1("Executive summary"))
story.append(P(
    "This project develops and evaluates a machine-learning model that estimates an adult's "
    "risk of diabetes from 21 routine health, behavioural and demographic indicators, using "
    f"the CDC's BRFSS 2015 survey of <b>{m['n_rows']:,} respondents</b> "
    f"(diabetes prevalence {m['prevalence']*100:.1f}%). The work covers the full pipeline: "
    "exploratory analysis, model training, imbalance-aware evaluation, decision-threshold "
    "selection, and model explanation with SHAP."))
story.append(bullets([
    f"A <b>random forest</b> achieved the best discrimination: "
    f"<b>ROC-AUC {rf['roc_auc']:.2f}</b> and <b>PR-AUC {rf['pr_auc']:.2f}</b> "
    f"(about three times the {m['prevalence']:.2f} no-skill baseline). A logistic-regression "
    f"baseline was close behind (ROC-AUC {lr['roc_auc']:.2f}).",
    f"At an F1-optimal threshold ({m['chosen_threshold']:.2f}), the model identifies "
    f"<b>{cr['Diabetes']['recall']*100:.0f}% of true diabetes cases</b> (recall) at "
    f"{cr['Diabetes']['precision']*100:.0f}% precision, a recall-leaning operating point "
    "appropriate for screening.",
    "SHAP shows the strongest drivers are <b>general health, high blood pressure, BMI, high "
    "cholesterol and age</b>, consistent with the clinical literature, indicating the model "
    "learned genuine signal rather than noise.",
    "The raw probabilities are <b>over-confident</b> (a side effect of class weighting); the "
    "rank-ordering is reliable, but probability calibration is recommended before any "
    "probability-based use.",
    "The model is positioned as a <b>screening and educational aid, not a diagnostic tool.</b>"]))

# ---------- 1. BACKGROUND ----------
story.append(H1("1. Background and objective"))
story.append(P(
    "Type-2 diabetes is largely preventable, yet a substantial share of cases go undiagnosed "
    "until complications arise. Inexpensive, data-driven screening that flags high-risk "
    "individuals from information already collected in routine surveys or clinic visits can "
    "help prioritise confirmatory testing and lifestyle intervention."))
story.append(P(
    "<b>Objective:</b> build a model that predicts whether an adult has diabetes from routine "
    "indicators, evaluate it honestly given the class imbalance, and, crucially for a health "
    "setting, explain <i>why</i> it makes each prediction so that the output can be "
    "scrutinised rather than trusted blindly."))

# ---------- 2. DATA ----------
story.append(H1("2. Data"))
story.append(P(
    f"The data is the <b>CDC Diabetes Health Indicators</b> dataset (BRFSS 2015), comprising "
    f"<b>{m['n_rows':] if False else format(m['n_rows'], ',')} survey responses</b> and "
    f"{m['n_features']} predictor features plus a binary diabetes target. The 21 features span "
    "five themes:"))
story.append(bullets([
    "<b>Clinical:</b> high blood pressure, high cholesterol, cholesterol check, BMI.",
    "<b>Behaviour:</b> smoking, physical activity, fruit and vegetable intake, heavy alcohol use.",
    "<b>Conditions / mobility:</b> prior stroke, heart disease, difficulty walking.",
    "<b>Self-rated health:</b> general, mental and physical health days.",
    "<b>Access and demographics:</b> healthcare coverage, cost barriers, sex, age band, "
    "education, income."]))
story.append(P(
    f"Only <b>{m['prevalence']*100:.1f}%</b> of respondents have diabetes, so the classes are "
    "imbalanced. This is the single most important modelling consideration: a naive model that "
    "predicts 'no diabetes' for everyone would be ~86% accurate while being clinically useless. "
    "Accuracy is therefore set aside in favour of metrics that reward correctly ranking and "
    "catching the minority class."))
story += figure(FIG / "01_class_balance.png",
                "Figure 1. Class balance. The positive (diabetes) class is the minority, "
                "which dictates the evaluation strategy used throughout.", max_h=3.2*inch)

# ---------- 3. EDA ----------
story.append(H1("3. Exploratory data analysis"))
story.append(P(
    "Before modelling, the relationships between features and the target were examined. "
    "Respondents with diabetes tend to have higher BMI, poorer self-rated general health "
    "(higher values denote worse health), older age, and a greater prevalence of high blood "
    "pressure, all directions expected from clinical knowledge."))
story += figure(FIG / "02_features_by_target.png",
                "Figure 2. Distribution of key features by diabetes status (No vs. Yes). "
                "Clear separation in BMI, general health, age and blood pressure.", max_h=2.6*inch)
story.append(P(
    "Ranking each feature by its linear correlation with the target reinforces this picture: "
    "general health, high blood pressure, BMI, difficulty walking, high cholesterol and age "
    "are the strongest positive correlates, while income, education and physical activity are "
    "negatively associated with diabetes, echoing the well-documented social gradient in "
    "chronic disease."))
story += figure(FIG / "03_feature_correlation.png",
                "Figure 3. Correlation of each feature with the diabetes target. "
                "Red = positive association, blue = negative.", max_h=4.6*inch)

# ---------- 4. METHODOLOGY ----------
story.append(H1("4. Methodology"))
story.append(H2("4.1 Train / test split and preprocessing"))
story.append(P(
    "The data was split 75/25 into training and test sets using <b>stratified</b> sampling so "
    "that the 14% prevalence is preserved in both (test set: 63,420 respondents). Continuous "
    "features were standardised for the logistic-regression model; the tree-based model "
    "requires no scaling."))
story.append(H2("4.2 Models"))
story.append(bullets([
    "<b>Logistic regression</b> (baseline), interpretable linear model, with "
    "<i>class_weight = balanced</i> to counter the imbalance.",
    "<b>Random forest</b>, 300 trees, max depth 12, minimum 20 samples per leaf, also "
    "class-weighted. Captures non-linearities and feature interactions."]))
story.append(H2("4.3 Evaluation strategy"))
story.append(P(
    "Models are compared on <b>ROC-AUC</b> (overall ranking ability) and <b>PR-AUC / average "
    "precision</b> (more informative than ROC under heavy imbalance). The probability output of "
    "the best model is then converted to a yes/no decision using a threshold chosen to "
    f"<b>maximise the F1 score</b> (balancing precision and recall), which gave "
    f"{m['chosen_threshold']:.2f} rather than the naive 0.50. Calibration and a confusion "
    "matrix complete the picture, and SHAP is used for explanation."))

# ---------- 5. RESULTS ----------
story.append(H1("5. Results and interpretation"))

story.append(H2("5.1 Discrimination"))
tbl = Table([
    ["Model", "ROC-AUC", "PR-AUC"],
    ["Logistic regression (baseline)", f"{lr['roc_auc']:.3f}", f"{lr['pr_auc']:.3f}"],
    ["Random forest (best)", f"{rf['roc_auc']:.3f}", f"{rf['pr_auc']:.3f}"],
    ["No-skill baseline", "0.500", f"{m['prevalence']:.3f}"],
], colWidths=[3.1*inch, 1.5*inch, 1.5*inch])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), INK),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f5f7")]),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d6dbe0")),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
story += [tbl, Spacer(1, 8)]
story.append(P(
    f"The random forest reaches <b>ROC-AUC {rf['roc_auc']:.2f}</b>, indicating good ability to "
    "rank a randomly chosen diabetic above a randomly chosen non-diabetic. More tellingly for "
    f"imbalanced data, its <b>PR-AUC of {rf['pr_auc']:.2f}</b> is roughly three times the "
    f"{m['prevalence']:.2f} no-skill baseline. The logistic baseline is only marginally behind, "
    "which is itself an informative result: most of the predictive signal is captured by simple "
    "linear relationships, and the tree model's added value from interactions is modest."))
story += figure(FIG / "04_roc_pr_curves.png",
                "Figure 4. ROC (left) and precision-recall (right) curves. The PR curve sits "
                "well above the dashed no-skill line across the recall range.", max_h=2.7*inch)

story.append(H2("5.2 Operating point and confusion matrix"))
story.append(P(
    f"Applying the F1-optimal threshold ({m['chosen_threshold']:.2f}) yields, for the diabetes "
    f"class, a <b>recall of {cr['Diabetes']['recall']*100:.0f}%</b> and a "
    f"<b>precision of {cr['Diabetes']['precision']*100:.0f}%</b>. In plain terms: the model "
    f"correctly flags about {cr['Diabetes']['recall']*100:.0f} of every 100 people who truly "
    "have diabetes, but among everyone it flags, roughly "
    f"{cr['Diabetes']['precision']*100:.0f}% actually have the disease. This precision-recall "
    "trade-off is deliberate and sensible for a <b>screening</b> tool, where missing a true "
    "case (a false negative) is more costly than a false alarm that a confirmatory test can "
    "later rule out."))
story += figure(FIG / "05_confusion_matrix.png",
                "Figure 5. Confusion matrix at the chosen threshold (test set, 63,420 people).",
                max_h=3.2*inch)

story.append(H2("5.3 Probability calibration"))
story.append(P(
    "A model can rank well yet still output unreliable probabilities. The calibration curve "
    "shows the random forest is <b>not well calibrated</b>: the curve sits below the diagonal, "
    "meaning the model <b>over-estimates</b> risk (e.g. people it scores at 0.55 actually have "
    "diabetes only ~19% of the time). This is an expected consequence of using class weighting "
    "to fight imbalance. The practical implication is important and honest: the model's "
    "<b>rank-ordering is trustworthy</b> (so AUC, recall and the threshold logic hold), but the "
    "raw probabilities should not be read as literal risks. A natural next step is to wrap the "
    "model in probability calibration (e.g. isotonic regression via CalibratedClassifierCV)."))
story += figure(FIG / "06_calibration.png",
                "Figure 6. Calibration curve. Points below the diagonal indicate the model "
                "systematically over-estimates probability.", max_h=3.4*inch)

story.append(H2("5.4 Explainability with SHAP"))
story.append(P(
    "SHAP values quantify how much each feature pushes an individual's predicted risk up or "
    "down, and aggregate to a global importance ranking. The strongest drivers are "
    "<b>general health, high blood pressure, BMI, high cholesterol and age</b>, followed by "
    "difficulty walking, income and heart disease."))
story += figure(FIG / "07_shap_importance.png",
                "Figure 7. Global feature importance (mean absolute SHAP value).", max_h=3.6*inch)
story.append(P(
    "The beeswarm plot adds direction. High feature values (red) for poor general health, "
    "high BMI, presence of high blood pressure and cholesterol, and older age all push risk "
    "<i>upward</i>; the reverse pushes it down. Because these directions match established "
    "medical understanding, the explanations build trust that the model is learning real "
    "physiology and socio-economic gradients rather than artefacts of the survey."))
story += figure(FIG / "08_shap_beeswarm.png",
                "Figure 8. SHAP beeswarm: each point is one person; colour is the feature "
                "value; position shows that feature's effect on their predicted risk.",
                max_h=4.8*inch)

# ---------- 6. DISCUSSION ----------
story.append(H1("6. Discussion"))
story.append(P(
    "Three findings stand out. First, <b>self-rated general health is the single most powerful "
    "predictor</b>, a one-question, zero-cost signal that rivals clinical measurements, which "
    "is valuable for low-resource screening. Second, the <b>logistic model nearly matches the "
    "random forest</b>, so a fully transparent linear model is a defensible deployment choice "
    "where interpretability or auditability is paramount. Third, the performance level "
    "(ROC-AUC ~0.83) is consistent with published results on this self-reported dataset, "
    "suggesting the ceiling here is set by the data rather than the algorithm."))

# ---------- 7. LIMITATIONS ----------
story.append(H1("7. Limitations and responsible use"))
story.append(bullets([
    "<b>Self-reported survey data</b> carries recall and social-desirability bias; BMI, "
    "diagnoses and behaviours are not clinically verified.",
    "<b>Cross-sectional design</b> means the model captures association, not causation, it "
    "cannot say that changing a factor will change risk.",
    "<b>Population and era specific</b> (US adults, 2015); performance may not transfer to "
    "other countries or healthcare systems without revalidation.",
    "The target <b>combines diabetes types</b> and includes pre-existing diagnoses, so the "
    "task is closer to detection than true future-onset prediction.",
    "<b>Probabilities are uncalibrated</b> (Section 5.3) and should be calibrated before use.",
    "This is a <b>screening and educational artefact, not a medical device</b>. It must not be "
    "used for individual diagnosis or treatment decisions."]))

# ---------- 8. CONCLUSION ----------
story.append(H1("8. Conclusion"))
story.append(P(
    "This project delivers an explainable, honestly-evaluated diabetes risk model. The random "
    f"forest discriminates well (ROC-AUC {rf['roc_auc']:.2f}, PR-AUC {rf['pr_auc']:.2f}) and, at "
    f"a recall-leaning threshold, catches {cr['Diabetes']['recall']*100:.0f}% of true cases, "
    "an operating point suited to screening. SHAP confirms the model relies on clinically "
    "coherent factors, and the analysis is transparent about its weaknesses, notably "
    "probability calibration. Clear next steps are probability calibration, tuning the decision "
    "threshold to real clinical costs, and external validation on an independent population."))
story.append(H2("Reproducibility"))
story.append(P(
    "The entire analysis is reproducible from a single script (<i>analysis.py</i>) using Python "
    "with pandas, scikit-learn, SHAP and matplotlib. Code, data and all figures are available "
    "at github.com/Kingsley-amg/diabetes-risk-prediction."))

# ---------- footer with page numbers + author (no third-party branding) ----------
def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d6dbe0")); canvas.setLineWidth(0.5)
    canvas.line(0.9*inch, 0.7*inch, letter[0]-0.9*inch, 0.7*inch)
    canvas.setFont("Helvetica", 8); canvas.setFillColor(MUTED)
    canvas.drawString(0.9*inch, 0.55*inch, f"Diabetes Risk Prediction  |  {AUTHOR}")
    canvas.drawRightString(letter[0]-0.9*inch, 0.55*inch, f"Page {doc.page}")
    canvas.restoreState()

doc = SimpleDocTemplate(
    str(OUT / "Diabetes_Risk_Prediction_Full_Report.pdf"), pagesize=letter,
    leftMargin=0.9*inch, rightMargin=0.9*inch, topMargin=0.9*inch, bottomMargin=0.9*inch,
    title="Diabetes Risk Prediction - Technical Report", author=AUTHOR,
    subject="Explainable machine-learning project report", creator=AUTHOR)
doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=footer)
print("Wrote", OUT / "Diabetes_Risk_Prediction_Full_Report.pdf")
