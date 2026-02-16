
REPRODUCIBLE COLD-START RECOMMENDATION EXPERIMENTS
Extension of the Let It Go Framework with KNN-based Delta Transfer

=================================================

1. Overview
-----------
This repository reproduces and extends the experiments presented in the "Let It Go" paper
and its accompanying GitHub implementation.

The original framework addresses cold-start item recommendation using SASRec with
trainable item deltas. In this work, we evaluate an extension based on KNN-based transfer
of warm-item deltas to cold items, augmented with an optional confidence gating mechanism.

All experiments were executed and validated in a controlled Google Colab environment to
ensure reproducibility.

-------------------------------------------------

2. Original Source
------------------
The base code is taken from the official repository of the paper:
https://github.com/ArtemF42/let-it-go

The core architecture was preserved. All changes are extensions to the original logic.

-------------------------------------------------

3. Key Contributions
--------------------
- Reproduction of the original baseline experiments from the Let It Go framework.
- Extension of the trainable-delta framework with:
  * KNN-based delta transfer from warm to cold items.
  * Optional confidence gating (similarity-based or entropy-based).
- Evaluation across three datasets, each including baseline and improved sub-runs.
- Fully documented and reproducible Colab execution.

-------------------------------------------------

4. Project Structure
--------------------
let-it-go/
├── configs/
│   └── main_delta_knn_fixed.yaml     (extended configuration)
├── scripts/
│   └── run_delta_knn_fixed.py        (extended training & evaluation script)
├── source/
│   └── original framework code
├── notebooks/
│   └── experiments_colab.ipynb
└── README.txt

-------------------------------------------------

5. Hardware and Runtime Requirements
------------------------------------
Hardware:
- GPU: NVIDIA T4 / A100 (recommended)
- RAM: at least 16GB
- Disk: at least 20GB free space

Runtime:
- Python 3.9 or 3.10
- CUDA-enabled environment
- Google Colab with GPU runtime enabled

CPU-only execution is not recommended due to dataset size and training cost.

-------------------------------------------------

6. Environment Setup (Explanation)
----------------------------------
The environment is prepared inside Google Colab.

This includes:
- Activating a predefined Conda environment.
- Setting environment variables for deterministic behavior.
- Running ClearML in offline mode.
- Enabling GPU execution.

All dependency adjustments were made only to resolve compatibility issues and were
manually verified to preserve the original experimental logic.

-------------------------------------------------

7. Datasets and Experiment Design
---------------------------------
Experiments were conducted on three datasets.

For each dataset, two main variants were executed:

A) BASELINE:
- Original SASRec with trainable item deltas.
- No KNN-based transfer is applied to cold items.

B) IMPROVED (OURS):
- Cold-item deltas are initialized using KNN transfer from warm items.
- A confidence gating mechanism controls the strength of the transferred delta.

Each dataset contains multiple sub-runs, including different random seeds and multiple
values of K. The Google Colab notebook documents the exact run order and configuration.

-------------------------------------------------

8. Configuration (main_delta_knn_fixed.yaml)
--------------------------------------------
The configuration file extends the original setup by introducing a `cold_knn` section.

Key parameters include:
- Enabling or disabling KNN-based transfer.
- Number of neighbors (K).
- Temperature and scaling parameters.
- Confidence gate type and thresholds.
- Chunk size for memory-efficient similarity computation.

All original configuration options remain supported.

-------------------------------------------------

9. Training and Evaluation Logic
--------------------------------
The training script extends the original pipeline with:

- Optional KNN-based transfer of warm-item deltas to cold items.
- A gating mechanism that modulates the transferred delta based on similarity confidence.
- A constrained optimizer that enforces a maximum norm on delta embeddings.

If KNN transfer is disabled, the behavior exactly matches the original baseline.
-------------------------------------------------

11. Dataset Availability
------------------------
The datasets used in this study (raw and processed) are available via Google Drive:

https://drive.google.com/drive/folders/1Xq6CsBxf5R8_yKram9qUQvyzVt5FOyIN

The folder contains the datasets required to reproduce all experiments described
in this repository. Dataset preparation and usage are documented in the Google
Colab notebook.


-------------------------------------------------

10. Execution Flow (High-Level)
-------------------------------
To reproduce the experiments:

1. Open the provided Google Colab notebook.
2. Enable GPU runtime.
3. Prepare the environment and datasets.
4. Run baseline experiments for each dataset.
5. Run improved (KNN-based) experiments using the same protocol.
6. Collect results from ClearML logs and evaluation tables.

The Colab notebook serves as the authoritative reference for execution order.

-------------------------------------------------

11. Output Artifacts
--------------------
Each experiment produces:
- Evaluation metrics for cold, warm, and combined item sets.
- Logged experiment metadata (ClearML offline).
- Model checkpoints.
- Saved embedding manager state.

-------------------------------------------------

12. Reproducibility Notes
-------------------------
- All random seeds are fixed per sub-run.
- Execution order is deterministic when cells are run sequentially.
- Minor numerical differences may occur across GPU types.

-------------------------------------------------

13. Known Limitations
---------------------
- High computational cost.
- Dependence on pretrained item embeddings.
- KNN transfer assumes sufficient warm-item coverage.

-------------------------------------------------

14. Attribution and AI Assistance
---------------------------------
The original Let It Go framework is fully credited to its authors.

AI-assisted tools were used solely to resolve dependency compatibility issues and to adapt
the code to different runtime environments. All modifications were manually reviewed and
validated.

-------------------------------------------------

=================================================
 בעברית (Hebrew Version)
=================================================

1. סקירה כללית
--------------
מאגר זה משחזר ומרחיב את הניסויים שהוצגו במאמר "Let It Go" ובקוד ה־GitHub הרשמי שלו.

המסגרת המקורית מטפלת בבעיה של המלצות לפריטי Cold-Start באמצעות SASRec עם Trainable Deltas.
בעבודה זו אנו בוחנים הרחבה המבוססת על העברת דלתאות מפריטי Warm לפריטי Cold באמצעות KNN,
עם מנגנון Gate אופציונלי למדידת ביטחון.

כל הניסויים בוצעו ותועדו בסביבת Google Colab לצורך שחזור מלא.

-------------------------------------------------

2. תכנון ניסויים
----------------
הניסויים בוצעו על שלושה מערכי נתונים.

לכל Dataset בוצעו שני סוגי הרצות:
- BASELINE: המימוש המקורי ללא KNN.
- IMPROVED (OURS): העברת דלתאות לפריטי Cold באמצעות KNN עם Gate.

לכל Dataset קיימות תתי־הרצות עם Seeds שונים וערכי K שונים.
ה־Colab הוא מקור האמת לסדר ההרצות.

-------------------------------------------------

3. קונפיגורציה והרצה (הסבר)
----------------------------
ההרצות מבוצעות בתוך Colab וכוללות:
- הכנת סביבה.
- טעינת הדאטה.
- הרצת Baseline ולאחר מכן הרצת השיטה המוצעת.

ההבדל המרכזי בין ההרצות הוא הפעלת או ביטול KNN וה־Gate.

-------------------------------------------------

4. פלט ותוצרים
---------------
כל הרצה מפיקה:
- מדדי הערכה (Cold / Warm / Combined).
- Checkpoints של המודל.
- תיעוד מלא של ההרצה.

-------------------------------------------------

5. הערות שחזור
---------------
- Seeds קבועים.
- הרצה סדרתית של התאים.
- ייתכנו הבדלים מספריים קטנים בין סוגי GPU.

-------------------------------------------------
