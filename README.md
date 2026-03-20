Cell 1 — Install Dependencies
Installs all required libraries (XGBoost, scikit-learn, pandas, numpy, matplotlib, seaborn) into the Colab environment. Must run first before anything else.

Cell 2 — Imports
Loads all installed libraries into memory and sets random seeds to ensure reproducible results every time the notebook is run.

Cell 3 — Generate Dataset
Creates a realistic synthetic dataset with 500 training rows and 100 test rows matching the exact assignment schema. Includes intentional noise — missing values, contradictory signals, label noise, and short vague texts like "ok" and "fine" to simulate real-world messy data.

Cell 4 — Feature Engineering
Transforms raw data into numbers the model can learn from. Converts journal text into TF-IDF features, encodes categorical columns ordinally, imputes missing values with medians, and normalizes metadata using StandardScaler.

Cell 5 — Train Emotional State Model
Trains an XGBoost classifier wrapped with sigmoid calibration to predict one of 10 emotional states. Runs 5-fold cross-validation and prints a full classification report with accuracy, precision, and recall per class.

Cell 6 — Confusion Matrix
Draws a heatmap showing which emotional states the model confuses with each other. Rows are true states, columns are predicted states, diagonal is correct predictions.

Cell 7 — Intensity Model
Trains an XGBoost Regressor to predict emotional intensity on a scale of 1 to 5. Uses regression instead of classification because intensity is ordinal — being off by 1 is better than being off by 4. Prints MAE and RMSE and plots true vs predicted distributions.

Cell 8 — Decision Engine
Defines the rule-based logic that decides what action to recommend (box_breathing, deep_work, rest, journaling, etc.) and when to do it (now, within_15_min, later_today, tonight). Uses predicted state, intensity, energy, stress, and time of day. Includes a live demo table.

Cell 9 — Uncertainty Modeling
Checks 5 signals per prediction — low confidence, low margin between top 2 classes, short text, contradictory signals, and missing face hint — to decide if a prediction should be flagged as uncertain. Plots a confidence score histogram across all test predictions.

Cell 10 — Generate predictions.csv
Runs the full pipeline on every test row and saves the final output CSV with all required columns: id, predicted_state, predicted_intensity, confidence, uncertain_flag, what_to_do, when_to_do, and supportive_message.

Cell 11 — Ablation Study
Compares two model versions — text only vs text plus metadata — to measure how much metadata improves accuracy. Shows results in a bar chart. Proves metadata is critical when text is short or vague.

Cell 12 — Error Analysis
Finds all failure cases where the state was predicted wrong or intensity was off by 2 or more. Prints a detailed breakdown of the top 10 failures including the text, true vs predicted labels, confidence scores, and automatic diagnosis of why the model failed.

Cell 13 — Robustness Tests
Tests 3 deliberately tricky edge cases — extremely short text with missing metadata, contradictory signals, and mixed emotional language — to verify the system handles them gracefully with appropriate uncertainty flags.

Cell 14 — Download predictions.csv
Triggers a browser download of the final predictions.csv directly to your computer. This is the main submission deliverable.
