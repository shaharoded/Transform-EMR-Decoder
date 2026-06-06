import os

# Go two levels up from this config file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Data file paths (relative to project root)
TAK_REPO_PATH            = os.path.join(PROJECT_ROOT, 'intervene_ar', 'config', 'tak-repo-portable.json')
TRAIN_TEMPORAL_DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'train', 'temporal_data.csv')
TRAIN_CTX_DATA_FILE      = os.path.join(PROJECT_ROOT, 'data', 'train', 'context_data.csv')
TEST_TEMPORAL_DATA_FILE  = os.path.join(PROJECT_ROOT, 'data', 'test', 'temporal_data.csv')
TEST_CTX_DATA_FILE       = os.path.join(PROJECT_ROOT, 'data', 'test', 'context_data.csv')
QA_DATA_FILE             = os.path.join(PROJECT_ROOT, 'data', 'source', 'qa_data.csv')

# Define the prediction targets, <bot>, <eot> tokens to terminate the inference.
# Outcome-snip (16 -> 11 head targets): five outcomes that never achieved
# above-prevalence discrimination under any recipe across the P-/I-sequences
# were removed as outcome-head targets — HYPEROSMOLALITY_EVENT, INFECTION_EVENT,
# ACIDOSIS_EVENT, ATHEROSCLEROSIS_EVENT, ACUTE_RESPIRATORY_DISORDER_EVENT. Their
# tokens REMAIN in the LM vocabulary (the tokenizer is built from training data,
# not from this list), so their occurrences still shape backbone context; they
# simply stop being head-BCE targets, CBM-forbid-protected, and sampler-upweighted.
OUTCOMES = [
    "DISGLYCEMIA_EVENT_Hyperglycemia",
    "DISGLYCEMIA_EVENT_Hypoglycemia",
    "HYPEROSMOLALITY_EVENT",
    "CARDIO-VASCULAR_DISORDER_EVENT",
    "KIDNEY_COMPLICATION_EVENT",
    "KETOACIDOSIS_EVENT",
    "ACIDOSIS_EVENT",
]
# Note: prediction targets are different from thesis dataset (Kinneret) due to different available prediction targets
# KETOACIDOSIS_EVENT and ACIDOSIS_EVENT are available in the data, but low support will auto-reduct them (OUTCOME_RARE_THRESHOLD_PCT)

ADMISSION_TOKEN = "ADMISSION_EVENT"
DEATH_TOKEN = "DEATH_EVENT"
RELEASE_TOKEN = "RELEASE_EVENT"

TERMINAL_OUTCOMES = [RELEASE_TOKEN, DEATH_TOKEN]

MEAL_TOKENS = ["MEAL_CONTEXT_Breakfast", "MEAL_CONTEXT_Lunch", "MEAL_CONTEXT_Dinner", "MEAL_CONTEXT_Night-Snack"] # Keep ordered! concept_value tokens

# Minimum patient prevalence (%) for an outcome to be included in the outcome head.
# Outcomes below this threshold are dropped — they have too few positive examples to learn from.
OUTCOME_RARE_THRESHOLD_PCT = 1.0

USE_QA_DATA = False  # Phase D/E done; best model is M-256 non-QA
# History window (hours from admission) used when aggregating QA ComplianceScore into
# context features. At eval time DataProcessor overrides this with max_input_days * 24
# so QA features match the k-day seed actually given to the model.
QA_HISTORY_HOURS_DEFAULT = 48

# inclusion/exclusion criteria to filter all datasets.
# %_PATTERN% events carry treatment-quality signal — keep them only when QA features
# are enabled, drop them otherwise so the LM does not have to model pattern markers
# the model does not consume.
_temporal_filters = ["WHERE Value NOT LIKE '%Steady%'"]
if not USE_QA_DATA:
    _temporal_filters.append("WHERE ConceptName NOT LIKE '%_PATTERN%'")

INCLUSION_EXCLUSION_CRITERIA = {
    "temporal": _temporal_filters,
    "context": [],
}