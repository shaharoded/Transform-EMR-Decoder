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

# Prediction targets = canonical "Complications" event family from the TAK repo
# (Mediator/core/knowledge-base). The single source of truth is the TAK repo:
# every TAK with family=='event' and category=='Complications' is listed here.
# Outcomes that fail the OUTCOME_RARE_THRESHOLD_PCT prevalence check on the
# post-observation window are demoted to regular LM tokens — they stay in the
# tokenizer vocab (so the backbone still sees and next-token-predicts them),
# they just stop being outcome-head targets / CBM-forbid-protected / sampler-
# upweighted. DEATH_EVENT also appears in TERMINAL_OUTCOMES; listing it here
# is harmless (set-union dedups it inside the tokenizer).
OUTCOMES = [
    "ACIDOSIS_EVENT",
    "ACUTE_RESPIRATORY_DISORDER_EVENT",
    "CARDIO-VASCULAR_DISORDER_EVENT",
    "DEATH_EVENT",
    "DIABETIC_COMA_EVENT",
    "HYPERGLYCEMIA_EVENT",
    "HYPEROSMOLALITY_EVENT",
    "HYPOGLYCEMIA_EVENT",
    "INFECTION_EVENT",
    "KETOACIDOSIS_EVENT",
    "KIDNEY_COMPLICATION_EVENT",
    "OTHER_COMPLICATION_EVENT",
    "SEVERE_HYPERGLYCEMIA_EVENT",
    "SEVERE_HYPOGLYCEMIA_EVENT",
]

ADMISSION_TOKEN = "ADMISSION_EVENT"
DEATH_TOKEN = "DEATH_EVENT"
RELEASE_TOKEN = "RELEASE_EVENT"

TERMINAL_OUTCOMES = [RELEASE_TOKEN, DEATH_TOKEN]

MEAL_TOKENS = ["MEAL_CONTEXT_Breakfast", "MEAL_CONTEXT_Lunch", "MEAL_CONTEXT_Dinner", "MEAL_CONTEXT_Night-Snack"] # Keep ordered! concept_value tokens

# Minimum patient prevalence (%) for an outcome to be included in the outcome head.
# Outcomes below this threshold are dropped — they have too few positive examples to learn from.
OUTCOME_RARE_THRESHOLD_PCT = 1.0

USE_QA_DATA = True  # Step 4 QA toggle on M-128 platform: keeps %_PATTERN% events (vocab grows from non-QA 453) + adds QA ComplianceScore context features (ctx_dim grows from 7). Revert to False after.
# Observation window (hours from admission) the model is seeded with. Same window is
# used to (a) aggregate QA ComplianceScore into context features and (b) define the
# "post-observation" range over which outcome support is measured for rare-outcome
# demotion. At eval time DataProcessor overrides this with max_input_days * 24 so
# QA features match the k-day seed actually given to the model.
OBSERVATION_WINDOW_HOURS = 48

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