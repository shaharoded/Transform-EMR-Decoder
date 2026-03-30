import os

# Go two levels up from this config file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Data file paths (relative to project root)
TAK_REPO_PATH            = os.path.join(PROJECT_ROOT, 'transform_emr', 'config', 'tak-repo-portable.json')
TRAIN_TEMPORAL_DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'train', 'synthetic_diabetes_temporal_data.csv')
TRAIN_CTX_DATA_FILE      = os.path.join(PROJECT_ROOT, 'data', 'train', 'synthetic_diabetes_context_data.csv')
TEST_TEMPORAL_DATA_FILE  = os.path.join(PROJECT_ROOT, 'data', 'test', 'synthetic_diabetes_temporal_data.csv')
TEST_CTX_DATA_FILE       = os.path.join(PROJECT_ROOT, 'data', 'test', 'synthetic_diabetes_context_data.csv')

# Define the prediction targets, <bot>, <eot> tokens to terminate the inference
OUTCOMES = [
    "DISGLYCEMIA_EVENT",
    "KIDNEY_COMPLICATION_EVENT",
    "ACUTE_RESPIRATORY_DISORDER_EVENT",
    "INFECTION_EVENT",
    "ATHEROSCLEROSIS_EVENT",
    "CARDIO-VASCULAR_DISORDER_EVENT",
    "NERVOUS_SYSTEM_DISORDER_EVENT",
    "NEUROVASCULAR_COMPLICATION_EVENT",
    "SKIN_ULCER_EVENT",
    "OTHER_COMPLICATION_EVENT",
    "RETINOPATHY_EVENT",
    "HYPEROSMOLALITY_EVENT",
    "DIABETIC_COMA_EVENT",
    "KETOACIDOSIS_EVENT",
    "ACIDOSIS_EVENT",
]

ADMISSION_TOKEN = "ADMISSION_EVENT"
DEATH_TOKEN = "DEATH_EVENT"
RELEASE_TOKEN = "RELEASE_EVENT"

TERMINAL_OUTCOMES = [RELEASE_TOKEN, DEATH_TOKEN]

MEAL_TOKENS = ["MEAL_CONTEXT_Breakfast", "MEAL_CONTEXT_Lunch", "MEAL_CONTEXT_Dinner", "MEAL_CONTEXT_Night-Snack"] # Keep ordered! concept_value tokens