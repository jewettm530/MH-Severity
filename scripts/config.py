"""Configuration for the TCP mental-health symptom-severity pipeline."""

from pathlib import Path
import os
from dotenv import load_dotenv


# Project root:
# .../MH_Severity_Behavioral_and_Scans/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Load the local .env file.
load_dotenv(PROJECT_ROOT / '.env')


# Dataset location comes from each teammate's local .env file. Scripts that
# only inspect generated outputs can still run without a .env; raw-data scripts
# will then look for the generic project-local data/ directory.
DATA_DIR = Path(os.getenv('TCP_DATA_DIR', PROJECT_ROOT / 'Dataset/ds005237')).expanduser().resolve()

PHENOTYPE_DIR = DATA_DIR / 'phenotype'
FMRI_DIR = DATA_DIR / 'fMRI_timeseries_clean_denoised_GSR_parcellated'
MOTION_DIR = DATA_DIR / 'motion_FD'
DEMOS_FILE = PHENOTYPE_DIR / 'demos.csv'


# Outputs are stored inside the project repository.
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

CLEAN_BEHAVIOR_DIR = OUTPUT_DIR / 'behavioral_by_file'
EXPERIMENT_DIR = OUTPUT_DIR / 'experiment_datasets'
CROSS_CATEGORY_DIR = OUTPUT_DIR / 'cross_category_datasets'
RESULTS_DIR = OUTPUT_DIR / 'results'
MODELS_DIR = RESULTS_DIR / 'models'
PREDICTIONS_DIR = RESULTS_DIR / 'predictions'
IMPORTANCE_DIR = RESULTS_DIR / 'feature_importance'
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
N_REPEATS = 3
MAX_MISSING_FRACTION = 0.25
SUBJECT_ID_COLUMN = 'subjectkey'

# The dataset consistently uses 999/9999 for unavailable values. Do not
# globally replace 99 because 99 can be a valid score in some instruments.
MISSING_CODES = [
    999, 9999, -999, -9999,
    '999', '9999', '-999', '-9999',
    '', 'NA', 'N/A',
]

# Ambiguous repeated-nine values are audited rather than automatically erased.
# Some instruments can legitimately contain scores of 99.
AMBIGUOUS_REPEATED_NINES = [99, -99, '99', '-99']


# Imaging feature engineering settings.
# Regional parcel-strength variables are extracted for every parcel, then
# reduced inside each training/CV fold to avoid feature-selection leakage.
EXPECTED_N_PARCELS = 488
GRAPH_DENSITY = 0.10
IMAGING_REGIONAL_SELECT_K = 30
IMAGING_REGIONAL_PCA_COMPONENTS = 15

REST_FILE_PATTERNS = [
    'task-restAP_run-01_bold_Atlas_hp2000_clean_GSR_parcellated.h5',
    'task-restAP_run-02_bold_Atlas_hp2000_clean_GSR_parcellated.h5',
    'task-restPA_run-01_bold_Atlas_hp2000_clean_GSR_parcellated.h5',
    'task-restPA_run-02_bold_Atlas_hp2000_clean_GSR_parcellated.h5',
]

TARGETS = {
    'qids': {
        'category': 'depression',
        'file': 'qids01.csv',
        'column': 'qvtot',
        'label': 'QIDS depression severity',
        'valid_min': 0,
        'valid_max': 27,
    },
    'stai': {
        'category': 'anxiety',
        'file': 'stai01.csv',
        'column': 'staiy_state',
        'label': 'STAI state anxiety severity',
        'valid_min': 20,
        'valid_max': 80,
    },
    'shaps': {
        'category': 'anhedonia',
        'file': 'shaps01.csv',
        'column': 'shaps_total',
        'label': 'SHAPS anhedonia severity',
        'valid_min': 0,
        'valid_max': 56,
    },
    'pss': {
        'category': 'stress',
        'file': 'pss01.csv',
        'column': 'pss_totalscore',
        'label': 'PSS perceived stress',
        'valid_min': 0,
        'valid_max': 40,
    },
}

# Direct symptom-category scales. A scale can appear in more than one category
# when it has clinically meaningful subscales for multiple domains.
CATEGORY_FILES = {
    'depression': [
        'madrs01.csv', 'dass01.csv', 'masq01.csv', 'rrs01.csv',
        'poms01.csv', 'qids01.csv',
    ],
    'anxiety': [
        'anxsi01.csv', 'dass01.csv', 'masq01.csv', 'pdss01.csv',
        'stai01.csv',
    ],
    'anhedonia': [
        'shaps01.csv', 'teps01.csv', 'masq01.csv', 'bisbas01.csv',
    ],
    'stress': [
        'pss01.csv', 'dass01.csv', 'cerq01.csv', 'ctq01.csv',
    ],
}

# Clinically plausible contributors beyond the target's own category.
RELEVANT_CATEGORIES = {
    'qids': ['depression', 'anxiety', 'stress', 'anhedonia'],
    'stai': ['anxiety', 'stress', 'depression', 'anhedonia'],
    'shaps': ['anhedonia', 'depression', 'stress', 'anxiety'],
    'pss': ['stress', 'anxiety', 'depression', 'anhedonia'],
}

# Broad behavioral files used by the "all" scope. Administrative-only files,
# definition files, notes, and target/task files not suitable as psychiatric
# predictors are excluded.
ALL_BEHAVIOR_FILES = [
    'anxsi01.csv', 'bapq01.csv', 'bis01.csv', 'bisbas01.csv', 'cerq01.csv',
    'cgi01.csv', 'cogfq01.csv', 'ctq01.csv', 'dass01.csv', 'dospert01.csv',
    'ecr01.csv', 'fagerstrom01.csv', 'lrift01.csv', 'madrs01.csv',
    'masq01.csv', 'mcas01.csv', 'mspss01.csv', 'nffi01.csv', 'panss01.csv',
    'pdss01.csv', 'poms01.csv', 'pss01.csv', 'pum01.csv', 'qids01.csv',
    'rrs01.csv', 'rsri01.csv', 'shaps01.csv', 'sils01.csv', 'stai01.csv',
    'subuqp201.csv', 'tci01.csv', 'teps01.csv', 'ymrs01.csv',
]

SYMPTOM_FILES = ['madrs01.csv','dass01.csv','masq01.csv','rrs01.csv','poms01.csv',
                 'qids01.csv','anxsi01.csv','pdss01.csv','stai01.csv','shaps01.csv',
                 'teps01.csv', 'pss01.csv','panss01.csv','ymrs01.csv','cgi01.csv']

NON_SYMPTOM_FILES = [f for f in ALL_BEHAVIOR_FILES if f not in SYMPTOM_FILES]

BEHAVIOR_SCOPES = ['all', 'relevant', 'category', 'non-symptom']
INPUT_TYPES = ['imaging_only', 'behavioral_only', 'multimodal']
