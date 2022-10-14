import os
import sys

from datasets import load_metric

from tunip.env import NAUTS_HOME


conda_prefix = os.environ.get('CONDA_PREFIX', None)
packages_path = f"{conda_prefix}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages" if conda_prefix else '.'

TOKEN_CLASSIFICATION_METRIC = load_metric(f"{packages_path}/tweak/metrics/iob_seq_scores.py")
SEQUENCE_CLASSIFICATION_METRIC = load_metric(f"{packages_path}/tweak/metrics/label_only_scores.py")
# TOKEN_CLASSIFICATION_METRIC = load_metric(f"{NAUTS_HOME}/ner/metrics/iob_seq_scores.py")
# SEQUENCE_CLASSIFICATION_METRIC = load_metric(f"{NAUTS_HOME}/ner/metrics/label_only_scores.py")
