from datetime import datetime
from src import *
import logging
logger_spacy = logging.getLogger("spacy")
logger_spacy.setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", message=r"\[W108\]", category=UserWarning)