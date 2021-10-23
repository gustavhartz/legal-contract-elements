import logging
import warnings
from datetime import datetime

from src import *

logger_spacy = logging.getLogger("spacy")
logger_spacy.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r"\[W108\]", category=UserWarning)
