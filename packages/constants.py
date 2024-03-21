import os
from dotenv import load_dotenv

load_dotenv()

SCRATCH_DIR = os.path.join(os.getenv("SCRATCH_DIR"), "dp_tokenization")
os.makedirs(f"{SCRATCH_DIR}", exist_ok=True)