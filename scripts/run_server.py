import os
import sys

sys.path.insert(0, os.path.abspath("src"))
os.environ["PYTHONPATH"] = "src"
os.environ["PRISMX_COLLECTION_NAME"] = "c100k_raw"
os.environ["PRISMX_DB_PATH"] = "data/c100k_raw/text_store_raw.db"
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import uvicorn

if __name__ == "__main__":
    uvicorn.run("prismx.api.app:app", host="127.0.0.1", port=8000, reload=False)
