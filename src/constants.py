from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

DATA_PATH = BASE_DIR / "data"

TEST_DATA_PATH = DATA_PATH / "test.csv"

CONFIG_PATH = BASE_DIR / "config"

ENV_PATH = CONFIG_PATH / ".env"