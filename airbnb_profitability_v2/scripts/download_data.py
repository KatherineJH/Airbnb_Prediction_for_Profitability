"""Download merged_file.csv from the public Google Drive link (needs `pip install gdown`)."""
import sys
from pathlib import Path

FILE_ID = "1rxWScCzr-SH2H55IC8xxijs6lXjkaPMZ"
dest = Path(__file__).resolve().parents[1] / "data" / "raw" / "merged_file.csv"
try:
    import gdown
except ImportError:
    sys.exit("pip install gdown   (or download the file manually into data/raw/)")
dest.parent.mkdir(parents=True, exist_ok=True)
gdown.download(id=FILE_ID, output=str(dest), quiet=False)
sys.path.insert(0, str(dest.parents[2] / "src"))
from airbnb_profit.data import sha256_file  # noqa: E402

print("sha256:", sha256_file(dest), "\n-> put this into config data.expected_sha256 to pin the file.")
