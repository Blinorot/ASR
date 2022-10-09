import gzip
import os
import shutil
from pathlib import Path

from speechbrain.utils.data_utils import download_file

URL_LINKS = {
    "lm": "https://www.openslr.org/resources/11/3-gram.arpa.gz", 
}

def main():
    data_dir = Path(__file__).absolute().resolve().parent.parent.parent
    data_dir = data_dir / "data" / "lm" / "librispeech"
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / "3-gram.arpa.gz"
    arpa_path = data_dir / "3-gram.arpa"

    if not arpa_path.exists():
        print("Loading lm")
        if not arc_path.exists():
            download_file(URL_LINKS["lm"], dest=arc_path)
        with gzip.open(str(arc_path), 'rb') as f_in:
            with open(str(arpa_path), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(arc_path)
    

if __name__ == "__main__":
    main()
