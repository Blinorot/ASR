import gzip
import os
import shutil
from pathlib import Path

from speechbrain.utils.data_utils import download_file

URL_LINKS = {
    "MIT": "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip",
}

def main():
    data_dir = Path(__file__).absolute().resolve().parent.parent
    data_dir = data_dir / "data" / "reverb" / "mit"
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / "Audio.zip"

    if not arc_path.exists():
        print(f"Loading MIT IR Survey")
        download_file(URL_LINKS["MIT"], arc_path)
    shutil.unpack_archive(str(arc_path), str(data_dir))
    for fpath in (data_dir / "Audio").iterdir():
        shutil.move(str(fpath), str(data_dir / fpath.name))
    os.remove(arc_path)
    shutil.rmtree(str(data_dir / "Audio"))
    shutil.rmtree(str(data_dir / "__MACOSX"))
    

if __name__ == "__main__":
    main()
