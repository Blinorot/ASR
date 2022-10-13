import gzip
import os
import shutil
from pathlib import Path

from speechbrain.utils.data_utils import download_file

URL_LINKS = {
    "FSD": "https://zenodo.org/record/2529934/files/FSDnoisy18k.audio_train.zip?download=1",
}

def main():
    data_dir = Path(__file__).absolute().resolve().parent.parent
    data_dir = data_dir / "data" / "noise" / "fsd"
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / "FSDnoisy18k.audio_train.zip"

    if not arc_path.exists():
        print(f"Loading FSD Noise Dataset")
        download_file(URL_LINKS["FSD"], arc_path)
    shutil.unpack_archive(str(arc_path), str(data_dir))
    for fpath in (data_dir / "FSDnoisy18k.audio_train").iterdir():
        shutil.move(str(fpath), str(data_dir / fpath.name))
    os.remove(arc_path)
    shutil.rmtree(str(data_dir / "FSDnoisy18k.audio_train"))


if __name__ == "__main__":
    main()
