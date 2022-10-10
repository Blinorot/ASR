import gzip
import os
import shutil
from pathlib import Path

from speechbrain.utils.data_utils import download_file


def main():
    data_dir = Path(__file__).absolute().resolve().parent.parent.parent
    data_dir = data_dir / "data" / "lm" / "nvidia"
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / "files.zip"
    arpa_path = data_dir / "4gram-pruned-0_1_7_9-ru-lm-set-1.0.arpa"

    if not arpa_path.exists():
        print("Loading lm")
        assert arc_path.exists(), "Please Download LM from the website:\
             https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/speechtotext_ru_ru_lm"
        
        shutil.unpack_archive(str(arc_path), str(data_dir))
        os.remove(arc_path)
    

if __name__ == "__main__":
    main()
