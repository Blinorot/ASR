import gzip
import os
import shutil
from pathlib import Path

import gdown
from speechbrain.utils.data_utils import download_file

URL_LINKS = {
    "lstm_ru_1": "https://drive.google.com/uc?id=17aqU_YS1rZD9vLIfN_vd2ZtrV9UG9ZW6&export=download", 
    "lstm_en_1": "https://drive.google.com/uc?id=1FXzEjjT6BXEtmDYmsIHjlbAyEWwaYhj2&export=download"
}

def main():
    data_dir = Path(__file__).absolute().resolve().parent.parent
    data_dir = data_dir / "saved" / "models" / "pre_trained"
    data_dir.mkdir(exist_ok=True, parents=True)

    for name, url in URL_LINKS.items():
        name_split = name.split('_')
        model_name = name_split[0]
        lng = name_split[1]
        config_num = name_split[2]
        model_lng_dir = data_dir / f"{model_name}_{lng}"
        model_lng_dir.mkdir(exist_ok=True, parents=True)
        arc_path = model_lng_dir / f"config_{config_num}.zip"
        final_path = model_lng_dir / f"config_{config_num}"

        if not final_path.exists():
            print(f"Loading {name}")
            if not arc_path.exists():
                gdown.download(URL_LINKS[name], str(arc_path))
            shutil.unpack_archive(str(arc_path), str(model_lng_dir))
            os.remove(arc_path)
    

if __name__ == "__main__":
    main()
