import json
import logging
import os
import shutil
from pathlib import Path

import jsonlines
import pandas as pd
import torchaudio
from hw_asr.base.base_dataset import BaseDataset
from hw_asr.utils import ROOT_PATH
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

logger = logging.getLogger(__name__)

URL_LINKS = {
    "farfield": "https://sc.link/1Z3",
    "train_1": "https://sc.link/MvQ",
    "train_2": "https://sc.link/NwL",
    "train_3": "https://sc.link/Oxg",
    "train_4": "https://sc.link/Pyz",
    "train_5": "https://sc.link/Qz7",
    "train_6": "https://sc.link/RAL",
    "train_7": "https://sc.link/VG5",
    "train_8": "https://sc.link/WJW",
    "train_9": "https://sc.link/XKk", 
}

class GolosDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ru_golos"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_dataset(self):
        print(f"Loading GOLOS_farfield_6_7_8_9")
        for elem in ["crowd6", "crowd7", "crowd8", "crowd9", "farfield"]:
            arch_path = self._data_dir / f"train_{elem}.tar"

            if not arch_path.exists():
                download_file(URL_LINKS[f"train_{elem}"], arch_path)
            shutil.unpack_archive(arch_path, self._data_dir)
            if i == 9:
                shutil.move(str(self._data_dir / "train" / "manifest.jsonl"),\
                            str(self._data_dir / "manifest.jsonl"))
            os.remove(str(arch_path))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_dataset()

        wav_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".wav") for f in filenames]):
                wav_dirs.add(dirpath)
        for wav_dir in tqdm(
                list(wav_dirs), desc=f"Preparing golos folders: {part}"
        ):
            wav_dir = Path(wav_dir)
            trans_path = self._data_dir / "manifest.jsonl"
            with jsonlines.open(str(trans_path)) as reader:
                for obj in reader.iter(type=dict):
                    if "farfield" not in str(wav_dir):
                        path_check = f"crowd/{str(wav_dir)[-1]}"
                    else:
                        path_check = "farfield"
                    if  path_check not in obj["audio_filepath"]:
                        continue
                    w_id = obj['id'] + ".wav"
                    w_text = obj['text'].strip()
                    wav_path = wav_dir / w_id
                    t_info = torchaudio.info(str(wav_path))
                    length = t_info.num_frames / t_info.sample_rate
                    index.append(
                        {
                            "path": str(wav_path.absolute().resolve()),
                            "text": w_text.lower(),
                            "audio_len": length,
                        }
                    )
        return index
