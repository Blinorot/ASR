import argparse
import os
import shutil
from pathlib import Path

from speechbrain.utils.data_utils import download_file
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

URL_LINKS = {
    "vocabulary": "https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz", 
}

def main(args):
    data_dir = Path(__file__).absolute().resolve().parent.parent.parent
    data_dir = data_dir / "data" / "datasets" / "librispeech"
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / "librispeech-lm-norm.txt.gz"
    txt_path = data_dir / "librispeech-vocab.txt"

    if not txt_path.exists():
        print("Loading vocabulary")
        download_file(URL_LINKS["vocabulary"], dest=arc_path)
        shutil.unpack_archive(arc_path, data_dir)
        os.remove(arc_path)
    
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["^", " ", "[UNK]", "|", "'"], vocab_size=args.vocabulary)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train([str(txt_path)], trainer)
    save_path = Path(__file__).absolute().resolve().parent / "tokenizer.json"
    tokenizer.save(str(save_path))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Tokenizer Script")
    args.add_argument(
        "-v",
        "--vocabulary",
        default=60,
        type=int,
        help="vocabulary size for tokenizer (default 60)",
    )
    main(args.parse_args())
