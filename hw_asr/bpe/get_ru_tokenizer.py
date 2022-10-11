import argparse
import re
from pathlib import Path

import jsonlines
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def main(args):
    data_dir = Path(__file__).absolute().resolve().parent.parent.parent
    data_dir = data_dir / "data" / "datasets" / "ru_commonvoice"
    data_dir.mkdir(exist_ok=True, parents=True)

    train_dir = data_dir / "train.tsv"
    assert train_dir.exists(), "Please download RU Common Voice first"

    txt_path = data_dir / "train.txt"
    df = pd.read_csv(str(train_dir), sep="\t")
    with open(str(txt_path), "w") as f:
        for _, row in df.iterrows():
            text = row["sentence"]
            text = text.lower()
            text = re.sub(r"[^а-я ]", "", text)
            f.write(text + "\n")

    data_dir = data_dir.parent / "ru_golos"
    train_dir = data_dir / "manifest.jsonl"
    assert train_dir.exists(), "Please download RU Golos first"

    golos_txt_path = data_dir / "train.txt"
    with open(str(golos_txt_path), "w") as f:
        with jsonlines.open(str(train_dir)) as reader:
            for obj in reader.iter(type=dict):
                text = obj["text"]
                text = text.lower()
                text = re.sub(r"[^а-я ]", "", text)
                f.write(text + "\n")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["^", " ", "[UNK]", "|", "'"], vocab_size=args.vocabulary)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train([str(txt_path), str(golos_txt_path)], trainer)
    save_path = Path(__file__).absolute().resolve().parent / "ru_tokenizer.json"
    tokenizer.save(str(save_path))

    print(tokenizer.get_vocab())


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Tokenizer Script")
    args.add_argument(
        "-v",
        "--vocabulary",
        default=80,
        type=int,
        help="vocabulary size for tokenizer (default 80)",
    )
    main(args.parse_args())
