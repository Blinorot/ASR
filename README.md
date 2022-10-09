# ASR

This is a repository for ASR homework of HSE DLA Course. Project includes Normalized-BiLSTM and DeepSpeechV2-like models written in PyTorch and trained\evaluated on [LibriSpeech](https://www.openslr.org/12) and Russian Partition of [Common Voice 11.0](https://commonvoice.mozilla.org/en/datasets).

## Getting Started

These instructions will help you to run a project on your machine.

### Prerequisites

Install [pyenv](https://github.com/pyenv/pyenv#installation) following the instructions in the official repository.

### Installation

Clone the repository into your local folder:

```bash
cd path/to/local/folder
git clone https://github.com/Blinorot/ASR.git
```

Install `python 3.9.7.` for `pyenv` and create virtual environment:

```bash
pyenv install 3.9.7
cd path/to/cloned/ASR/project
~/.pyenv/versions/3.9.7/bin/python -m venv asr_env
```

Install required python packages to python environment:

```bash
source asr_env
pip install -r requirements.txt
```

Donwload KenLM model for LibriSpeech Corpus by running `hw_asr/lm/get_lm.py` script.

## Project structure

Repository is structured in the following way.

-   `hw_asr` includes code for all used objects and functions including:

    -   `augmentations`: code for audio augmentations. Wave and Spectrogram augmentations are divided into two folders `wave_augmentations` and `spectrogram_augmentations` respectively. `base.py` includes base class for all augmentations, `sequential.py` and `random_apply.py` implement sequential calling of augmentations and their random apply respectively.

    -   `base`: base class code for datasets, metrics, models, text_encoders and trainers in the correspongding `base_name.py` file.

    -   `batch_sampler`: code for dataloader batch sampler which groups audios by length (TODO).

    -   `bpe`: includes script `generate_tokenizer.py` for generation of HuggingFace Tokenizer for Librispeech Corpus and `tokenizer.json` &mdash; the BPE default script.

    -   `collate_fn`: code for the corresponding function of the dataloader.

    -   `configs`: configs for model training in `model_name/config_name.json` files and for evaluation in `test_configs/config_name.json`.

    -   `datasets`: code for downloading and structuring datasets. Includes code for Custom (Dir) Audio Dataset, LibriSpeech, LJSpeech and RU Common Voice 11.0 in the corresponding `name_dataset.py` file.

    -   `lm`: contains script for downloading KenLM model for LibriSpeech Corpus.

    -   `logger`: code for different loggers (including W&B) and some utils for visualization.

    -   `loss`: includes wrapper over PyTorch CTC Loss.

    -   `metric`: includes CER\WER metric classes for Argmax and Beam Search (with optional LM)predictions and metrics calculation.

    -   `model`: code for simple baseline MLP, Normalized-BiLSTM and DeepSpeechV2-like models (original with normalized gru, residual connections version) architectures.

    -   `tests`: unit-tests that check basic functionality.

    -   `text_encoder`: includes code for char encoder and its CTC version.

    -   `trainer`: code for training models.

    -   `utils`: basic utils including `parse_config.py` for parsing `.json` configs, `object_loading.py` for dataloaders structuring and `util.py` for files\device control and metrics wrapper.

-   `data` folder consists of downloaded datasets folders created by running `name_dataset.py` during evaluation or training.

-   `saved` folder consists of logs and model checkpoints \ their configs in `log` and `models` subdirs respectively.

-   `test_data` includes some data for basic evaluation.

-   `Dockerfile` &mdash; dockerfile to run in Docker (TODO).

-   `requirements.txt` includes all packages required to run the project.

-   `train.py` script for training models.

-   `test.py` script for evaluating models.

## Tests

There are some unit-tests in `hw_asr/tests` folder that check basic functionality.

To run all test write:

```bash
python3 -m unittest discover hw_asr/tests
```

To tun specific test write:

```bash
python3 -m unittest hw_asr/tests/test_name.py
```

Existing tests are the following:

-   `test_config.py`: basic test for `ConfigParser` object.

-   `test_dataloader.py`: checks `collate_fn` function and basic work of dataloaders.

-   `test_metrics.py`: checks CER and WER metrics counter implementation.

-   `test_text_encoder.py`: checks `ctc_decode` function and `beam_search` (TODO)

## BPE

In order to generate HuggingFace Tokenizer run the following command:

```bash
python3 hw_asr/bpe/generate_tokenizer.py -v vocabulary_size
```

Where `vocabulary_size` value defines the number of tokens in Tokenizer. Default value is set to 60. Default Tokenizer config is presented in `hw_asr/bpe/tokenizer.json` config.

## Training

To train model run the following script:

```
python3 train.py -c hw_asr/configs/model_name/config_name.json
```

## Testing

To evaluate model run the following script:

```
python3 test.py -c hw_asr/configs/test_configs/config_name.json \
    -r saved/model/path/to/model_best.pth
```

## Pre-trained models

### English

Pre-trained model for English language is Normalized-BiLSTM trained on LibriSpeech Corpus with `hw_asr/configs/lstm/libri.json` config.

The saved checkpoint can be found in `saved/models/pre-trained/lstm_en/model_best.pth` and corresponding checkpoint in `saved/models/pre-trained/lstm_en/config.json`. **TODO**

### Russian

Pre-trained model for Russian language is Normalized-BiLSTM trained on Russian Partition of Common Voice 11.0 Corpus with `hw_asr/configs/lstm/ru.json` config. **TODO**

## Adding your own models

To add your own model create `new_model.py` file decribing model architecture in `hw_asr/model` directory and and new class in `hw_asr/model/__init__.py`. Afterwards create new config in `hw_asr/configs/new_model/` desribing pre-processing, text encoders, training scheme, etc.

## Authors

-   Petr Grinberg

## License

This project is licensed under the [MIT](LICENSE) License - see the [LICENSE](LICENSE) file for details.

## Credits

Template was taken from [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template).
