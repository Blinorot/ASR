# ASR

This is a repository for ASR homework of HSE DLA Course. Project includes Normalized-BiLSTM and DeepSpeechV2-like models written in PyTorch and trained\evaluated on [LibriSpeech](https://www.openslr.org/12) and Russian Partition of [Common Voice 11.0](https://commonvoice.mozilla.org/en/datasets). For Russian ASR an extra dataset was used: [Golos](https://github.com/sberdevices/golos).

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

### Data downloading

Donwload KenLM model for LibriSpeech Corpus by running `hw_asr/lm/get_lm.py` script.

Download [MIT IR Survey](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html) by running `scripts/get_IR.py`. It is needed for Reverb augmentation.

Donwload [FSDNoisy18K - train part](https://annotator.freesound.org/fsd/downloads/) by running `scripts/get_noise.py`. It is needed for NoiseFromFiles augmentation.

### Extra data downloading for Russian ASR

Due to necessity for email\registration for dowloading the is no script for downloading RU Common Voice 11.0 and Russian LM. Users should download archives by themselves, the scripts will do the left steps:

-   [RU Common Voice 11.0](https://commonvoice.mozilla.org/en/datasets): set "language" option to "Russian", enter email and download. Put the archive in `data/datasets/ru_commonvoice/` directory.

-   [Riva ASR Russian LM](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/speechtotext_ru_ru_lm): register on Nvidia website and download archive with LMs to `data/lm/nvidia/` directory.

The dataset archive will unpack during training\testing. For LM run `hw_asr/lm/get_ru_lm.py` script first.

## Project structure

Repository is structured in the following way.

-   `hw_asr` includes code for all used objects and functions including:

    -   `augmentations`: code for audio augmentations. Wave and Spectrogram augmentations are divided into two folders `wave_augmentations` and `spectrogram_augmentations` respectively. `base.py` includes base class for all augmentations, `sequential.py` and `random_apply.py` implement sequential calling of augmentations and their random apply respectively.

    -   `base`: base class code for datasets, metrics, models, text_encoders and trainers in the correspongding `base_name.py` file.

    -   `batch_sampler`: code for dataloader batch sampler which groups audios by length (TODO).

    -   `bpe`: includes script `generate_tokenizer.py` for generation of HuggingFace Tokenizer for Librispeech Corpus and `tokenizer.json` &mdash; the BPE default script. There are additional files for Russian data with `ru` in filename.

    -   `collate_fn`: code for the corresponding function of the dataloader.

    -   `configs`: configs for model training in `model_name/config_name.json` files and for evaluation in `test_configs/config_name.json`.

    -   `datasets`: code for downloading and structuring datasets. Includes code for Custom (Dir) Audio Dataset, LibriSpeech, LJSpeech, Golos and RU Common Voice 11.0 in the corresponding `name_dataset.py` file.

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

-   `scripts` folder consists of different `.py` files for downloading Impulse Responses, Noise and Pre-trained models.

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

## Augmentations

Supported Augmentations are the following:

-   Reverb: reverberation from Impulse Responses dataset

-   NoiseFromFiles: noises from noise dataset

-   Noise: colored noise

-   TimeStretch: spectrogram-based stretching

-   Time\Frequency Masking (a.k.a. SpecAug): masking regions on spectrogram

-   PitchShift: shifting the pitch

-   Gain: change volume of the audio

## BPE

In order to generate HuggingFace Tokenizer run the following command:

```bash
python3 hw_asr/bpe/generate_tokenizer.py -v vocabulary_size
```

For Russian version:

```bash
python3 hw_asr/bpe/generate_ru_tokenizer.py -v vocabulary_size
```

Where `vocabulary_size` value defines the number of tokens in Tokenizer. Default value is set to 60 (80 for Russian). Default Tokenizer configs are presented in `hw_asr/bpe/tokenizer.json` and `hw_asr/bpe/ru_tokenizer.json` configs.

## Training

To train model run the following script:

```
python3 train.py -c hw_asr/configs/model_name/config_name.json
```

To resume training from checkpoint (resume optimizers, schedulers etc.)

```
python3 train.py -r path\to\saved\checkpoint.pth
```

To train model with initialization from checkpoint:

```
python3 train.py -c hw_asr/configs/model_name/config_name.json \
    -p path\to\saved\checkpoint.pth
```

## Testing

To evaluate model run the following script:

```
python3 test.py -c hw_asr/configs/test_configs/config_name.json \
    -r saved/model/path/to/model_best.pth
```

Test config should contain:

-   `metrics` key with considered metrics
-   `data` key with inner `test` key with considered dataset for evaluation
-   `text_encoder` key if you want to use LMs, Russian or BPE
-   `test` key for logging the results

See `hw_asr/configs/test_configs/*.json` for examples.

`data` option can also be provided by `-t` option during the call of the file.

The result of the test function is a json file containing pairs (target, {prediction}\{beam prediction}\{beam+lm prediction}) and average of metrics calculated on batches. Pairs in file are done with the beam size providied by `--beamsize` option during the call of the file.

## Pre-trained models

### English

Pre-trained model for English language is Normalized-BiLSTM trained on LibriSpeech Corpus with `hw_asr/configs/lstm/libri_bpe.json` and fine-tuden with `hw_asr/configs/lstm/libri_bpe.json` configs.

The model can be downloaded by running `scripts/get_pretrained.py` script. The saved checkpoint can be found in `saved/models/pre-trained/lstm_en/config_{1, 2}/model_best.pth` and corresponding checkpoint in `saved/models/pre-trained/lstm_en/config_{1, 2}/config.json`.

| Partition  | CER (%) | WER (%) | WER (%) with LM |
| ---------- | ------- | ------- | --------------- |
| test-clean | $15.17$ | $41.87$ | $19.65$         |
| test-other | TBD     | TBD     | TBD             |

### Russian

Pre-trained model for Russian language is Normalized-BiLSTM trained on Russian Partition of Common Voice 11.0 Corpus and Golos with `hw_asr/configs/lstm/ru.json` config. Then successively fine-tuned with `hw_asr/configs/lstm/ru{2-5}.json`.

The model (after each config) can be downloaded by running `scripts/get_pretrained.py` script. The saved checkpoint can be found in `saved/models/pre-trained/lstm_ru/config_{1-5}/model_best.pth` and corresponding checkpoint in `saved/models/pre-trained/lstm_ru/config_{1-5}/config.json`.

| Partition | CER (%) | CER (%) with LM | WER(%) |
| --------- | ------- | --------------- | ------ |
| test      | $20.9$  | $52.2$          | $$     |

## Adding your own models

To add your own model create `new_model.py` file decribing model architecture in `hw_asr/model` directory and and new class in `hw_asr/model/__init__.py`. Afterwards create new config in `hw_asr/configs/new_model/` desribing pre-processing, text encoders, training scheme, etc.

## Authors

-   Petr Grinberg

## License

This project is licensed under the [MIT](LICENSE) License - see the [LICENSE](LICENSE) file for details.

## Credits

Template was taken from [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template).
