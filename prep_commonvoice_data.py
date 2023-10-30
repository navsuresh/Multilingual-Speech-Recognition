#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
import sys
sys.path.append('/content/fairseq')
import pandas as pd
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
)
from torchaudio.datasets import LIBRISPEECH, COMMONVOICE
from tqdm import tqdm
import re
import string


log = logging.getLogger(__name__)

SPLITS = [
    "train.tsv",
    "test.tsv", 
    "dev.tsv", 
    "invalidated.tsv", 
    "validated.tsv", 
    # "other.tsv"
]

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]

def process(args):
    out_root = Path(args.output_root).absolute()
    out_root.mkdir(exist_ok=True)
    # Extract features
    feature_root = out_root / "fbank80"
    feature_root.mkdir(exist_ok=True)
    for split in SPLITS:
        print(f"Fetching split {split}...")
        dataset = COMMONVOICE(out_root.as_posix(), tsv=split)
        print("Extracting log mel filter bank features...")
        for wav, sample_rate, cv_dict in tqdm(dataset):
            sample_id = f"{cv_dict['path']}"
            extract_fbank_features(
                wav, sample_rate, feature_root / f"{sample_id}.npy"
            )
    # Pack features into ZIP
    zip_path = out_root / "fbank80.zip"
    print("ZIPing features...")
    create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(zip_path)
    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    for split in SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = COMMONVOICE(out_root.as_posix(), tsv=split)
        for _, _, cv_dict in tqdm(dataset):
            sample_id = f"{cv_dict['path']}"
            manifest["id"].append(sample_id)
            manifest["audio"].append(audio_paths[sample_id])
            manifest["n_frames"].append(audio_lengths[sample_id])
            manifest["tgt_text"].append(cv_dict['sentence'])
            manifest["speaker"].append(cv_dict['client_id'])
            # manifest["accent"].append(cv_dict['accent'])
            # manifest["age"].append(cv_dict['age'])
            # manifest["gender"].append(cv_dict['gender'])

        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest), out_root / f"{split}"
        )
        if split.startswith("train"):
            train_text.extend(manifest["tgt_text"])
    # Generate vocab
    vocab_size = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size}"
    with NamedTemporaryFile(mode="w") as f:
        for t in train_text:
            f.write(t + "\n")
        gen_vocab(
            Path(f.name),
            out_root / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size,
        )
    # Generate config YAML
    gen_config_yaml(
        out_root,
        spm_filename=spm_filename_prefix + ".model",
        specaugment_policy="ld"
    )
    # Clean up
    shutil.rmtree(feature_root)

def asr_normalize(text):
    return " ".join(re.sub(r'\([^ ]*\)', '', text).translate(str.maketrans('', '', string.punctuation)).lower().split())
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", "-o", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=10000, type=int)
    args = parser.parse_args()

    process(args)
if __name__ == "__main__":
  main()