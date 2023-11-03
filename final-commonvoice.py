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
from datasets import load_dataset
import numpy as np
from torch import from_numpy
log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


def process(args):
    out_root = Path(args.output_root).absolute()
    out_root.mkdir(exist_ok=True)
    # # Extract features
    feature_root = out_root / "fbank80"
    feature_root.mkdir(exist_ok=True)

            
    common_voice = {}
    common_voice['train'] = load_dataset("mozilla-foundation/common_voice_13_0", "gn", split="train")
    common_voice['test'] = load_dataset("mozilla-foundation/common_voice_13_0", "gn", split="test")
    common_voice['validated'] = load_dataset("mozilla-foundation/common_voice_13_0", "gn", split="validation")
    for split in common_voice:
        print(f"Extracting log mel filter bank features... for {split}..")
        for sample in tqdm(common_voice[split]):
            sample_id = f"{sample['path'].split('/')[-1]}"
            extract_fbank_features(
                    from_numpy(np.expand_dims(sample['audio']['array'], 0)), sample['audio']['sampling_rate'], feature_root / f"{sample_id}.npy"
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
    for split in common_voice:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        for sample in tqdm(common_voice[split]):
            manifest["id"].append(sample['path'].split('/')[-1])
            manifest["audio"].append(audio_paths[sample_id])
            manifest["n_frames"].append(audio_lengths[sample_id])
            manifest["tgt_text"].append(sample['sentence'])
            manifest["speaker"].append(sample['client_id'])

        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest), out_root / f"{split}.tsv"
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
