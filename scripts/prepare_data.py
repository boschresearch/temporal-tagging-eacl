""" Data preparation methods for training new models.
Related to the multilingual temporal tagging (EACL 2023).
Copyright (c) 2022 Robert Bosch GmbH
@author: Lukas Lange

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer

from bcai_temporal_tagging import TimexTokenizer

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_name_or_path",
    default="xlm-roberta-base",
    type=str,
    help="Path or name of a huggingface model that is used for MLM tokenization.",
)
parser.add_argument(
    "--language",
    default="multilingual",
    type=str,
    help="Two-letter language code for the language that should be processed (e.g., 'en' or 'de') or 'multilingual'. "
    "Using multilingual will convert all available languages. "
    "After that, its not necessary to call this script for individual languages.",
)
parser.add_argument(
    "--inp_path",
    default="data/data_weak_supervision",
    type=str,
    help="Input directory that should be processed.",
)
parser.add_argument(
    "--out_path_ner",
    default="data/data_ner",
    type=str,
    help="Output directory where the extraction files in the json NER format are stored. "
    "Note that these are not tokenized and can be used by different models.",
)
parser.add_argument(
    "--out_path_mlm",
    default="data/data_mlm",
    type=str,
    help="Output directory where the normalization files in our tokenized json MLM format are stored. "
    "Note that these are tokenized and therefore specific to the input tokenizer.",
)
parser.add_argument(
    "--seed",
    default=0,
    type=int,
    help="Random seed for making the train/dev split reproducable.",
)
parser.add_argument(
    "--data_size",
    default=1.0,
    type=float,
    help="Control the amount of data considered compared to the paper. "
    "0 < data_size < 1 : decrease data (suited for testing). "
    "data_size > 1 : increase data (limited by the amount of data available).",
)
parser.add_argument(
    "--cache_dir",
    default="cached",
    type=str,
    help="Cache dir for storing huggingface files.",
)


def get_bio_tokens(sent: dict[str, any]) -> tuple[list[str], list[str]]:
    tokens, timex_labels = [], []
    for token in sent["tokens"]:
        text, begin, end = token["text"], token["start"], token["end"]
        if not token["text"].strip():
            continue
        tokens.append(text)
        label = "O"
        for ann in sent["timex3"]:
            if begin == ann["start"]:
                label = "B-" + ann["type"]
                break
            elif begin >= ann["start"] and end <= ann["end"]:
                label = "I-" + ann["type"]
                break
        timex_labels.append(label)
    return tokens, timex_labels


def get_sentences_from_file(
    filename: Path | str, document_limit: int = -1, pct_previous_sent: float = 0.66
) -> list[any]:
    if document_limit < 0:
        document_limit = float("inf")
    if not os.path.exists(filename):
        print(f"No file found for: {filename}")
        return []

    documents = []
    sentences_in_document = []
    keep_sentences = set()
    last_doc_id = None
    with open(filename, "r", encoding="utf-8") as fin:
        print(f"Read from: {filename}")
        for line in fin:
            if not line.strip():
                continue

            sent = json.loads(line)

            # Check if a new document starts
            doc_id = sent["meta_data"]["id"]
            if doc_id != last_doc_id:
                # Add sentences which should be kept
                sentences_to_keep = []
                for idx in sorted(keep_sentences):
                    sentences_to_keep.append(sentences_in_document[idx])
                if len(sentences_to_keep) > 0:
                    documents.append(sentences_to_keep)

                keep_sentences = set()
                sentences_in_document = []
                if len(documents) >= document_limit:
                    break

            last_doc_id = doc_id
            sentences_in_document.append(sent)

            if len(sent["tokens"]) == 0:
                continue
            if len(sent["timex3"]) > 0:
                sent_annotations = []
                # Get annotations offsets
                a_ranges = {}
                for ann in sent["timex3"]:
                    for idx in range(ann["start"], ann["end"]):
                        a_ranges[idx] = ann

                # Attach annotations to sentences
                idx = 0
                while idx < sent["tokens"][-1]["end"]:
                    if idx in a_ranges:
                        ann = a_ranges[idx]
                        sent_annotations.append(ann)
                        idx = ann["end"]
                    else:
                        idx += 1
                sent["timex3"] = sent_annotations

                if len(sent_annotations) > 0:
                    keep_sentences.add(len(sentences_in_document) - 1)
                    if (
                        len(sentences_in_document) > 1
                    ):  # also keep the previous sentence
                        r = random.randint(0, 100)
                        if r > pct_previous_sent:
                            keep_sentences.add(len(sentences_in_document) - 2)

    # Add sentences which should be kept
    sentences_to_keep = []
    for idx in sorted(keep_sentences):
        sentences_to_keep.append(sentences_in_document[idx])
    if len(sentences_to_keep) > 0:
        documents.append(sentences_to_keep)

    return documents


def get_documents_for_language(
    lang: str, path_news: Path, path_wiki: Path, limit: str = 5000
):
    documents_news = get_sentences_from_file(path_news / (lang + ".json"), limit)
    documents_wiki = get_sentences_from_file(path_wiki / (lang + ".json"), limit)

    documents_output = []
    num_news, num_wiki = 0, 0
    documents_left = len(documents_news) > num_news or len(documents_wiki) > num_wiki
    while documents_left:
        if num_news < len(documents_news):
            documents_output.append(documents_news[num_news])
            num_news += 1
        if num_wiki < len(documents_wiki):
            documents_output.append(documents_wiki[num_wiki])
            num_wiki += 1
        documents_left = (
            len(documents_news) > num_news or len(documents_wiki) > num_wiki
        )
        if len(documents_output) >= limit:
            break

    print(
        f"Returning {len(documents_output)} ({num_news} / {num_wiki}) documents for {lang}"
    )
    return documents_output


def split_document_list(
    documents: list[any], dev_pct: float = 0.05, test_pct: float = 0.0, seed: int = 0
):
    random.seed(seed)
    train, dev, test = [], [], []
    for doc in documents:
        r = random.random()
        if r <= test_pct:
            test.append(doc)
        elif r <= dev_pct + test_pct:
            dev.append(doc)
        else:
            train.append(doc)
    return train, dev, test


def store_list(fname, sentences: List[any]):
    with open(fname, "w", encoding="utf-8") as fout:
        for sent in sentences:
            fout.write(json.dumps(sent, ensure_ascii=False) + "\n")


def process_documents(documents, out_file_mlm: Path, out_file_ner: Path):
    idx = -1
    os.makedirs(out_file_mlm.parent, exist_ok=True)
    os.makedirs(out_file_ner.parent, exist_ok=True)
    with open(out_file_mlm, "w", encoding="utf-8") as fout_mlm:
        with open(out_file_ner, "w", encoding="utf-8") as fout_ner:
            for doc in tqdm(documents):
                for sent in doc:
                    idx += 1

                    try:
                        tokens, timex_labels = get_bio_tokens(sent)

                        fout_ner.write(
                            json.dumps(
                                {
                                    "id": str(idx),
                                    "tokens": tokens,
                                    "timex_tags": timex_labels,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    except Exception as e:
                        print(e)
                        print(traceback.format_exc())
                        print("(1 - NER) Cannot process this sentence!")
                        print(sent)
                        continue

                    try:
                        (
                            enc,
                            tokens,
                            groups,
                            sent_tagged,
                        ) = our_tokenizer.tokenize_sentence(
                            sent, sent, use_cir_value=True, use_dct=False
                        )

                        fout_mlm.write(
                            json.dumps(
                                {"id": str(idx), "input_ids": enc, "groups": groups},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    except Exception as e:
                        print(e)
                        print(traceback.format_exc())
                        print("(2 - MLM) Cannot process this sentence!")
                        print(sent)
                        continue


args = parser.parse_args()
args.inp_path = Path(args.inp_path)
args.out_path_ner = Path(args.out_path_ner)
args.out_path_mlm = Path(args.out_path_mlm)
assert (
    0.01 <= args.data_size <= 3.0
), "Data size can only be set between 1% and 300% compared to our orignal setup"

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path, cache_dir=args.cache_dir
)
our_tokenizer = TimexTokenizer(tokenizer)

is_multilingual = args.language.lower().startswith("multi")
if is_multilingual:
    languages = [
        f.replace(".json", "") for f in os.listdir(args.inp_path / "global_voices")
    ] + [f.replace(".json", "") for f in os.listdir(args.inp_path / "wikipedia")]
else:
    languages = [args.language.lower()]

multilingual_train_ner = []
multilingual_dev_ner = []
multilingual_train_mlm = []
multilingual_dev_mlm = []

for lang in sorted(set(languages)):
    # We use different sizes depending on the quality of HeidelTime rules.
    if lang in ["de", "en"]:
        size = 25_000 * args.data_size

    else:
        continue

    if True:
        pass
    elif lang in [
        "ar",
        "id",
        "vi",
        "et",
        "zh",
        "eu",
        "ca",
        "es",
        "fr",
        "it",
        "pt",
        "ro",
        "nl",
        "el",
        "hr",
        "pl",
        "uk",
        "hi",
    ]:
        size = 15_000 * args.data_size
    else:
        size = 5_000 * args.data_size

    docs = get_documents_for_language(
        lang,
        path_news=args.inp_path / "global_voices",
        path_wiki=args.inp_path / "wikipedia",
        limit=size,
    )

    if len(docs) > 0:
        process_documents(
            docs,
            args.out_path_mlm / f"mono-{lang}_mlm.json",
            args.out_path_ner / f"mono-{lang}_ner.json",
        )

    for task in ["ner", "mlm"]:
        out_path = args.out_path_mlm if task == "mlm" else args.out_path_ner

        idx = 0
        sentences = []
        with open(out_path / f"mono-{lang}_{task}.json", "r", encoding="utf-8") as fin:
            for line in tqdm(fin):
                if line.strip():
                    sentence = json.loads(line)
                    sentence["id"] = f"{lang}-{idx}"
                    sentences.append(sentence)
                    idx += 1

        train, dev, _ = split_document_list(sentences, seed=args.seed)

        if is_multilingual:
            if task == "ner":
                multilingual_train_ner.extend(train)
                multilingual_dev_ner.extend(dev)
            else:
                multilingual_train_mlm.extend(train)
                multilingual_dev_mlm.extend(dev)

        if len(train) > 0:
            print(f"Store train file for {lang} ({len(train)}, {task}))")
            store_list(out_path / f"mono-{lang}_{task}_train.json", train)
        else:
            print(f"No train file for: {lang} ({task})")

        if len(dev) > 0:
            print(f"Store dev file for {lang} ({len(dev)}, {task})")
            store_list(out_path / f"mono-{lang}_{task}_dev.json", train)
        else:
            print(f"No dev file for: {lang} ({task})")
    print()

if is_multilingual:
    print("Create multilingual files")
    store_list(
        args.out_path_ner / "multilingual_ner_train.json", multilingual_train_ner
    )
    store_list(args.out_path_ner / "multilingual_ner_dev.json", multilingual_dev_ner)
    store_list(
        args.out_path_mlm / "multilingual_mlm_train.json", multilingual_train_mlm
    )
    store_list(args.out_path_mlm / "multilingual_mlm_dev.json", multilingual_dev_mlm)
