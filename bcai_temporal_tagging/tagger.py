""" Our neural temporal tagger using sequence taggers for the extraction
and masked language modeling for the normalization of temporal expressions.
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

import os
import pickle
import re
import traceback

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

from bcai_temporal_tagging.fileutils import get_resources
from bcai_temporal_tagging.heideltime import (
    HolidayProcessor,
    Logger,
    correctDurationValue,
    resolve_ambigious_string,
)
from bcai_temporal_tagging.tokenization import TimexTokenizer

try:
    import pycrfsuite as crf
except:
    print("Install pycrfsuite for viterbi decoding")

try:
    import spacy
except:
    print("Install spacy for tokenization and pos-tagging methods")


class TemporalTagger:
    def __init__(
        self,
        mlm_model_name_or_path: str,
        ner_model_name_or_path: str = None,
        decoding_strategy: str = "sim",
        language: str = "en",
        heideltime_path: str = "resources/heideltime/",
        spacy_path: str = "resources/spacy/",
        crf_path: str = "resources/crf/crf__multilingual_all.pycrfsuite",
        device: str = "cuda",
    ):
        self.device = (
            torch.device(device) if torch.cuda.is_available else torch.device("cpu")
        )

        self.mlm_model = AutoModelForMaskedLM.from_pretrained(
            mlm_model_name_or_path
        ).to(self.device)
        self.mlm_tokenizer = TimexTokenizer(
            AutoTokenizer.from_pretrained(mlm_model_name_or_path)
        )

        if ner_model_name_or_path is not None:
            ner_model = AutoModelForTokenClassification.from_pretrained(
                ner_model_name_or_path
            ).to(self.device)
            ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name_or_path)
            self.extraction_pipeline = pipeline(
                "token-classification",
                model=ner_model,
                tokenizer=ner_tokenizer,
                aggregation_strategy="simple",
            )
        else:
            self.extraction_pipeline = None

        self.language = language
        self.heideltime_path = heideltime_path
        self.spacy_path = spacy_path
        self.patters, self.normalization, self.rules = self._load_heideltime_resources(
            "en", False
        )
        self.spacy_model = self._load_spacy_model(language)

        if decoding_strategy == "sim":
            decoding_strategy = "simultaneous"
        elif decoding_strategy == "seq":
            decoding_strategy = "sequential"
        elif decoding_strategy == "crf":
            decoding_strategy = "viterbi"
        assert decoding_strategy in [
            "simultaneous",
            "sequential",
            "viterbi",
        ], f"Unknown decoding_strategy: {decoding_strategy}"
        self.decoding_strategy = decoding_strategy

        if decoding_strategy == "viterbi":
            # print('Load CRF from: ' + crf_path)
            self.crf_tagger = crf.Tagger()
            self.crf_tagger.open(crf_path)
        else:
            self.crf_tagger = None

    def extract_and_normalize(
        self,
        text,
        dct: str = "XXXX-XX-XX",
        anchoring_mode: str = "news",
        previous_dates: list[str] = None,
        return_cir: bool = False,
    ):
        assert (
            self.extraction_pipeline is not None
        ), "Please provide an extraction model during initialization to use this function. "

        if previous_dates is None:
            previous_dates = []

        temporal_expressions = self.extraction_pipeline(text)

        # Merge expressions divided by the extraction model.
        # This was not part of the paper evaluation
        temporal_expressions = self._merge_annotations(temporal_expressions)

        annotations = []
        for exp in tqdm(temporal_expressions):
            begin, end = exp["start"], exp["end"]
            pred_type = exp["entity_group"]
            value = self.normalize(
                text,
                timex_begin=begin,
                timex_end=end,
                timex_type=pred_type,
                dct=dct,
                anchoring_mode=anchoring_mode,
                previous_dates=previous_dates,
                return_cir=return_cir,
            )
            annotations.append(
                {
                    "begin": begin,
                    "end": end,
                    "text": exp["word"],
                    "type": pred_type,
                    "value": value,
                    "score": exp["score"],
                }
            )
            if pred_type in ["DATE", "TIME"]:
                previous_dates.append(value)
        return annotations

    def normalize(
        self,
        text: str,
        timex_begin: int,
        timex_end: int,
        timex_type: str = "DATE",
        dct: str = "XXXX-XX-XX",
        anchoring_mode: str = "news",
        previous_dates: list[str] = None,
        return_cir: bool = False,
        context_window: int = 100,
    ) -> str:
        if previous_dates is None:
            previous_dates = []
        assert anchoring_mode in [
            "news",
            "narratives",
        ], f"Unknown anchoring_mode: {anchoring_mode}"

        ann_info = (timex_begin, timex_end, timex_type, "XXXX")
        pred_cir_value, pred_fields = self._predict_cir_value(text, ann_info)

        if return_cir:
            return pred_cir_value

        next_sent_begin = text.find(".", timex_end)
        if next_sent_begin < timex_end:
            next_sent_begin = timex_end + context_window
        context = text[timex_begin - context_window : next_sent_begin]
        norm_value = self.anchor_cir(
            pred_cir_value,
            text=context,
            dct=dct,
            anchoring_mode=anchoring_mode,
            previous_dates=previous_dates,
        )

        return norm_value

    def anchor_cir(
        self,
        cir: str,
        text: str = "",
        dct: str = "XXXX-XX-XX",
        anchoring_mode: str = "news",
        previous_dates: list[str] = None,
    ) -> str:
        if previous_dates is None:
            previous_dates = []
        assert anchoring_mode in [
            "news",
            "narratives",
        ], f"Unknown anchoring_mode: {anchoring_mode}"

        norm_value = cir

        if "P" == norm_value[0]:
            norm_value = correctDurationValue(norm_value)

        if "funcDateCalc" in norm_value:
            has_function = True
            split = norm_value.find(" ")
            norm_value, function_term = norm_value[:split], norm_value[split + 1 :]
        else:
            has_function = False
            function_term = None

        if "UNDEF" in norm_value:
            if norm_value == "UNDEF-centuryday":
                norm_value = "UNDEF-century"
            elif norm_value == "UNDEF-centuryyear":
                norm_value = "UNDEF-century"
            elif norm_value == "UNDEF-centurycentury":
                norm_value = "UNDEF-century"
            elif norm_value == "UNDEF-centurydayTNI":
                norm_value = "UNDEF-century"
            elif norm_value == "UNDEF-year-04TMO":
                norm_value = "UNDEF-year-04"
            elif norm_value == "UNDEF-REFUNIT-year-PLUS-1":
                norm_value = "UNDEF-REF-year-PLUS-1"
            elif norm_value == "UNDEF-REFUNIT-year-MINUS-2":
                norm_value = "UNDEF-REF-year-MINUS-2"
            elif norm_value == "UNDEF-REFUNIT-year-MINUS-1":
                norm_value = "UNDEF-REF-year-MINUS-1"
            elif norm_value == "UNDEF-year-04TMO":
                norm_value = "UNDEF-year-04"
            elif re.match(r"UNDEF-year-\d\dT\S+", norm_value):
                m = re.match(r"(UNDEF-year-\d\d)T\S+", norm_value)
                norm_value = m.group(1)

            try:
                norm_value = resolve_ambigious_string(
                    norm_value,
                    self.normalization,
                    dct,
                    context=text,
                    linearDates=previous_dates,
                    spacyModel=self.spacy_model,
                    documentType=anchoring_mode,
                )
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                print("Cannot normalize: " + norm_value)

        if has_function:
            try:
                norm_value = HolidayProcessor.evalute_timex(
                    norm_value + " " + function_term
                )
            except:
                norm_value = "XXXX"

        return norm_value

    def prepare_timeml(
        self,
        text: str,
        annotations: list[dict],
        dct: str = "XXXX-XX-XX",
        add_cir: bool = False,
    ):
        timeml_text = '<?xml version="1.0"?>\n'
        # timeml_text += '<!DOCTYPE TimeML SYSTEM "TimeML.dtd">\n'
        timeml_text += '<TimeML xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        timeml_text += 'xsi:noNamespaceSchemaLocation="http://timeml.org/timeMLdocs/TimeML_1.2.1.xsd">\n'
        if dct is not None and dct != "XXXX-XX-XX":
            timeml_text += '<DCT><TIMEX3 functionInDocument="CREATION_TIME" '
            timeml_text += 'temporalFunction="false" tid="t0" type="DATE" '
            timeml_text += 'value="' + dct + '">' + dct + "</TIMEX3></DCT>\n"
        timeml_text += "<TEXT>\n"

        cur_tid = 1
        last_end = 0

        annotations = [ann for ann in annotations if "begin" in ann]
        sorted_annotations = sorted(annotations, key=lambda x: x["begin"])
        for a, ann in enumerate(sorted_annotations):
            # begin = doc_text.find(ann['text'], last_end)
            # end = begin + len(ann['text'])

            end = int(ann["end"])
            begin = int(ann["begin"])

            value = ann["value"]
            if "UNDEF" in value:
                value = "XXXX"
            elif "n.a." in value:
                value = "XXXX"
            ambig_value = ann["cir_value"] if "cir_value" in ann else value

            timeml_text += text[last_end:begin]
            timeml_text += '<TIMEX3 tid="t' + str(cur_tid)
            timeml_text += '" type="' + ann["type"]
            timeml_text += '" value="' + value
            if add_cir:
                timeml_text += '" cir="' + ambig_value
            timeml_text += '">'
            timeml_text += text[begin:end]
            timeml_text += "</TIMEX3>"
            cur_tid += 1

            last_end = end

        if last_end > 0:
            timeml_text += text[last_end:]
        else:
            timeml_text += text

        timeml_text += "\n</TEXT>\n"
        timeml_text += "</TimeML>\n"

        # escape ascii characters (not really needed for evaluation)
        return (
            timeml_text.encode("unicode_escape")
            .decode("utf8")
            .replace("\\n", "\n")
            .replace("&quot;", '"')
        )

    def _predict_cir_value(self, sentence, annotation):
        input_ids, tokens, groups, sent_tagged = self._tokenize_sentence(
            sentence, [annotation]
        )

        masked_input_ids = [
            idx
            if g != self.mlm_tokenizer.group_idx_value
            else self.mlm_tokenizer.tokenizer.mask_token_id
            for idx, g in zip(input_ids, groups)
        ]
        attention_mask = [1 for _ in range(len(masked_input_ids))]

        if self.decoding_strategy == "simultaneous":
            outputs = self.mlm_model(
                input_ids=torch.tensor([masked_input_ids]).to(self.device),
                attention_mask=torch.tensor([attention_mask]).to(self.device),
            )
            masked_index = torch.nonzero(
                torch.tensor(masked_input_ids).to(self.device)
                == self.mlm_tokenizer.tokenizer.mask_token_id,
                as_tuple=False,
            )
            value_tokens = []
            for x in masked_index:
                logits = outputs[0][0][x.item()]
                probs = logits.softmax(dim=0)
                values, predictions = probs.topk(5)
                text_predictions = self.mlm_tokenizer.decode_text_ids(
                    predictions.cpu().numpy()
                )
                value_tokens.append(text_predictions[0])

        elif self.decoding_strategy == "sequential":
            value_ids, value_tokens = [], []
            for mask_idx in range(0, 11):

                if mask_idx > 0:  # Replace the first mask with prediction
                    for i in range(0, len(masked_input_ids)):
                        idx = masked_input_ids[i]
                        if idx == self.mlm_tokenizer.tokenizer.mask_token_id:
                            masked_input_ids[i] = value_ids[-1]
                            break

                outputs = self.mlm_model(
                    input_ids=torch.tensor([masked_input_ids]).to(self.device),
                    attention_mask=torch.tensor([attention_mask]).to(self.device),
                )
                masked_index = torch.nonzero(
                    torch.tensor(masked_input_ids).to(self.device)
                    == self.mlm_tokenizer.tokenizer.mask_token_id,
                    as_tuple=False,
                )

                x = masked_index[0]
                logits = outputs[0][0][x.item()]
                probs = logits.softmax(dim=0)
                values, predictions = probs.topk(5)
                text_predictions = self.mlm_tokenizer.decode_text_ids(
                    predictions.cpu().numpy()
                )
                value_tokens.append(text_predictions[0])
                value_ids.append(predictions[0])

        elif self.decoding_strategy == "viterbi":
            outputs = self.mlm_model(
                input_ids=torch.tensor([masked_input_ids]).to(self.device),
                attention_mask=torch.tensor([attention_mask]).to(self.device),
            )
            masked_index = torch.nonzero(
                torch.tensor(masked_input_ids).to(self.device)
                == self.mlm_tokenizer.tokenizer.mask_token_id,
                as_tuple=False,
            )

            value_logits = []
            for j, x in enumerate(masked_index):
                if j >= 11:
                    break  # skip other annotations in one sentence
                logits = outputs[0][0][x.item()]
                probs = logits.softmax(dim=0)
                value_logits.append(probs[self.mlm_tokenizer.our_vocabulary_offset :])

            x_seq = [
                {
                    f"feat{parameterId}": float(weight)
                    for parameterId, weight in enumerate(list(slot_features))
                }
                for slot_features in value_logits
            ]
            value_tokens = self.crf_tagger.tag(x_seq)

        else:
            raise ValueError(f"Unknown decoding_strategy: {self.decoding_strategy}")

        ambig_value = self.mlm_tokenizer.decode_value_tokens(value_tokens)
        return ambig_value, value_tokens

    def _tokenize_sentence(self, sent, annotations, dct=None, use_dct=False):
        sent_text = sent
        last_end = 0

        sent_tagged = f"<DCT>{dct}</DCT> " if use_dct else ""

        tokens, enc, groups = [], [], []

        for annotation in annotations:
            start, end, timex_type, timex_value = annotation

            # Tokenize previous texts
            tmp_enc, tmp_tokens, _ = self.mlm_tokenizer.text_tokenization(
                sent_text[last_end:start]
            )
            enc.extend(tmp_enc)
            tokens.extend(tmp_tokens)
            groups.extend([self.mlm_tokenizer.group_idx_text for _ in tmp_tokens])

            # Tokenize annotation start
            enc.append(
                self.mlm_tokenizer.fixed_vocab_to_ids[self.mlm_tokenizer.timex_start]
            )
            tokens.append(self.mlm_tokenizer.timex_start)
            groups.append(self.mlm_tokenizer.group_idx_markup)

            # Tokenize type
            tmp_enc, tmp_tokens = self.mlm_tokenizer.value_tokenization(
                timex_type, only_type=True
            )
            enc.extend(tmp_enc)
            tokens.extend(tmp_tokens)
            groups.extend([self.mlm_tokenizer.group_idx_type for _ in tmp_tokens])

            # Tokenize annotation mid
            enc.append(
                self.mlm_tokenizer.fixed_vocab_to_ids[self.mlm_tokenizer.timex_mid]
            )
            tokens.append(self.mlm_tokenizer.timex_mid)
            groups.append(self.mlm_tokenizer.group_idx_markup)

            # Tokenize value
            try:
                tmp_enc, tmp_tokens = self.mlm_tokenizer.value_tokenization(timex_value)
            except KeyError as e:
                print("Cannot process " + timex_value)
                raise e

            enc.extend(tmp_enc)
            tokens.extend(tmp_tokens)
            groups.extend([self.mlm_tokenizer.group_idx_value for _ in tmp_tokens])

            # Tokenize annotation end
            enc.append(
                self.mlm_tokenizer.fixed_vocab_to_ids[self.mlm_tokenizer.timex_end]
            )
            tokens.append(self.mlm_tokenizer.timex_end)
            groups.append(self.mlm_tokenizer.group_idx_markup)

            # Tokenize annotated text
            tmp_enc, tmp_tokens, _ = self.mlm_tokenizer.text_tokenization(
                sent_text[start:end]
            )
            enc.extend(tmp_enc)
            tokens.extend(tmp_tokens)
            groups.extend([self.mlm_tokenizer.group_idx_annotated for _ in tmp_tokens])

            # Tokenize annotation closing
            enc.append(
                self.mlm_tokenizer.fixed_vocab_to_ids[self.mlm_tokenizer.timex_closing]
            )
            tokens.append(self.mlm_tokenizer.timex_closing)
            groups.append(self.mlm_tokenizer.group_idx_markup)

            prev_text = sent_text[last_end:start]
            a1 = f'<TIMEX3 type="{timex_type}" value="{timex_value}">'
            a2 = sent_text[start:end]
            a3 = f"</TIMEX3>"
            sent_tagged += prev_text + a1 + a2 + a3
            last_end = end

        if last_end > 0:
            sent_tagged += sent_text[last_end:]
            tmp_enc, tmp_tokens, _ = self.mlm_tokenizer.text_tokenization(
                sent_text[last_end:]
            )
            enc.extend(tmp_enc)
            tokens.extend(tmp_tokens)
            groups.extend([self.mlm_tokenizer.group_idx_text for _ in tmp_tokens])
        else:
            sent_tagged = sent_text
            enc, tokens, _ = self.mlm_tokenizer.text_tokenization(sent_text)
            groups = [self.mlm_tokenizer.group_idx_text for _ in tokens]

        enc = (
            [self.mlm_tokenizer.tokenizer.bos_token_id]
            + enc
            + [self.mlm_tokenizer.tokenizer.eos_token_id]
        )
        tokens = (
            [self.mlm_tokenizer.tokenizer.bos_token]
            + tokens
            + [self.mlm_tokenizer.tokenizer.eos_token]
        )
        groups = (
            [self.mlm_tokenizer.group_idx_special]
            + groups
            + [self.mlm_tokenizer.group_idx_special]
        )

        return enc, tokens, groups, sent_tagged

    def _load_heideltime_resources(
        self, lang, overwrite_cache=False, cache_dir="cache/"
    ):
        resource_names = {
            "en": "english",
            "de": "german",
            "ca": "auto-catalan",
            "es": "spanish",
            "it": "italian",
            "nl": "dutch",
            "pl": "auto-polish",
            "pt": "portuguese",
            "et": "estonian",
            "eu": "auto-basque",
            "fr": "french",
            "id": "auto-indonesian",
            "ro": "auto-romanian",
            "el": "auto-greek",
            "hr": "croatian",
            "uk": "auto-ukrainian",
            "vi": "vietnamese",
        }

        os.makedirs(cache_dir, exist_ok=True)
        cache_file = cache_dir + resource_names[lang] + ".pkl"
        # print('Load HeidelTime resources')
        if os.path.exists(cache_file) and not overwrite_cache:
            # print(' - Load from file: ' + cache_file)
            with open(cache_file, "rb") as file:
                patterns, normalization, rules = pickle.load(file)
        else:
            res_path = self.heideltime_path + resource_names[lang]
            patterns, normalization, rules = get_resources(res_path)
            # print(' - Create resource file: ' + cache_file)
            with open(cache_file, "wb") as file:
                pickle.dump((patterns, normalization, rules), file)
        return patterns, normalization, rules

    def _load_spacy_model(self, lang):
        spacy_model_mapping = {
            "ca": "ca_core_news_sm-3.2.0",
            "da": "da_core_news_sm-3.2.0",
            "de": "de_core_news_sm-3.2.0",
            "en": "en_core_web_sm-3.2.0",
            "el": "el_core_news_sm-3.2.0",
            "es": "es_core_news_sm-3.2.0",
            "fr": "fr_core_news_sm-3.2.0",
            "it": "it_core_news_sm-3.2.0",
            "ja": "ja_core_news_sm-3.2.0",
            "lt": "lt_core_news_sm-3.2.0",
            "nb": "nb_core_news_sm-3.2.0",
            "nl": "nl_core_news_sm-3.2.0",
            "mk": "mk_core_news_sm-3.2.0",
            "pl": "pl_core_news_sm-3.2.0",
            "pt": "pt_core_news_sm-3.2.0",
            "ro": "ro_core_news_sm-3.2.0",
            "ru": "ru_core_news_sm-3.2.0",
            "zh": "zh_core_web_sm-3.2.0",
            "xx": "xx_ent_wiki_sm-3.2.0",
        }
        spacy_model = (
            spacy_model_mapping[lang]
            if lang in spacy_model_mapping
            else spacy_model_mapping["xx"]
        )
        try:
            return spacy.load(self.spacy_path + spacy_model)
        except:
           try:
               return spacy.load(lang)
           except:
               print(f'Cannot load spacy model: {lang}')
               return None

    @staticmethod
    def _merge_annotations(annotations):
        merged_annotations = []

        def _matching_entity_group(a1, a2):
            if a1["entity_group"] == a1["entity_group"]:
                return True
            return a1["entity_group"] in ["DATE", "TIME"] and a2["entity_group"] in [
                "DATE",
                "TIME",
            ]

        def _matching_boundaries(a1, a2):
            matches = a1["end"] >= (a2["start"] - 1)
            return matches

        def _adjust_group(a1, a2):
            if a1["entity_group"] == "DATE" and a2["entity_group"] in "TIME":
                return "TIME"
            elif a1["entity_group"] == "TIME" and a2["entity_group"] in "DATE":
                return "TIME"
            return a1["entity_group"]

        prev_ann = None
        for ann in annotations:
            if (
                prev_ann
                and _matching_entity_group(prev_ann, ann)
                and _matching_boundaries(prev_ann, ann)
            ):
                prev_ann["word"] += " " + ann["word"]
                prev_ann["entity_group"] = _adjust_group(prev_ann, ann)
                prev_ann["end"] = ann["end"]
                prev_ann["score"] = max(prev_ann["score"], ann["score"])
            else:
                if prev_ann:
                    merged_annotations.append(prev_ann)
                prev_ann = ann

        if prev_ann:
            merged_annotations.append(prev_ann)

        return merged_annotations
