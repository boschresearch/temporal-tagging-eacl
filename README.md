<!---

    Copyright (c) 2022 Robert Bosch GmbH and its subsidiaries.

-->

# Multilingual MLM Temporal Tagging Resources

This repository contains the companion material for the following publication:

> Lukas Lange, Jannik Strötgen, Heike Adel, Dietrich Klakow. **Multilingual Normalization of Temporal Expressions with Masked Language Models.** EACL 2023.

Please cite this paper if using the dataset or the code, and direct any questions regarding the dataset
to [Lukas Lange](mailto:lukas.lange@de.bosch.com).
The paper can be found at the [ACL Anthology](https://www.aclweb.org/anthology/TODO) or at
[ArXiv](https://arxiv.org/abs/2205.10399).


## Purpose of this Software

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be maintained nor monitored in any way.

## Setup


### Install dependencies

We use a conda environment with python 3.8 for our experiments.
You might recreate our environment following these steps:
```shell
conda create --name temporal-tagging python=3.8
conda activate temporal-tagging
```

Install torch following the instructions from
https://pytorch.org/get-started/locally/
matching your operating system and your GPU/CPU.

For example **(Note: this installs a cpu-only version most likely)**:
```shell
pip install torch torchvision torchaudio
```
We used torch 1.8.2 in our experiments. Later 1.x versions should be working as well.

Now install the other dependencies.
Note that the provided versions are tested with our code.
Other versions might work as well.
```shell
pip install transformers==4.21.3 tokenizers==0.12.1 pyarrow==6.0.1 datasets==2.4.0 seqeval==1.2.2 spacy==3.2.0 python_crfsuite==0.9.7
```

We also provide an easy-to-use interface which requires gradio
```shell
pip install gradio
```

Alternative, you can reproduce our cpu-only environment by installing dependencies from the requirements file.
```shell
pip install -r requirements.txt
```

### Download our models and data

Download the models from our release page https://github.com/boschresearch/temporal-tagging-eacl/releases/tag/Resources and extract them.
Create a directory named `data/data_weak_supervision` and move the extracted `global_voices` and/or `wikipedia` directories there. 
The extracted model directories should be moved to `models`.

The resulting directories should have the following structure: 

```
tree .
├ ...
├───data
│   └───data_weak_supervision
│       ├───global_voices
│       └───wikipedia
├───models
│   ├───mlm-xlm-base-multilingual-temporal
│   ├───ner-xlm-base-base-multilingual-gold-temporal
│   └───ner-xlm-base-base-multilingual-weak-temporal
├ ...
```

After this, the models are available in the [models](models) and [data](data) directories, respectively. 

We provide the following pre-trained models for the *extraction*:
* [XLM-base model trained on our weakly supervised extractions](models/ner-xlm-base-base-multilingual-weak-temporal)
* [XLM-base model trained on gold-standard extractions](models/ner-xlm-base-base-multilingual-gold-temporal)

and the following model for the *extraction*:
* [XLM-base model trained for our masked language modeling approach](models/mlm-xlm-base-multilingual-temporal)




## Temporal tagging with our models



### GUI
Within your activated conda environment run the following to start the GUI.
It will provide information on the web link once the model is loaded.

```shell
python gui.py
```

With the current config, you should be able to access the demo server under http://127.0.0.1:7242.

### Load models in python

You can load the temporal tagger by given paths to our MLM model and our NER model.
```python
from bcai_temporal_tagging import TemporalTagger

tagger = TemporalTagger(
    "models/mlm-xlm-base-multilingual-temporal",
    "models/ner-xlm-base-base-multilingual-gold-temporal",
)
```
The second model (NER for extraction) is optional if you provide the annotation yourself.


The model has three main functions for end users. First for normalizing a single annotation in a text:
`normalize(text: str, timex_begin: int, timex_end: int, timex_type: str = "DATE", dct: str = "XXXX-XX-XX",
anchoring_mode: str = "news", previous_dates: list[str] = None, return_cir: bool = False) -> str)`
You are requried to provide the overall `text` and information on the annotation as standoff values
like the `timex_begin` and `timex_end` as character offsets from the text, and the `timex_type` (optional).
It has more optional arguments to control the normalization behaviour including `dct`, `anchoring_model` and `previous_dates`.
For example:
```python
text = "W.D. Currie published the Mercury Mascot last year."
annotation = "last year"
begin = text.find(annotation)
end = begin + len(annotation)
value = tagger.normalize(text, begin, end, "DATE", dct='1905-01-01')
```
with output: `value=1904`

Second, you can extract and normalize in a single step using `extract_and_normalize(self, text, dct: str = "XXXX-XX-XX", anchoring_mode: str = "news", previous_dates: list[str] = None, return_cir: bool = False)`.
This function shares the optional arguments from before,
but does not require any timex annotation information as these are extracted with the sequence labeling model.
For example:
```python
text = "This text was written in 1905. W.D. Currie published the Mercury Mascot last year."
annotations = tagger.extract_and_normalize(text, anchoring_mode="narratives")
```
with output: `annotations=[{'begin': 25, 'end': 29, 'text': '1905', 'type': 'DATE', 'value': '1905', 'score': 0.99898285}, {'begin': 72, 'end': 81, 'text': 'last year', 'type': 'DATE', 'value': '1904', 'score': 0.9979806}]`

Finally, you can also generate TimeML output with our tagger, that is compatible with the TempEval2 evaluation script.
For example:
```python
text = "This text was written in 1905. W.D. Currie published the Mercury Mascot last year."
annotations = tagger.extract_and_normalize(text, anchoring_mode="narratives")
timeml_text = tagger.prepare_timeml(text, annotations)
```
with output `timeml_text=`
```xml
<?xml version="1.0"?>
<TimeML xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://timeml.org/timeMLdocs/TimeML_1.2.1.xsd">
<TEXT>
This text was written in <TIMEX3 tid="t1" type="DATE" value="1905">1905</TIMEX3>. W.D. Currie published the Mercury Mascot <TIMEX3 tid="t2" type="DATE" value="1904">last year</TIMEX3>.
</TEXT>
</TimeML>
```


Checkout [examples.py](examples.py) for more detailed information on how to use our temporal tagger.






## Train your own models


### Data preparation


```shell
python scripts/prepare_data.py \
    --model_name_or_path xlm-roberta-base \
    --language multilingual \
    --inp_path data/data_weak_supervision \
    --out_path_ner data/data_ner \
    --out_path_mlm data/data_mlm_xlm \
```

This will create the NER files for extraction in [data/data_ner](data/data_ner)
and the MLM files for normalization in [data/data_mlm](data/data_mlm).

#### Example input documents:
```json
{
  "id": "44115_McCulloch_County,_Texas_s28",
  "text": "W.D. Currie published the Mercury Mascot from 1904 to 1907.",
  "tokens": [{"text": "\n", "start": 0, "end": 1}, {"text": "W.D.", "start": 1, "end": 5}, {"text": "Currie", "start": 6, "end": 12}, {"text": "published", "start": 13, "end": 22}, {"text": "the", "start": 23, "end": 26}, {"text": "Mercury", "start": 27, "end": 34}, {"text": "Mascot", "start": 35, "end": 41}, {"text": "from", "start": 42, "end": 46}, {"text": "1904", "start": 47, "end": 51}, {"text": "to", "start": 52, "end": 54}, {"text": "1907", "start": 55, "end": 59}, {"text": ".", "start":59, "end": 60}], "timex3": [{"start": 47, "rulename": "date_r12a-explicit", "end": 51, "text": "1904", "type": "DATE", "value": "1904", "tid": "t16", "cir_value": "1904"}],
  "meta_data": {"id": "44115_McCulloch_County,_Texas", "dct": "2021-11-09"}
}
```


#### Example NER:
```json
{
  "id": "en-1923",
  "tokens": ["W.D.", "Currie", "published", "the", "Mercury", "Mascot", "from", "1904", "to", "1907", "."],
  "timex_tags": ["O", "O", "O", "O", "O", "O", "O", "B-DATE", "O", "O", "O"]
}
```
Transform the gold standard corpora into the same NER format to perform the evaluation with our functions.


#### Example MLM:
```json
{
  "id": "en-1923",
  "input_ids": [0, 601, 5, 397, 5, 17065, 5056, 91376, 70, 123106, 53, 3010, 47924, 1295, 250007, 250068, 250008, 250002, 250130, 250115, 250002, 250002, 250002, 250002, 250002, 250002, 250002, 250002, 250009, 96929, 250010, 47, 91234, 5, 2],
  "groups": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 5, 2, 1, 1, 1, 0]
}
```
We can see, that this input is not human-readable, as the tokenizer ids are provided.
So you have to create this file again if you switch the tokenizer.
Remember that you only need these files for training.
The `groups` attribute lists the corresponding part of the sentence. In our implementation:
* 0: Tokenizer markup
* 1: Normal text (e.g., "W.D.", "Currie", "published", ...)
* 2: TimeML markup (e.g., "<TIMEX3", "type=", "</TIMEX3>", ..)
* 3: TimeML type (e.g., "DATE")
* 4: TimeML value or our CIR value (e.g., "1904", "UNDEF", "last", "[PAD]", ...)
* 5: Text of the temporal expression (e.g., "1904", "1907", ...)



### Train sequence tagging model for extraction

```shell
python scripts/train_extraction_model.py \
    --model_name_or_path xlm-roberta-base \
    --output_dir models/ner-xlm-base-base-multilingual-weak-temporal-new/ \
    --train_file data/data_ner/multilingual_ner_train.json \
    --validation_file data/data_ner/multilingual_ner_dev.json \
    --do_eval \
    --do_train \
    --warmup_steps 100 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --text_column_name tokens \
    --label_column_name timex_tags \
    --pad_to_max_length \
    --max_seq_length 512
```


###  Train language model for normalization

```shell
python scripts/train_normalization_model.py \
    --model_name_or_path xlm-roberta-base \
    --train_file data/data_mlm_xlm/multilingual_mlm_train.json \
    --validation_file data/data_mlm_xlm/multilingual_mlm_dev.json \
    --do_train \
    --do_eval \
    --output_dir models/mlm-xlm-base-multilingual-temporal-new/ \
    --fp16 \
    --warmup_steps 2500 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --save_strategy epoch \
    --overwrite_cache \
    --overwrite_output_dir \
    --pad_to_max_length \
    --max_seq_length 512
```

Note that this script currently only supports roberta-based models.
If you want to support different models, change the word embedding access from
`model.roberta.embeddings.word_embeddings` to `model.<your_model>.embeddings.word_embeddings`.
You can also think of an abstracted way to access the word embeddings.

#### Multi-GPU training
You can utilize multiple GPUs by replacing the first line with the following:
```shell
python -m torch.distributed.launch --nproc_per_node <num_gpus> scripts/<scriptname>.py \
```
This works with both training scripts.


## License

The code in this repository is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE.txt) file for details.
For a list of other open source components included in this project, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).

The weakly-supervised data created for the paper located in the
folder [data/data_weak_supervision](data/data_weak_supervision) is
licensed under a [Creative Commons Attribution-ShareAlike 4.0 International
License](http://creativecommons.org/licenses/by-sa/4.0/) (CC-BY-SA-4.0).
See the [DATA_LICENSE](DATA_LICENSE.txt) and the [data/README](data/README.md) files for details.
