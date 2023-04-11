import gradio as gr
import sys

sys.path.append("./")

from datetime import date

from bcai_temporal_tagging import TemporalTagger

tagger: TemporalTagger = None


def init_model():
    global tagger
    tagger = TemporalTagger(
        "models/mlm-xlm-base-multilingual-temporal",
        "models/ner-xlm-base-base-multilingual-gold-temporal",
        decoding_strategy="sim",
        device="cpu",
    )


def temporal_tagging(text, dct, anchoring_mode, decoding_strategy, language):
    if dct is None or not dct.strip():
        dct = "XXXX-XX-XX"
    if language != tagger.language:
        tagger.spacy_model = tagger._load_spacy_model(language)
    tagger.decoding_strategy = decoding_strategy.split(" ")[0]

    annotations = tagger.extract_and_normalize(
        text, dct=dct, anchoring_mode=anchoring_mode
    )
    timeml_text = tagger.prepare_timeml(text, annotations)

    entities = []
    for ann in annotations:
        entities.append(
            {
                "entity": f'{ann["type"]} / {ann["value"]}',
                "start": ann["begin"],
                "end": ann["end"],
            }
        )

    output_anns = {"text": text, "entities": entities}

    return output_anns, timeml_text


def launch_demo():
    demo = gr.Interface(
        temporal_tagging,
        [
            gr.Textbox(
                label="Text", lines=5, placeholder="Input your text here", value=""
            ),
            gr.Textbox(
                label="Document Creation Time (DCT)",
                lines=1,
                placeholder="yyyy-mm-dd",
                value=str(date.today()),
            ),
            gr.Dropdown(
                label="Anchoring mode",
                choices=["news", "narratives"],
                value="news",
                multiselect=False,
            ),
            gr.Dropdown(
                label="Decoding strategy",
                choices=[
                    "simultaneous (fast)",
                    "sequential (best)",
                    "viterbi (w/ crf)",
                ],
                value="simultaneous (fast)",
                multiselect=False,
            ),
            gr.Dropdown(
                label="Language (used for spacy to detect verb tense)",
                choices=[
                    "ca",
                    "da",
                    "de",
                    "en",
                    "el",
                    "es",
                    "fr",
                    "it",
                    "ja",
                    "lt",
                    "nb",
                    "nl",
                    "mk",
                    "pl",
                    "pt",
                    "ro",
                    "ru",
                    "zh",
                    "xx",
                ],
                value="en",
                multiselect=False,
            ),
        ],
        [
            gr.HighlightedText(
                label="Extracted and Normalized Temporal Expressions",
                combine_adjacent=False,
            ).style(
                color_map={
                    "DATE": "red",
                    "TIME": "blue",
                    "DURATION": "orange",
                    "SET": "red",
                }
            ),
            gr.Textbox(
                label="TimeML",
                lines=5,
            ),
        ],
    )
    demo.launch(server_port=7242, share=True)


if __name__ == "__main__":
    init_model()
    launch_demo()
