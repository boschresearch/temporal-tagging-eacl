import sys

sys.path.append("./")

from bcai_temporal_tagging import TemporalTagger

print(
    "For loading our temporal tagger, you just have to specify the path to our trained MLM model. "
    "Checkout the README for instructions on how to obtain it. "
    'You can also remove the "device" argument and let the model run on CUDA GPUs if available. '
)
tagger = TemporalTagger("models/mlm-xlm-base-multilingual-temporal", device="cpu")


print(
    "# The following text contains two year expression. "
    "We are going to normalize the first one. This example is straighforward."
)
text = "W.D. Currie published the Mercury Mascot from 1904 to 1907."
annotation = "1904"
begin = text.find(annotation)
end = begin + len(annotation)
value = tagger.normalize(text, begin, end, "DATE")
print(f"Text: {text}")
print(f"Annotation: {annotation} ({begin};{end})")
print(f"-> normalized to: {value}")


print(
    '\n- The following text contains an relative expression "last year" '
    "that cannot be resolved without further information."
)
text = "W.D. Currie published the Mercury Mascot last year."
annotation = "last year"
begin = text.find(annotation)
end = begin + len(annotation)
value = tagger.normalize(text, begin, end, "DATE")
print(f"Text: {text}")
print(f"Annotation: {annotation} ({begin};{end})")
print(f"-> normalized to: {value}")


print("\n- Adding the document creation time (DCT) can help here.")
text = "W.D. Currie published the Mercury Mascot last year."
annotation = "last year"
begin = text.find(annotation)
end = begin + len(annotation)
value = tagger.normalize(text, begin, end, "DATE", dct="1905-01-01")
print(f"Text: {text}")
print(f"Annotation: {annotation} ({begin};{end})")
print(f"-> normalized to: {value}")


print(
    "\n- Alternatively, you can add a list of related temporal expressions, "
    "e.g., ones that appeared previously in the document. "
    'For this, you have to change the anchoring mode from "news" to "narratives".'
)
text = "W.D. Currie published the Mercury Mascot last year."
annotation = "last year"
begin = text.find(annotation)
end = begin + len(annotation)
value = tagger.normalize(
    text, begin, end, "DATE", previous_dates=["1905-01-01"], anchoring_mode="narratives"
)
print(f"Text: {text}")
print(f"Annotation: {annotation} ({begin};{end})")
print(f"-> normalized to: {value}")


print(
    "\n\nIf you provide a sequence tagging model for the extraction, "
    "the tagger can also detect temporal expressions in the text. "
)
tagger = TemporalTagger(
    "models/mlm-xlm-base-multilingual-temporal",
    "models/ner-xlm-base-base-multilingual-gold-temporal",
    device="cpu",
)


print("- Now we do not insert the extracted annotations anymore")
text = "W.D. Currie published the Mercury Mascot last year."
annotations = tagger.extract_and_normalize(text, dct="1905-01-01")
print(f"Text: {text}")
print(f"Annotation(s): {annotations}")


print(
    "- Moreover, the previous found and normalized dates are provided "
    "to sequential dates when anchoring mode is narratives."
)
text = (
    "This text was written in 1905. W.D. Currie published the Mercury Mascot last year."
)
annotations = tagger.extract_and_normalize(text, anchoring_mode="narratives")
print(f"Text: {text}")
print(f"Annotation(s): {annotations}")


print("\n- Finally, you can also create a TimeML-formatted file with the tagger")
text = (
    "This text was written in 1905. W.D. Currie published the Mercury Mascot last year."
)
annotations = tagger.extract_and_normalize(text, anchoring_mode="narratives")
timeml_text = tagger.prepare_timeml(text, annotations)
print(tagger.prepare_timeml(text, annotations))
