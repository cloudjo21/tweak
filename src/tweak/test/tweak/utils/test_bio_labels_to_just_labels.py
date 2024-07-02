import os

from tunip.corpus_utils_v2 import CorpusRecord, CorpusSeqLabel, CorpusToken


filepath = f"{os.environ['HOME']}/temp/user/ed/mart/corpus/ner/nc_ner.small/20220727_000000_000000/dev_corpus.tjsonl"


def get_just_labels(obj: CorpusRecord):
    has_boundary = True 
    label_entries = []
    label = None
    for obj_label in obj.labels:
        # obj_label.
        if obj_label.label.startswith('B-') and not has_boundary:
            label_entries.append(CorpusSeqLabel(start=start, end=end, label=label))
            start = obj_label.start
            end = obj_label.end
            label = obj_label.label[2:]
            has_boundary = False
        if obj_label.label.startswith('B-') and has_boundary:
            if label:
                label_entries.append(CorpusSeqLabel(start=start, end=end, label=label))
            start = obj_label.start
            end = obj_label.end
            label = obj_label.label[2:]
            has_boundary = False
        elif obj_label.label.startswith('I-'):
            end = obj_label.end
            has_boundary = True
    
    if has_boundary:
        label_entries.append(CorpusSeqLabel(start=start, end=end, label=label))
    print(label_entries)


with open(filepath, encoding="utf-8") as f:
    guid = 0

    for line in f:
        guid = guid + 1

        obj = CorpusRecord.parse_raw(line)
        # text = obj.text
        # surfaces = [t.surface for t in obj.tokens]
        # ner_starts = [t.start for t in obj.labels]
        # ner_ends = [t.end for t in obj.labels]
        # iob_ner_tags = [t.label for t in obj.labels]

        if guid == 2:
            get_just_labels(obj)
            break

# seq_labels = [{"end": 2, "label": "B-ORG", "start": 0}, {"end": 4, "label": "I-ORG", "start": 2}, {"end": 5, "label": "I-ORG", "start": 4}, {"end": 7, "label": "B-ORG", "start": 6}, {"end": 9, "label": "I-ORG", "start": 7}, {"end": 11, "label": "I-ORG", "start": 9}, {"end": 13, "label": "I-ORG", "start": 11}, {"end": 37, "label": "B-TRM", "start": 36}, {"end": 41, "label": "I-TRM", "start": 37}, {"end": 43, "label": "I-TRM", "start": 41}, {"end": 44, "label": "I-TRM", "start": 43}, {"end": 45, "label": "I-TRM", "start": 44}, {"end": 46, "label": "I-TRM", "start": 45}, {"end": 47, "label": "I-TRM", "start": 46}, {"end": 49, "label": "I-TRM", "start": 47}, {"end": 50, "label": "I-TRM", "start": 49}, {"end": 66, "label": "B-TRM", "start": 63}, {"end": 72, "label": "B-TRM", "start": 69}, {"end": 83, "label": "B-TRM", "start": 80}, {"end": 86, "label": "I-TRM", "start": 84}, {"end": 88, "label": "I-TRM", "start": 86}, {"end": 94, "label": "I-TRM", "start": 89}, {"end": 95, "label": "I-TRM", "start": 94}, {"end": 97, "label": "B-TRM", "start": 96}, {"end": 99, "label": "I-TRM", "start": 97}, {"end": 101, "label": "I-TRM", "start": 99}, {"end": 104, "label": "I-TRM", "start": 102}, {"end": 105, "label": "I-TRM", "start": 104}, {"end": 110, "label": "I-TRM", "start": 106}, {"end": 113, "label": "I-TRM", "start": 111}, {"end": 117, "label": "B-TRM", "start": 116}, {"end": 120, "label": "I-TRM", "start": 117}, {"end": 121, "label": "I-TRM", "start": 120}]

