import argparse
import json
import nltk
import os
import spacy
import spacy_transformers

spacy.prefer_gpu()

nlp = spacy.load("en_core_web_trf")


def extract_nphrases(stories):
    extractions = {}

    for sid in list(stories.keys()):
        nps_in_story = []
        sents = nltk.sent_tokenize(stories[sid])
        for idx, sent in enumerate(sents):
            # odd cases (obtained from RoViST-VG code)
            sent = sent.replace("[male]", "male")
            sent = sent.replace("[female]", "female")
            sent = sent.replace("[location]", "location")
            sent = sent.replace("[organization]", "organization")

            doc = nlp(sent)
            nps_in_image = []
            for chunk in doc.noun_chunks:
                nps_in_image.append(chunk.text.lower())

            nps_in_image = dict.fromkeys(nps_in_image)
            nps_in_story += list(nps_in_image)

        if idx % 100 == 0:
            print(f'NPs extracted for {idx + 1} samples')

        extractions[sid] = nps_in_story

    return extractions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='data/sample_stories.json',
                        help='path to file with stories (?.json)')
    parser.add_argument('--output_file', default='data/sample_nphrases.json',
                        help='path to file with NPs (?.json)')
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        with open(args.input_file, 'r') as fh:
            stories = json.load(fh)
        fh.close()
        print(f'read {len(list(stories.keys()))} stories from {args.input_file}\n')
        NPs = extract_nphrases(stories)
        print(f'extracting NPs for {len(NPs)} samples complete\n')
        with open(args.output_file, 'w') as fh:
            json.dump(NPs, fh)
        fh.close()
        print(f'NPs saved to {args.output_file}\n')
    else:
        print(f'stories file does not exist at {args.input_file}')
