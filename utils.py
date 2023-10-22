from nltk.stem import WordNetLemmatizer
from PIL import Image
import sys
import torch
from torch.nn.functional import normalize

wnl = WordNetLemmatizer()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

cr_dict = {}
with open('data/concreteness_ratings.csv', 'r') as fh:
    cr_vals = fh.readlines()[1:]
fh.close()
for rec in cr_vals:
    cr_dict[rec.strip().split(';')[0]] = float(rec.strip().split(';')[1])
print(f'loaded concreteness ratings for {len(cr_dict)} words\n')

with open('data/all_pronouns.txt') as fh:
    pronouns = fh.readlines()
    pronouns = [p.strip() for p in pronouns]
fh.close()


def get_concreteness_ratings(NPs):
    filtered_NPs = []
    for NP in NPs:
        if NP not in filtered_NPs and NP not in pronouns:
            filtered_NPs.append(NP)

    cr_weights = []
    for NP in filtered_NPs:
        weight, cnt = 0.0, 0
        words_in_NP = NP.strip().split(' ')
        for word in words_in_NP:
            if word in cr_dict:
                weight += cr_dict[word]
                cnt += 1
            else:
                word = wnl.lemmatize(word)
                try:
                    weight += cr_dict[word]
                    cnt += 1
                except:
                    pass

        if cnt > 0:
            cr_weights.append(weight / cnt)
        else:
            cr_weights.append(1)

    return filtered_NPs, cr_weights


def get_image_ids(dataset, sid, sid_2_iids=None):
    if dataset == 'VIST':
        iids = sid_2_iids[sid]
        iids = [int(x) for x in iids]
    elif dataset == 'AESOP':
        iids = [sid + f'_{str(idx)}' for idx in range(3)]
    elif dataset == 'VWP':
        scene_id = sid.split(';')[0]
        iids = [scene_id + f'_{str(idx)}' for idx in range(sid_2_iids[scene_id])]
    else:
        print(f'get_image_ids() not implemented for dataset: {dataset}')
        sys.exit(1)

    return iids


def get_max_alignment_scores(NPs_embs, iids, preprocess, model, IRs_df, IRs_path, B=10):
    alignment_scores = torch.zeros((len(iids), NPs_embs.shape[0]))
    for idx in range(len(iids)):
        iid, IRs = iids[idx], []
        try:
            IR_boxes = IRs_df[IRs_df['image_id'] == iid]['bbox'].tolist()[:B]
            for IR_box in IR_boxes:
                IR_name = str(iid) + '_' + IR_box + '.jpg'
                IRs.append(preprocess(Image.open(IRs_path + '/' + IR_name).convert('RGB')).unsqueeze(0).to(device))

            with torch.no_grad():
                IRs_embs = normalize(model.encode_image(torch.stack(IRs).squeeze(1)), p=2, dim=-1)

            alignment_matrix = NPs_embs @ IRs_embs.T
            alignment_scores[idx] = torch.max(alignment_matrix, dim=1)[0].cpu()
        except Exception as e:
            print(f'image {iid} not used for alignment', e)

    max_alignments = torch.max(alignment_scores, dim=0)[0]

    # <UNCOMMENT lines below for indices of best matching images per NP> #
    # max_alignments_inds = torch.max(alignment_scores, dim=0)[1]
    # print(f'max_alignments_inds: {max_alignments_inds}')

    return max_alignments


def penalize_concretize_normalize(NPs_scores, cr_weights, theta=0.5, NPs=None):
    NPs_scores *= 2.5
    NPs_scores = torch.where(NPs_scores >= theta, NPs_scores, -(theta - NPs_scores))
    NPs_scores = NPs_scores * cr_weights

    if NPs != None:
        print(f'scores of NPs: {dict(zip(NPs, NPs_scores.tolist()))}')

    return (NPs_scores.sum() / NPs_scores.size()[0]).item()
