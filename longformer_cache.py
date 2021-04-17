import os
import sys
import json
import torch

import numpy as np
from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("/shared/nas/data/m1/wen17/BERT_CACHE/longformer-base-4096", use_fast=True)
tokenizer = AutoTokenizer.from_pretrained("longformer-base-4096", use_fast=True)

# model = AutoModel.from_pretrained("/shared/nas/data/m1/wen17/BERT_CACHE/longformer-base-4096")
model = AutoModel.from_pretrained("longformer-base-4096")

if torch.cuda.is_available():
    model = model.cuda()
def cache_ace(doc, outpath):
    cnt_all = len(doc.keys())
    cnt_cur = 0
    for doc_id in doc:
        cnt_cur += 1
        text = doc[doc_id]["doc"]
        sent_offset = doc[doc_id]["sent_offset"]
        sent_offset.append(len(text))

        doc_emb = []
        doc_text = ""
        origin_token_offset = []
        for t in text:
            if doc_text != "":
                doc_text += " "
            origin_token_offset.append((len(doc_text), len(doc_text)+len(t)))
            doc_text += t

        tokenized_output = tokenizer(doc_text, return_offsets_mapping=True, add_special_tokens=False)
        pieces = tokenized_output["input_ids"]
        tid2pid = [0]
        for j in range(1, len(origin_token_offset)):
            token_start_offset = origin_token_offset[j][0]
            for k, (piece_start_offset, piece_end_offset) in enumerate(tokenized_output['offset_mapping']):
                    if piece_start_offset == 0 and piece_end_offset == 0:
                        continue
                    if piece_start_offset == token_start_offset:
                        token_to_start_piece_idx = k
                        break
            tid2pid.append(token_to_start_piece_idx)
        tid2pid.append(len(pieces))

        ids = torch.tensor([pieces])
        if torch.cuda.is_available():
            ids = ids.cuda()
        emb = model(ids)
        print(emb[0].size())
        piece_emb = emb[0].squeeze(0).cpu().detach().numpy()
        for tid in range(1,len(tid2pid)):
            doc_emb.append(np.mean(piece_emb[tid2pid[tid-1]:tid2pid[tid]], axis=0))
            print(tokenizer.decode(pieces[tid2pid[tid-1]:tid2pid[tid]]))

        np.save(os.path.join(outpath, doc_id+'.npy'), doc_emb)
        print('cache bert emb for {}-th/{} doc: doc_id = {} with doc_emb shape = {}'.format(cnt_cur, cnt_all,doc_id, len(doc_emb)))

if __name__ == '__main__':

    dataset = 'ace'

    if len(sys.argv) != 2:
        raise ValueError("Missing doc_ace.json")
    input_file = sys.argv[1]
    output_dir = "./longformer_cache"

    if dataset == 'ace':
        inpath = os.path.join(input_file)
        outdir = os.path.join(output_dir)

        doc = json.load(open(inpath, 'r'))
        cache_ace(doc, outdir)
