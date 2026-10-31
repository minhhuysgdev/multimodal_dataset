# -*- coding:utf-8 -*-
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

def acc_log(trues, preds, preds_h, preds_m):
    if preds is not None:
        acc_pll(preds, trues)
    if preds_h is not None:
        acc(preds_h, trues, 'pseudo-label')
    if preds_m is not None:
        acc(preds_m, trues, 'model-predict')

def acc_pll(ps, ts):
    compare = []
    for p, t in zip(ps, ts):
        compare.append(p[t] > 0)
    compare = np.array(compare)
    s = int(compare.sum())
    l = len(compare)
    if l > 0:
        print(f'candidate acc: {s} / {l} = {round(s/l, 4)}')

def acc(ps, ts, type):
    compare = (ps == ts)
    if len(compare) > 0:
        print(f'{type} acc: {compare.sum()} / {len(compare)} = {round(compare.sum() / len(compare), 4)}')

def evaluate(args, model_s1, eval_dataset, data_collator, return_output=False):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=data_collator,
    )

    eval_dict = {'s1': 0.0}

    model_s1.eval()
    
    length = len(eval_dataloader.dataset)
    preds_s1 = torch.zeros((length)).long().cuda()
    confs_s1 = torch.zeros((length)).cuda()
    trues = torch.zeros((length)).long().cuda()

    for _, inputs in enumerate(eval_dataloader):
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        with torch.no_grad():
            index = inputs['index']
            labels = inputs['labels']
            myinputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
            logits_s1 = model_s1(**myinputs)
            logits = F.softmax(logits_s1, dim=1)
            preds_s1[index] = torch.argmax(logits_s1, dim=1)
            confs_s1[index] = torch.max(logits, dim=1)[0]
            trues[index] = labels.long()
    
    compare = (preds_s1 == trues)
    eval_dict['s1'] = float(sum(compare)) / len(compare)

    if return_output:
        return eval_dict, preds_s1.cpu().numpy(), confs_s1.cpu().numpy()

    return eval_dict