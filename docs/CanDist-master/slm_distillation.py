# -*- coding:utf-8 -*-
import logging
import os
import random
import copy
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, KLDivLoss
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import trange
import pandas as pd
import torch.nn.functional as F
import time

from transformers import (
    AdamW,
    RobertaConfig,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from models.modeling_roberta import RobertaForSequenceClassification_Modified
from utils.eval import *
from utils.config import config
from utils.datasets import ClassificationDataset, MyDataCollator

logger = logging.getLogger(__name__)


torch.set_printoptions(profile="full")

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def initialize(args, t_total, num_labels, epoch):
    config_class, model_class, _ = (RobertaConfig, RobertaForSequenceClassification_Modified, RobertaTokenizer)
    config_s1 = config_class.from_pretrained(
        args.student1_config_name if args.student1_config_name else args.student1_model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
        hidden_dropout_prob=args.hidden_dropout_prob,
    )
    model_s1 = model_class.from_pretrained(
        args.student1_model_name_or_path,
        from_tf=bool(".ckpt" in args.student1_model_name_or_path),
        config=config_s1,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model_s1.to(args.device)

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters_1 = [
        {
            "params": [p for n, p in model_s1.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model_s1.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer_s1 = AdamW(optimizer_grouped_parameters_1, lr=args.learning_rate, \
            eps=args.adam_epsilon, betas=(args.adam_beta1,args.adam_beta2))
    scheduler_s1 = get_linear_schedule_with_warmup(
        optimizer_s1, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        # model_t = torch.nn.DataParallel(model_t)
        model_s1 = torch.nn.DataParallel(model_s1)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model_s1 = torch.nn.parallel.DistributedDataParallel(
            model_s1, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    model_s1.zero_grad()

    return model_s1, optimizer_s1, scheduler_s1


def evaluation(args, model_s1, eval_dataset, data_collator, best_eval_dict):    
    eval_dict = evaluate(args, model_s1, eval_dataset, data_collator)

    for k in eval_dict.keys():
        logger.info("  %s = %s", k, str(eval_dict[k]))

    for k in best_eval_dict.keys():
        if eval_dict[k] > best_eval_dict[k]:
            best_eval_dict[k] = eval_dict[k]

            if args.local_rank in [-1, 0]:
                path = os.path.join(args.output_dir+k, "checkpoint-best")
                logger.info("Saving model checkpoint to %s", path)
                if not os.path.exists(path):
                    os.makedirs(path)
                if k == 's1':
                    model = model_s1
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(path)


def train(args, train_dataset_aug, data_collator, traindev_dataset, testdev_dataset, dev_dataset):
    """ Train the model """
    num_labels = args.num_labels
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset_aug) if args.local_rank==-1 else DistributedSampler(train_dataset_aug)
    train_dataloader = DataLoader(
        train_dataset_aug,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=data_collator,
    )

    t_total = len(train_dataloader)//args.gradient_accumulation_steps*args.num_train_epochs

    model_s1, optimizer_s1, scheduler_s1 = initialize(args, t_total, num_labels, 0)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset_aug))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0

    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    best_eval_dict = {
        's1-train': 0.0,
        's1-train-epoch': -1,
        's1-test': 0.0,
        's1-test-epoch': -1,
        's1-val': 0.0,
        's1-val-epoch': -1,
    }

    # pseudo-labels
    length = len(train_dataloader.dataset)
    confidence1 = torch.zeros((length, num_labels)).cuda()
    for i in range(length):
        confidence1[i] = torch.tensor(train_dataloader.dataset[i]['inputs'].label).cuda()

    for epoch in train_iterator:

        time_s = time.time()
        
        chosen_list1 = {
            'train': torch.zeros(length).bool().cuda(),  # for training
            'clean': torch.zeros(length).bool().cuda(),  # label clean
            'p-clean': torch.zeros(length).bool().cuda(),  # partial label clean
            'high-conf': torch.zeros(length).bool().cuda(),  # high-confidence
        }
        chosen_list_hc_match = torch.zeros(length).bool().to(args.device)
            
        if epoch >= args.warmup_epoch:

            # not used for training
            df = pd.read_csv(f'data/{args.task_name}/train.csv', header=None)
            trues = np.array(df[0].tolist())
            logger.info(f"Epoch:{epoch}, selection information")

            # ================================= evaluating model outputs =================================
            t_model_s1 = copy.deepcopy(model_s1).to(args.device)
            t_model_s1.eval()
                
            outputs_all1 = torch.zeros((length, num_labels)).cuda()
            loss_all1 = torch.zeros((length)).cuda()
            embeddings_all1 = torch.zeros((length, args.embedding_dim)).cuda()
            
            for step, inputs in enumerate(train_dataloader):
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                with torch.no_grad():
                    index = inputs['index']
                    bs = len(index)
                    labels1 = confidence1[index]
                    del inputs['index']
                    del inputs['labels']
                    myinputs = {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                        "is_feat": True,
                    }
                    embeddings1, logits1 = t_model_s1(**myinputs)
                    embeddings_all1[index] = embeddings1.detach()
                    outputs_all1[index] = torch.softmax(logits1.clone().detach(), dim=1)
                    loss_fct = CrossEntropyLoss(reduction='none')
                    loss1 = loss_fct(logits1, labels1)
                    loss_all1[index] = loss1
        
            # ================================= selection for model1 =================================
            logger.info(f"Selection for model-1")
            high_conf_all = outputs_all1.max(dim=-1)[0]
            pred_idx_all = outputs_all1.max(dim=-1)[1]
            preds_m1 = pred_idx_all.cpu().numpy()                 
            preds1 = confidence1.cpu().numpy()
            preds_h1 = np.argmax(preds1, axis=1)

            if epoch > 0:
                # class-wise selection
                chosen_list1['clean'].fill_(False)
                print(f"\nFor class-wise small loss selection tau: {args.tau_loss}")
                for i in range(num_labels):
                    max_indices = torch.argmax(confidence1, dim=1)
                    idx_i = torch.where(max_indices == i)[0]
                    if len(idx_i) == 0:
                        print(f'No prediction on class {i}')
                        continue
                    sorted_idx_i = torch.argsort(loss_all1[idx_i])
                    chosen_num = max(int(len(idx_i) * args.tau_loss), 1)
                    chosen_list1['clean'][idx_i[sorted_idx_i[:chosen_num]]] = True
                chosen_idx = chosen_list1['clean'].cpu().numpy()
                acc_log(trues[chosen_idx], preds1[chosen_idx], preds_h1[chosen_idx], preds_m1[chosen_idx])

                # select partial clean (model output in pred)
                chosen_list1['p-clean'].fill_(False)
                for i in range(length):
                    if confidence1[i][pred_idx_all[i]] > 0:
                        chosen_list1['p-clean'][i] = True
                print('\nFor chosen partial clean')
                chosen_idx = chosen_list1['p-clean'].cpu().numpy()
                acc_log(trues[chosen_idx], preds1[chosen_idx], preds_h1[chosen_idx], preds_m1[chosen_idx])

                # high-conf on others
                idx_chosen = torch.where(high_conf_all[~chosen_list1['clean']] > args.tau_conf)[0]  # expect clean labels
                chosen_list1['high-conf'].fill_(False)
                if len(idx_chosen) > 0:
                    idx_ur = torch.where(chosen_list1['clean'] == False)[0]
                    chosen_list1['high-conf'][idx_ur[idx_chosen]] = True 
                    print(f'\nFor chosen high-conf: {args.tau_conf}')
                    chosen_idx = chosen_list1['high-conf'].cpu().numpy()
                    acc_log(trues[chosen_idx], None, None, preds_m1[chosen_idx])
                pred_idx_all1 = pred_idx_all.clone()

                # logging others
                unchosen_list = ~(chosen_list1['clean'] + chosen_list1['p-clean'] + chosen_list1['high-conf'])
                print('\nFor unchosen')
                chosen_idx = unchosen_list.cpu().numpy()
                acc_log(trues[chosen_idx], preds1[chosen_idx], preds_h1[chosen_idx], preds_m1[chosen_idx])
           
        if epoch >= args.warmup_epoch:
            chosen_list_pc_match = chosen_list1['p-clean']
            chosen_list_hc_match = chosen_list1['high-conf']
            
            chosen_list1['train'] = chosen_list1['clean'] + chosen_list_pc_match + chosen_list_hc_match
            chosen_idx = chosen_list1['train'].cpu().numpy()
            print('\nFor train')
            acc_log(trues[chosen_idx], preds1[chosen_idx], preds_h1[chosen_idx], preds_m1[chosen_idx])
                
        else:
            logger.info('Warmup stage, No selection')
            chosen_list1['train'].fill_(True)
            chosen_list1['clean'].fill_(False)
            chosen_list_hc_match.fill_(False)

        for step, inputs in enumerate(train_dataloader):
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            model_s1.train()

            w = linear_rampup(epoch, 10)
            labels = inputs.get("labels")  # inputs_bt.label stores original partial labels
            bs = labels.shape[0]
            batch_index = inputs['index']

            mask_train1 = chosen_list1['train'][batch_index]
            mask_clean1 = chosen_list1['clean'][batch_index]
            mask_hc_m = chosen_list_hc_match[batch_index]
            index_x1 = torch.where(mask_train1)[0]
            index_u1 = torch.where(~mask_train1)[0]
            index_c1 = torch.where(mask_clean1)[0]
            index_uc1 = torch.where(~mask_clean1)[0]
            index_hcm = torch.where(mask_hc_m)[0]

            myinputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "is_feat": True,
            }
            embeddings1, logits1 = model_s1(**myinputs)
            
            labels1 = confidence1[batch_index].clone()
            loss_fct = CrossEntropyLoss(reduction='none')   
            loss_fct_un = KLDivLoss(reduction='sum')
            
            if epoch >= args.warmup_epoch: 
                labels1_sp = labels1[index_c1] ** (1 / args.sharpen)
                labels1[index_c1] = labels1_sp / labels1_sp.sum(dim=1, keepdim=True)
            if epoch >= args.warmup_epoch:                    
                pred_tgt = pred_idx_all1[batch_index]
                labels_m = torch.zeros_like(labels).cuda()
                labels_m[torch.arange(bs), pred_tgt] = 1.0
                labels1[index_hcm] = labels_m[index_hcm]
            loss_l1 = loss_fct(logits1[index_x1], labels1[index_x1]).mean()
            
            # embedding-level mix-up
            if args.is_mu:
                valid_embeddings = embeddings1[index_x1]
                all_targets = labels1[index_x1].clone().detach()
                rand_idx = torch.randperm(valid_embeddings.size(0))
                l = np.random.beta(4,4) 
                l = max(l, 1 - l)
                mixed_embeddings1 = l * valid_embeddings + (1 - l) * valid_embeddings[rand_idx]
                mixed_targets1 = l * all_targets + (1 - l) * all_targets[rand_idx]
                # loss
                mixed_logits1 = model_s1.classifier(mixed_embeddings1)
                loss_mix1 = loss_fct(mixed_logits1, mixed_targets1).mean()
                final_loss1 = loss_l1 + w * loss_mix1
            else:
                final_loss1 = loss_l1
            
            # consistency loss, if with data augmentation
            if args.is_cr:
                myinputs_bt = {
                    "input_ids": inputs["input_ids_bt"],
                    "attention_mask": inputs["attention_mask_bt"],
                }
                logits_bt1 = model_s1(**myinputs_bt)
                loss_cr1 = loss_fct(logits_bt1[index_x1], labels1[index_x1]).mean()
                loss_cr_un1 = 0.5 * loss_fct_un(F.log_softmax(logits1[index_u1]/args.temp_u, dim=-1), F.softmax(logits_bt1[index_u1].detach().clone(), dim=-1)) \
                    + 0.5 * loss_fct_un(F.log_softmax(logits_bt1[index_u1]/args.temp_u, dim=-1), F.softmax(logits1[index_u1].detach().clone(), dim=-1))
                final_loss1 = final_loss1 + w * (loss_cr1 + loss_cr_un1)
            
            # proden update confidence-1
            new_labels = labels1.clone()
            new_labels[new_labels > 0]  = 1
            new_labels = new_labels * F.softmax(logits1.detach(), dim=1)
            new_labels = new_labels / new_labels.sum(dim=1, keepdim=True)
            confidence1[batch_index] = new_labels.detach()            

            if args.n_gpu > 1:
                final_loss1 = final_loss1.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                final_loss1 = final_loss1 / args.gradient_accumulation_steps

            final_loss1.backward()

            if (step+1) % 15 == 0:
                print(f"  Epoch:{epoch}, Step:{step}, Lr:{optimizer_s1.param_groups[0]['lr']}, Total_loss1:{final_loss1.item()}, Labeled_loss1:{loss_l1.item()}")
                
            if (step+1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model_s1.parameters(), args.max_grad_norm)
                optimizer_s1.step()
                scheduler_s1.step()  # Update learning rate schedule
                model_s1.zero_grad()
                global_step += 1

        time_e = time.time()
        logger.info(f"TIME:{time_e-time_s}")
        
        '''
        best_eval_dict = {
            's1-train': 0.0,
            's1-train-epoch': -1,
            's1-test': 0.0,
            's1-test-epoch': -1,
        }
        '''
        logger.info("***** Epoch : %d, evaluation traindev *****", epoch)
        eval_dict = evaluate(args, model_s1, traindev_dataset, data_collator)
        for k in eval_dict.keys():
            logger.info("  %s = %s", k, str(eval_dict[k]))
        if eval_dict['s1'] > best_eval_dict['s1-train']:
            best_eval_dict['s1-train'] = eval_dict['s1']
            best_eval_dict['s1-train-epoch'] = epoch
            if args.local_rank in [-1, 0]:
                path = os.path.join(args.output_dir+k, "checkpoint-best-train")
                logger.info("Saving model checkpoint to %s", path)
                if not os.path.exists(path):
                    os.makedirs(path)
                model_to_save = (
                    model_s1.module if hasattr(model_s1, "module") else model_s1
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(path)
        
        logger.info("***** Epoch : %d, evaluation testdev *****", epoch)
        eval_dict = evaluate(args, model_s1, testdev_dataset, data_collator)
        for k in eval_dict.keys():
            logger.info("  %s = %s", k, str(eval_dict[k]))
        if eval_dict['s1'] > best_eval_dict['s1-test']:
            best_eval_dict['s1-test'] = eval_dict['s1']
            best_eval_dict['s1-test-epoch'] = epoch
            if args.local_rank in [-1, 0]:
                path = os.path.join(args.output_dir+k, "checkpoint-best-test")
                logger.info("Saving model checkpoint to %s", path)
                if not os.path.exists(path):
                    os.makedirs(path)
                model_to_save = (
                    model_s1.module if hasattr(model_s1, "module") else model_s1
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(path)
        
        logger.info("***** Epoch : %d, evaluation valdev *****", epoch)
        eval_dict = evaluate(args, model_s1, dev_dataset, data_collator)
        for k in eval_dict.keys():
            logger.info("  %s = %s", k, str(eval_dict[k]))
        if eval_dict['s1'] >= best_eval_dict['s1-val']:
            best_eval_dict['s1-val'] = eval_dict['s1']
            best_eval_dict['s1-val-epoch'] = epoch
            if args.local_rank in [-1, 0]:
                path = os.path.join(args.output_dir+k, "checkpoint-best-val")
                logger.info("Saving model checkpoint to %s", path)
                if not os.path.exists(path):
                    os.makedirs(path)
                model_to_save = (
                    model_s1.module if hasattr(model_s1, "module") else model_s1
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(path)

def main():
    args = config()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", "%m/%d/%Y %H:%M:%S")
    logging_fh = logging.FileHandler(os.path.join(args.output_dir, 'results.log'))
    logging_fh.setLevel(logging.DEBUG)
    logging_fh.setFormatter(formatter)
    logger.addHandler(logging_fh)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    tokenizer = RobertaTokenizer.from_pretrained(
        args.tokenizer_name,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Load datasets
    # for training (LLM's annotation)
    train_dataset_aug = ClassificationDataset(args.data_dir, tokenizer, args.task_name, args.max_seq_length,
                                  args.overwrite_cache, mode="train_chat", num_labels=args.num_labels, label_source=args.label_source)
    # for evaluation
    train_dataset = ClassificationDataset(args.data_dir, tokenizer, args.task_name, args.max_seq_length,
                                  args.overwrite_cache, mode="train", num_labels=args.num_labels)
    test_dataset = ClassificationDataset(args.data_dir, tokenizer, args.task_name, args.max_seq_length,
                                 args.overwrite_cache, mode="test", num_labels=args.num_labels)
    # for validation
    dev_dataset = ClassificationDataset(args.data_dir, tokenizer, args.task_name, args.max_seq_length,
                                 args.overwrite_cache, mode="dev", num_labels=args.num_labels)
    traindev_dataset = train_dataset
    testdev_dataset = test_dataset
            
    train(args, train_dataset_aug, data_collator=MyDataCollator(), traindev_dataset=traindev_dataset, testdev_dataset=testdev_dataset, dev_dataset=dev_dataset)
    predict(args, train_dataset, test_dataset, data_collator=MyDataCollator())

def predict(args, train_dataset, test_dataset, data_collator):

    path = os.path.join(args.output_dir+'s1', "checkpoint-best-train")
    model_s1 = RobertaForSequenceClassification_Modified.from_pretrained(path)
    model_s1.to(args.device)
    logger.info("***** best transductive results *****")
    eval_dict_train = evaluate(args, model_s1, train_dataset, data_collator)
    for k in eval_dict_train.keys():
        logger.info("  %s = %s", k, str(eval_dict_train[k]))
    
    path = os.path.join(args.output_dir+'s1', "checkpoint-best-test")
    model_s1 = RobertaForSequenceClassification_Modified.from_pretrained(path)
    model_s1.to(args.device)
    logger.info("***** best inductive results *****")
    eval_dict_test = evaluate(args, model_s1, test_dataset, data_collator)
    for k in eval_dict_test.keys():
        logger.info("  %s = %s", k, str(eval_dict_test[k]))
    
    path = os.path.join(args.output_dir+'s1', "checkpoint-best-val")
    model_s1 = RobertaForSequenceClassification_Modified.from_pretrained(path)
    model_s1.to(args.device)
    logger.info("***** val transductive results *****")
    eval_dict_train = evaluate(args, model_s1, train_dataset, data_collator)
    for k in eval_dict_train.keys():
        logger.info("  %s = %s", k, str(eval_dict_train[k]))
    
    path = os.path.join(args.output_dir+'s1', "checkpoint-best-val")
    model_s1 = RobertaForSequenceClassification_Modified.from_pretrained(path)
    model_s1.to(args.device)
    logger.info("***** val inductive results *****")
    eval_dict_test = evaluate(args, model_s1, test_dataset, data_collator)
    for k in eval_dict_test.keys():
        logger.info("  %s = %s", k, str(eval_dict_test[k]))

if __name__ == "__main__":
    main()
