
import argparse
import os
from posix import environ
os.environ['CUDA_VISIBLE_DEVICES']='5'
import sys
import torch
import torch.nn.functional as F
import numpy as np
import math
import logging
import random
from dataset.persona_dataset import PersonaDataset
from batcher.persona_batcher import PersonaBatcher
from tqdm import tqdm
from str2bool import str2bool
import itertools
from datetime import datetime
import torch.nn as nn


from metric import f1_metric
from transformers import BertForSequenceClassification
from transformers import BertConfig

from transformers import AdamW, BertTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.optimization import (
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)
from torch.utils.data import DataLoader, Dataset
import json

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='Pre-training for Knowledge-Grounded Conversation')
# model
parser.add_argument("--debug",default=True,type=str2bool,help='debug mode, using small dataset')
parser.add_argument('--predict',type=str2bool,default=True)

#  files
parser.add_argument("--convo_path",type=str,default='/home/futc/persona/convo')
parser.add_argument("--persona_path",type=str,default='/home/futc/persona/history')
parser.add_argument("--knowledge_path",type=str,default='/home/futc/persona/knowledge')
parser.add_argument("--pseudo_path",type=str,default='/home/futc/2021work2/pseudo')

# model 
parser.add_argument("--vocab",type=str,default='/home/futc/bert-base-uncased') 
parser.add_argument("--priorp_model",type=str,default='/home/futc/bert-base-uncased') #(P|C) model
parser.add_argument("--semip_model",type=str,default='/home/futc/bert-base-uncased')# (P|CR), the distilled semip model 

# parser.add_argument("--count_path",type=str,default='/home/futc/2021work2/knowledge_count.json')
# parser.add_argument("--label_path",type=str,default='/home/futc/2021work2/label.json')
# training scheme
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--eval_batch_size', type=int, default=4)
parser.add_argument('--num_steps', type=int, default=1000000)
parser.add_argument('--accum_step', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--clip', type=float, default=2.0)
parser.add_argument('--schedule', type=str, default='linear')

parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--warmup_steps', type=int, default=500)
parser.add_argument('--num_epochs', type=int, default=3)

# log
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--valid_every', type=int, default=10000)

# save
parser.add_argument("--dump_path",type=str,default='/home/futc/2021work2/dump')
parser.add_argument('--exp_name', type=str, default='debug')
parser.add_argument('--log', type=str, default='log')
parser.add_argument('--seed', type=int, default=42)

parser.add_argument("--max_context_length",type=int,default=64)
parser.add_argument("--max_persona_length",type=int,default=64)
parser.add_argument("--max_response_length",type=int,default=64)
parser.add_argument("--max_knowledge_length",type=int,default=64)
parser.add_argument("--n_knowledge",type=int,default=32)
parser.add_argument("--n_persona",type=int,default=32)

# gpu
parser.add_argument('--gpu_list', type=str, default='4')
parser.add_argument('--gpu_ratio', type=float, default=0.85)
parser.add_argument('--n_device', type=int, default=8)
parser.add_argument('--no_cuda', type=str2bool, default=False)

parser.add_argument("--n_layer",type=int,default=6)


args = parser.parse_args()
if args.debug:
    args.print_every=2
    args.valid_every=8
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
out_dir = os.path.join(args.dump_path, args.exp_name)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
args.out_dir=out_dir
logger.addHandler(logging.FileHandler(os.path.join(args.out_dir, "log"), 'w'))
logger.info("\nParameters:")
for attr, value in sorted(vars(args).items()):
    logger.info("{}={}".format(attr.upper(), value))




def recall_f1(scores,knowledges,responses):
    count=[len(k) for k in knowledges]
    # all equals to the numbert of context
    assert len(knowledges)==len(responses)
    # all equals to the total number of knowledge sentences in every case
    assert sum(count)==len(scores)
    n=len(knowledges)
    preds=[]
    for i in range(n):
        score=scores[:len(knowledges[i])]
        scores=scores[len(knowledges[i]):]
        knowledge=knowledges[i]
        pred=knowledge[score.index(max(score))]
        preds.append(pred)
    return f1_metric(preds,responses)


def recall_metric(scores):
    r1,r2,r5,r10=0.,0.,0.,0.
    #count_path=r'/home/futc/cmudog/'+'train'+'_knowledge_count.json'
    #label_path=r'/home/futc/cmudog/'+'train'+'_label_index.json'
    with open(args.count_path,mode='r',encoding='utf-8')as f:
        knowledge_count=json.load(f)
    with open(args.label_path,mode='r',encoding='utf-8')as f:
        label=json.load(f)
    
    assert len(scores)==np.array(knowledge_count).sum()
    assert len(knowledge_count)==len(label)

    for i in range(len(knowledge_count)):
        score=scores[:knowledge_count[i]]
        scores=scores[knowledge_count[i]:]
        order=np.argsort(score)[::-1]
        gold=label[i]
        #gold=0 if correct_first else label[i]
        if gold in order[:1]:
            r1+=1
        if gold in order[:2]:
            r2+=1
        if gold in order[:5]:
            r5+=1
        if gold in order[:10]:
            r10+=1

    return r1/len(knowledge_count),r2/len(knowledge_count),r5/len(knowledge_count),r10/len(knowledge_count)




# Output directory for models and summaries

#print('Writing to {}\n'.format(out_dir))
#save_hparams(args, os.path.join(out_dir, 'hparams'))


# Checkpoint directory
# checkpoint_dir = os.path.join(out_dir, 'checkpoints')
# checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
# if not os.path.exists(checkpoint_dir):
#     os.makedirs(checkpoint_dir)
# sys.stdout.flush()

# Build dataset
time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.info("Create training dataset begin... | %s " % time_str)

train_dataset=PersonaDataset(args.convo_path,args.persona_path,args.knowledge_path,args.pseudo_path,'train',args.debug)
eval_dataset=PersonaDataset(args.convo_path,args.persona_path,args.knowledge_path,args.pseudo_path,'eval',args.debug)
train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=PersonaDataset.collate_fn)
eval_loader=DataLoader(eval_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=PersonaDataset.collate_fn)
train_loader=itertools.cycle(train_loader)
logger.info("train examples {}".format(len(train_dataset)))
logger.info("eval examples {}".format(len(eval_dataset)))

time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.info("Create training dataset end... | %s " % time_str)
tokenizer=BertTokenizer.from_pretrained(args.vocab)
tokenizer.add_special_tokens({'pad_token':'[PAD]','sep_token':'[SEP]'})
batcher=PersonaBatcher(device,tokenizer,args.n_knowledge,args.n_persona,args.max_context_length,args.max_response_length,args.max_knowledge_length,args.max_persona_length)

configuration=BertConfig(num_hidden_layers=args.n_layer)
semip_model=BertForSequenceClassification.from_pretrained(args.semip_model,config=configuration)
semip_model.resize_token_embeddings(len(tokenizer))
priorp_model=BertForSequenceClassification.from_pretrained(args.priorp_model,config=configuration)
priorp_model.resize_token_embeddings(len(tokenizer))

semip_model.to(device)
priorp_model.to(device)

no_decay = ["bias", "LayerNorm.weight"]
priorp_parameters = [
    {
        "params": [p for n, p in priorp_model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in priorp_model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
semip_parameters = [
    {
        "params": [p for n, p in semip_model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in semip_model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
priorp_optimizer = AdamW(priorp_parameters, lr=args.lr, eps=args.adam_epsilon)
semip_optimizer = AdamW(semip_parameters, lr=args.lr, eps=args.adam_epsilon)
total_steps = args.num_epochs * (len(train_dataset) / (args.batch_size * args.accum_step))
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
if args.schedule == 'linear':
    priorp_scheduler = get_linear_schedule_with_warmup(priorp_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    semip_scheduler = get_linear_schedule_with_warmup(semip_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
elif args.schedule == 'cosine':
    priorp_scheduler = get_cosine_schedule_with_warmup(priorp_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    semip_scheduler = get_cosine_schedule_with_warmup(semip_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)


def train_step(global_step):
    klloss_total = 0.0
    semip_model.eval()
    priorp_model.train()
    for _ in range(args.accum_step):
        context_list,response_list,persona_list,knowledge_list, plabel_list,klabel_list= next(train_loader)
        batcher.load(context_list,response_list,persona_list,knowledge_list,plabel_list,klabel_list)
        with torch.no_grad():
            batch_dict=batcher('p|cr',None,None)
            input_id=batch_dict['input_id']
            segment_id=batch_dict['segment_id']
            assert input_id.dim()==3 and input_id.shape==segment_id.shape
            bs,n_per,seq_len=input_id.shape
            semi_plogits=semip_model(input_ids=input_id.view(-1,seq_len),attention_mask=input_id.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id.view(-1,seq_len),return_dict=True)['logits']
            semi_plogits=semi_plogits.view(bs,n_per,-1)[:,:,1]
        batch_dict=batcher('p|c',None,None)
        input_id=batch_dict['input_id']
        segment_id=batch_dict['segment_id']
        assert input_id.dim()==3 and input_id.shape==segment_id.shape
        bs,n_per,seq_len=input_id.shape
        prior_plogits=priorp_model(input_ids=input_id.view(-1,seq_len),attention_mask=input_id.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id.view(-1,seq_len),return_dict=True)['logits']
        prior_plogits=prior_plogits.view(bs,n_per,-1)[:,:,1]
        klloss=F.kl_div(torch.log_softmax(prior_plogits,dim=1),torch.softmax(semi_plogits,dim=1))
        klloss_total+=klloss.item()
        klloss=klloss/args.accum_step
        klloss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in semip_model.parameters() if p.requires_grad], args.clip)
    if grad_norm >= 1e2:
        logger.info('WARNING : Exploding Gradients {:.2f}'.format(grad_norm))
    priorp_optimizer.step()
    priorp_scheduler.step()
    priorp_optimizer.zero_grad()

    if global_step % args.print_every == 0 and global_step != 0:
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info("Step: %d \t| ks_loss: %.3f \t| lr: %.8f \t| %s" % (
            global_step, klloss_total, semip_scheduler.get_lr()[0], time_str
        ))

def predict_step(global_step):
    #if split == 'test_seen':
    #    test_loader = test_seen_loader
    #else:
    #    raise ValueError
    hypothesis=[]
    priorp_model.eval()
    count = 0
    hit1=0
    hit2=0
    hit5=0
    hit10=0
    with torch.no_grad():
        for context_list,response_list,persona_list,knowledge_list,plabel_list,klabel_list in eval_loader:
            batcher.load(context_list,response_list,persona_list,knowledge_list,plabel_list,klabel_list)
            
            batch_dict=batcher('p|c',None,None)
            input_id=batch_dict['input_id']
            segment_id=batch_dict['segment_id']
            assert input_id.dim()==3 and input_id.shape==segment_id.shape
            bs,n_per,seq_len=input_id.shape
            logits = priorp_model(input_ids=input_id.view(-1,seq_len),attention_mask=input_id.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id.view(-1,seq_len),return_dict=True)['logits']
            logits = logits.view(bs,n_per,-1)[:,:,1]
            count += bs
            if count % 1000 == 0:
                logger.info("eval finishing {}".format(count))
            bs=len(context_list)
            ref=torch.tensor([persona_list[i].index(plabel_list[i]) for i in range(bs)],dtype=torch.long,device=device)
            # when only selected one persona
            # hyp=torch.max(logits,dim=1)[1]
            # hit1+=torch.sum(hyp==ref,dim=0).item()
            # when select multi
            #(bs,k)
            hyp=torch.topk(logits,k=10)[1]
            hit1+=torch.sum(torch.sum(hyp[:,:1]==ref[:,None],dim=1),dim=0).item()
            hit2+=torch.sum(torch.sum(hyp[:,:2]==ref[:,None],dim=1),dim=0).item()
            hit5+=torch.sum(torch.sum(hyp[:,:5]==ref[:,None],dim=1),dim=0).item()
            hit10+=torch.sum(torch.sum(hyp[:,:10]==ref[:,None],dim=1),dim=0).item()
            try:
                hyp=hyp.detach().cpu().tolist()
                for i in range(bs):
                    hypothesis.append('<#p#>'.join([persona_list[i][min(j,len(persona_list[i])-1)] for j in hyp[i]]))
            except:
                logger.info("error when decoding hypothesis")

    priorp_model.save_pretrained(os.path.join(args.out_dir,'{}step_priorp_model'.format(global_step)))
    logger.info("Saved model checkpoint \n")
    logger.info("hit at 1 is {:.4f}".format(hit1/count))
    logger.info("hit at 2 is {:.4f}".format(hit2/count))
    logger.info("hit at 5 is {:.4f}".format(hit5/count))
    logger.info("hit at 10 is {:.4f}".format(hit10/count))

    with open(os.path.join(args.out_dir,'{}step_persona'.format(global_step)),mode='w',encoding='utf-8') as f:
        for hyp in hypothesis:
            f.write(hyp+'\n')


    #logger.info("Saved model checkpoint to {}\n".format(checkpoint_dir))

    # f1=recall_metric(scores,test_knowledges,test_responses)
    # return f1
    # r1, r2, r5, r10 = recall_metric(scores)
    # logger.info("RECALL-1/2/5/10: {:.4f}/{:.4f}/{:.4f}/{:.4f}".format(r1, r2, r5, r10))
    # logger.info("**********************************")
    #sys.stdout.flush()

    # return {'r_at_1': r1, 'r_at_2': r2, 'r_at_5': r5, 'r_at_10': r10}

best_f1 = -1.
if args.predict:
    predict_step(0)
    #logger.info("predict result: the f1 between predict knowledge and response: {:.6f}".format(f1))
    exit()
for i in range(args.num_steps):
    train_step(i + 1)
    if (i + 1) % args.valid_every == 0:
        predict_step(i+1)