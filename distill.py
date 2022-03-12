
import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES']='3'
import torch
import torch.nn.functional as F
import numpy as np
import math
import logging
import random
from tqdm import tqdm
from str2bool import str2bool
import itertools
from datetime import datetime
import torch.nn as nn

from dataset.persona_dataset import PersonaDataset
from batcher.persona_batcher import PersonaBatcher
from metric import f1_metric
from transformers import BertForSequenceClassification
from transformers import BertModel
from transformers import BertConfig
from model.bert_transformer import BertPosterior

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
parser.add_argument('--predict',type=str2bool,default=False)

#  files
parser.add_argument("--convo_path",type=str,default='/home/futc/persona/convo')
parser.add_argument("--persona_path",type=str,default='/home/futc/persona/history')
parser.add_argument("--knowledge_path",type=str,default='/home/futc/persona/knowledge')
parser.add_argument("--pseudo_path",type=str,default='/home/futc/2021work2/pseudo')

# model 
parser.add_argument("--vocab",type=str,default='/home/futc/bert-base-uncased')
parser.add_argument("--semip_model",type=str,default='/home/futc/2021work2/dump/semip1/3000step_semip')
parser.add_argument("--dualkp_model",type=str,default='/home/futc/2021work2/dump/dual3/1000step_dualkp_model')# (P|CRK) dual learning model

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

# truncate
parser.add_argument("--max_context_length",type=int,default=64)
parser.add_argument("--max_persona_length",type=int,default=64)
parser.add_argument("--max_response_length",type=int,default=64)
parser.add_argument("--max_knowledge_length",type=int,default=64)
parser.add_argument("--n_persona",default=32,type=int)
parser.add_argument("--n_knowledge",default=32,type=int)

# architecture
parser.add_argument("--loss",type=str,default='standard')
parser.add_argument("--temperature",type=float,default=2.0)
parser.add_argument("--alpha",type=float,default=0.95)


# gpu
parser.add_argument('--gpu_ratio', type=float, default=0.85)
parser.add_argument('--n_device', type=int, default=8)
parser.add_argument('--no_cuda', type=str2bool, default=False)

# architecture
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

train_dataset=PersonaDataset(args.convo_path,args.persona_path,args.knowledge_path,args.pseudo_path,mode='train')
eval_dataset=PersonaDataset(args.convo_path,args.persona_path,args.knowledge_path,args.pseudo_path,mode='eval')
train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=PersonaDataset.collate_fn)
eval_loader=DataLoader(eval_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=PersonaDataset.collate_fn)
train_loader=itertools.cycle(train_loader)

time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.info("Create training dataset end... | %s " % time_str)
tokenizer=BertTokenizer.from_pretrained(args.vocab)
tokenizer.add_special_tokens({'pad_token':'[PAD]','sep_token':'[SEP]'})
batcher=PersonaBatcher(device,tokenizer,args.n_knowledge,args.n_persona,args.max_context_length,args.max_response_length,args.max_knowledge_length,args.max_persona_length)

configuration=BertConfig(num_hidden_layers=args.n_layer)
dualkp_model=BertForSequenceClassification.from_pretrained(args.dualkp_model,config=configuration)
dualkp_model.resize_token_embeddings(len(tokenizer))
semip_model=BertForSequenceClassification.from_pretrained(args.semip_model,config=configuration)
semip_model.resize_token_embeddings(len(tokenizer))

dualkp_model.to(device)
semip_model.to(device)

no_decay = ["bias", "LayerNorm.weight"]
dualkp_parameters = [
    {
        "params": [p for n, p in dualkp_model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in dualkp_model.named_parameters() if any(nd in n for nd in no_decay)],
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
dualkp_optimizer = AdamW(dualkp_parameters, lr=args.lr, eps=args.adam_epsilon)
semip_optimizer = AdamW(semip_parameters, lr=args.lr, eps=args.adam_epsilon)
total_steps = args.num_epochs * (len(train_dataset) / (args.batch_size * args.accum_step))
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
if args.schedule == 'linear':
    dualkp_scheduler = get_linear_schedule_with_warmup(dualkp_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    semip_scheduler = get_linear_schedule_with_warmup(semip_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
elif args.schedule == 'cosine':
    dualkp_scheduler = get_cosine_schedule_with_warmup(dualkp_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    semip_scheduler = get_cosine_schedule_with_warmup(semip_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)


def train_step(global_step):
    distill_loss_total = 0.0
    dualkp_model.eval()
    semip_model.train()
    for _ in range(args.accum_step):
        #torch.cuda.empty_cache()
        context_list,response_list,persona_list,knowledge_list, plabel_list,klabel_list= next(train_loader)
        batcher.load(context_list,response_list,persona_list,knowledge_list,plabel_list,klabel_list)
        with torch.no_grad():
            batch_dict=batcher('p|crk',klabel_list,None)
            input_id=batch_dict['input_id']
            segment_id=batch_dict['segment_id']
            assert input_id.dim()==3 and input_id.shape==segment_id.shape
            bs,n_per,seq_len=input_id.shape
            post_plogits=dualkp_model(input_ids=input_id.view(-1,seq_len),attention_mask=input_id.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id.view(-1,seq_len),return_dict=True)['logits']
            post_plogits=post_plogits.view(bs,n_per,-1)[:,:,1]
        
        batch_dict=batcher('p|cr',None,None)
        input_id=batch_dict['input_id']
        segment_id=batch_dict['segment_id']
        assert input_id.dim()==3 and input_id.shape==segment_id.shape
        bs,n_per,seq_len=input_id.shape
        semi_plogits=semip_model(input_ids=input_id.view(-1,seq_len),attention_mask=input_id.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id.view(-1,seq_len),return_dict=True)['logits']
        semi_plogits=semi_plogits.view(bs,n_per,-1)[:,:,1]
        target = torch.tensor([persona_list[i].index(plabel_list[i]) for i in range(len(persona_list))], dtype=torch.long, device=device)
        if args.loss=='standard':
            loss=F.cross_entropy(semi_plogits,target)*(1-args.alpha) + nn.KLDivLoss()(F.log_softmax(semi_plogits/args.temperature,dim=1), F.softmax(post_plogits/args.temperature,dim=1))* (args.alpha*args.temperature*args.temperature)
        else:
            #TODO: other kinds of 
            raise NotImplementedError
        distill_loss_total+=loss.item()
        loss=loss/args.accum_step
        loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in semip_model.parameters() if p.requires_grad], args.clip)
    if grad_norm >= 1e2:
        logger.info('WARNING : Exploding Gradients {:.2f}'.format(grad_norm))
    semip_optimizer.step()
    semip_scheduler.step()
    semip_optimizer.zero_grad()

    if global_step % args.print_every == 0 and global_step != 0:
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info("Step: %d \t| ks_loss: %.3f \t| lr: %.8f \t| %s" % (
            global_step, distill_loss_total, semip_scheduler.get_lr()[0], time_str
        ))

def predict_step(global_step):
    #if split == 'test_seen':
    #    test_loader = test_seen_loader
    #else:
    #    raise ValueError
    semip_model.eval()
    labels, scores = [], []
    count = 0
    hit1=0
    hit2=0
    hit5=0
    hit10=0
    with torch.no_grad():
        for context_list,response_list,persona_list,knowledge_list,plabel_list,klabel_list in eval_loader:
            batcher.load(context_list,response_list,persona_list,knowledge_list,plabel_list,klabel_list)
            batch_dict=batcher('p|cr',None,None)
            input_id=batch_dict['input_id']
            segment_id=batch_dict['segment_id']

            assert input_id.dim()==3 and input_id.shape==segment_id.shape
            bs,n_per,seq_len=input_id.shape
            logits = semip_model(input_ids=input_id.view(-1,seq_len),attention_mask=input_id.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id.view(-1,seq_len),return_dict=True)['logits']
            logits = logits.view(bs,n_per,-1)[:,:,1]
            count += bs
            if count % 1000 == 0:
                logger.info("eval finishing {}".format(count))
            bs=len(context_list)
            ref=torch.tensor([persona_list[i].index(plabel_list[i]) for i in range(bs)],dtype=torch.long,device=device)
            hyp=torch.topk(logits,k=10)[1]

            hit1+=(hyp[:,:1]==ref[:,None]).sum(1).sum(0).item()
            hit2+=(hyp[:,:2]==ref[:,None]).sum(1).sum(0).item()
            hit5+=(hyp[:,:5]==ref[:,None]).sum(1).sum(0).item()
            hit10+=(hyp[:,:10]==ref[:,None]).sum(1).sum(0).item()
        
    if not args.predict:
        semip_model.save_pretrained(os.path.join(args.out_dir,'{}step_semip_model'.format(global_step)))
        logger.info("Saved model checkpoint \n")
    
    logger.info("hit at 1 is {:.4f}".format(hit1/count))
    logger.info("hit at 2 is {:.4f}".format(hit2/count))
    logger.info("hit at 5 is {:.4f}".format(hit5/count))
    logger.info("hit at 10 is {:.4f}".format(hit10/count))

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
        #logger.info("test recall f1 result {:.6f}".format(test_result))
            #save_path = '{}-best'.format(checkpoint_prefix)
            #os.makedirs(save_path, exist_ok=True)