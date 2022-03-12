
import argparse
import os
from posix import environ
#os.environ['CUDA_VISIBLE_DEVICES']='1'
import sys
from numpy.lib.arraypad import pad
from tokenizers import InputSequence
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

from transformers.utils import dummy_flax_objects
from transformers.utils.dummy_pt_objects import BertModel
from metric import f1_metric
from batcher import PersonaBatcher
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
parser.add_argument("--predict",type=str2bool,default=False)

#  files
parser.add_argument("--convo_path",type=str,default='/home/futc/persona/convo')
parser.add_argument("--persona_path",type=str,default='/home/futc/persona/history')
parser.add_argument("--knowledge_path",type=str,default='/home/futc/persona/knowledge')
parser.add_argument("--pseudo_path",type=str,default='/home/futc/2021work2/pseudo')
parser.add_argument("--selected_path",type=str,default='')

# model 
parser.add_argument("--vocab",type=str,default='/home/futc/bert-base-uncased')
parser.add_argument("--priork_model",type=str,default='/home/futc/bert-base-uncased') #(K|CP) model 
parser.add_argument("--dualpk_model",type=str,default='/home/futc/bert-base-uncased')# (K|CRP) dual learning model
parser.add_argument("--n_layer",type=int,default=6)

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
parser.add_argument("--n_glue",default=1,type=int,help='how many selected persona to input')

# gpu
parser.add_argument('--gpu_list', type=str, default='4')
parser.add_argument('--gpu_ratio', type=float, default=0.85)
parser.add_argument('--n_device', type=int, default=8)
parser.add_argument('--no_cuda', type=str2bool, default=False)

# setting
parser.add_argument("--train_origin",type=str,default='pseudo')
parser.add_argument("--eval_origin",type=str,default='pseudo')

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


class VAEKDataset(Dataset):
    def __init__(self,convo_path,persona_path,knowledge_path,pseudo_path,selected_path,mode,debug=False) -> None:
        super(VAEKDataset,self).__init__()
        self.examples=[]
        self.selected=[]
        assert mode in ['train','eval']
        self.mode=mode
        for date in os.listdir(convo_path):
            if (date=='2015-05' or date=='2015-06') and mode=='train':
                continue
            if mode=='eval' and date!='2015-05':
                continue
            fconvo=open(os.path.join(convo_path,date),mode='r',encoding='utf-8')
            fper=open(os.path.join(persona_path,date),mode='r',encoding='utf-8')
            fknow=open(os.path.join(knowledge_path,date),mode='r',encoding='utf-8')
            fpseudo=open(os.path.join(pseudo_path,date),mode='r',encoding='utf-8')
            know_dict=json.load(fknow)
            per_dict=json.load(fper)
            for line1,line2 in zip(fconvo.readlines(),fpseudo.readlines()):
                data=json.loads(line1)
                author=data['author'][-1]
                sid=data['sid']
                label=json.loads(line2)

                self.examples.append({
                    'context':data['dialog'][:-1],
                    'response':data['dialog'][-1],
                    'knowledge':know_dict[sid],
                    'persona':per_dict[author],
                    'klabel':label['klabel'],
                    'plabel':label['plabel']
                })
            if debug:
                break
        if mode=='eval' and args.eval_origin=='selected':
            fselected=open(selected_path,mode='r',encoding='utf-8')
            for line in fselected.readlines():
                selectedp=str(line).strip('\n').split('<#p#>')[:args.n_glue]
                if isinstance(selectedp,list):
                    selectedp=' '.join(selectedp) 
                self.selected.append(selectedp)

        if debug and mode == 'eval':
            self.examples=self.examples[:16]
            self.selected=self.selected[:16]


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        example=self.examples[i]
        random.shuffle(example['persona'])
        random.shuffle(example['knowledge'])
        if self.mode=='eval' and args.eval_origin=='selected':
            return example['context'], example['response'],example['persona'], example['knowledge'], self.selected[i], example['klabel']
        else:
            return example['context'], example['response'],example['persona'], example['knowledge'], example['plabel'],example['klabel']

        # note that the context and persona is a list, but knowledge is a string, a single piece of knowledge

    @staticmethod
    def collate_fn(batch):
        context_list = [item[0] for item in batch]
        response_list= [item[1] for item in batch]
        persona_list = [item[2] for item in batch]
        knowledge_list = [item[3] for item in batch]
        plabel_list= [item[4] for item in batch]
        klabel_list= [item[5] for item in batch]
        return context_list,response_list,persona_list,knowledge_list,plabel_list,klabel_list


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

train_dataset=VAEKDataset(args.convo_path,args.persona_path,args.knowledge_path,args.pseudo_path,args.selected_path,'train',args.debug)
eval_dataset=VAEKDataset(args.convo_path,args.persona_path,args.knowledge_path,args.pseudo_path,args.selected_path,'eval',args.debug)
train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=VAEKDataset.collate_fn)
eval_loader=DataLoader(eval_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=VAEKDataset.collate_fn)
train_loader=itertools.cycle(train_loader)
logger.info("train examples {}".format(len(train_dataset)))
logger.info("eval examples {}".format(len(eval_dataset)))

time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.info("Create training dataset end... | %s " % time_str)
tokenizer=BertTokenizer.from_pretrained(args.vocab)
tokenizer.add_special_tokens({'pad_token':'[PAD]','sep_token':'[SEP]'})
batcher = PersonaBatcher(device,tokenizer,args.n_knowledge,args.n_persona,args.max_context_length,args.max_response_length,args.max_knowledge_length,args.max_persona_length*args.n_glue)

configuration=BertConfig(num_hidden_layers=args.n_layer)
dualpk_model=BertForSequenceClassification.from_pretrained(args.dualpk_model,config=configuration)
dualpk_model.resize_token_embeddings(len(tokenizer))

priork_model=BertForSequenceClassification.from_pretrained(args.priork_model,config=configuration)
priork_model.resize_token_embeddings(len(tokenizer))

dualpk_model.to(device)
priork_model.to(device)

no_decay = ["bias", "LayerNorm.weight"]
dualpk_parameters = [
    {
        "params": [p for n, p in dualpk_model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in dualpk_model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
priork_parameters = [
    {
        "params": [p for n, p in priork_model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in priork_model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

dualpk_optimizer = AdamW(dualpk_parameters, lr=args.lr, eps=args.adam_epsilon)
priork_optimizer = AdamW(priork_parameters, lr=args.lr, eps=args.adam_epsilon)
total_steps = args.num_epochs * (len(train_dataset) / (args.batch_size * args.accum_step))
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
if args.schedule == 'linear':
    dualpk_scheduler = get_linear_schedule_with_warmup(dualpk_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    priork_scheduler = get_linear_schedule_with_warmup(priork_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
elif args.schedule == 'cosine':
    dualpk_scheduler = get_cosine_schedule_with_warmup(dualpk_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    priork_scheduler = get_cosine_schedule_with_warmup(priork_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)


def train_step(global_step):
    klloss_total = 0.0
    dualpk_model.eval()
    priork_model.train()
    for _ in range(args.accum_step):
        context_list,response_list,persona_list,knowledge_list, inputp_list,klabel_list= next(train_loader)
        batcher.load(context_list,response_list,persona_list,knowledge_list,None,None)
        with torch.no_grad():
            batch_dict=batcher('k|crp',None,inputp_list)
            input_id=batch_dict['input_id']
            segment_id=batch_dict['segment_id']
            assert input_id.dim()==3 and input_id.shape==segment_id.shape
            bs,n_know,seq_len=input_id.shape
            dual_klogits=dualpk_model(input_ids=input_id.view(-1,seq_len),attention_mask=input_id.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id.view(-1,seq_len),return_dict=True)['logits']
            dual_klogits=dual_klogits.view(bs,n_know,-1)[:,:,1]

        batch_dict=batcher('k|cp',None,inputp_list)
        input_id=batch_dict['input_id']
        segment_id=batch_dict['segment_id']
        assert input_id.dim()==3 and input_id.shape==segment_id.shape
        bs,n_know,seq_len=input_id.shape
        prior_klogits=priork_model(input_ids=input_id.view(-1,seq_len),attention_mask=input_id.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id.view(-1,seq_len),return_dict=True)['logits']
        prior_klogits=prior_klogits.view(bs,n_know,-1)[:,:,1]
        
        klloss=F.kl_div(torch.log_softmax(prior_klogits,dim=1),torch.softmax(dual_klogits,dim=1))
        klloss_total+=klloss.item()
        klloss=klloss/args.accum_step
        klloss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in priork_model.parameters() if p.requires_grad], args.clip)
    if grad_norm >= 1e2:
        logger.info('WARNING : Exploding Gradients {:.2f}'.format(grad_norm))
    priork_optimizer.step()
    priork_scheduler.step()
    priork_optimizer.zero_grad()

    if global_step % args.print_every == 0 and global_step != 0:
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info("Step: %d \t| ks_loss: %.3f \t| lr: %.8f \t| %s" % (
            global_step, klloss_total, priork_scheduler.get_lr()[0], time_str
        ))

def predict_step(global_step):
    #if split == 'test_seen':
    #    test_loader = test_seen_loader
    #else:
    #    raise ValueError
    hypothesis=[]
    priork_model.eval()
    count = 0
    hit1=0
    hit2=0
    hit5=0
    hit10=0
    with torch.no_grad():
        for context_list,response_list,persona_list,knowledge_list,inputp_list,klabel_list in eval_loader:
            batcher.load(context_list,response_list,persona_list,knowledge_list,None,None)
            batch_dict=batcher('k|cp',None,inputp_list)
            input_id=batch_dict['input_id']
            segment_id=batch_dict['segment_id']
            assert input_id.dim()==3 and input_id.shape==segment_id.shape
            bs,n_know,seq_len=input_id.shape
            logits = priork_model(input_ids=input_id.view(-1,seq_len),attention_mask=input_id.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id.view(-1,seq_len),return_dict=True)['logits']
            logits = logits.view(bs,n_know,-1)[:,:,1]
            count += bs
            if count % 1000 == 0:
                logger.info("eval finishing {}".format(count))
            ref=torch.tensor([knowledge_list[i].index(klabel_list[i]) for i in range(bs)],dtype=torch.long,device=device)
            # hyp=torch.max(logits,dim=1)[1]
            # hit1+=torch.sum(hyp==ref,dim=0).item()
            hyp=torch.topk(logits,k=10)[1]
            hit1+=(hyp[:,:1]==ref[:,None]).sum(1).sum(0).item()
            hit2+=(hyp[:,:2]==ref[:,None]).sum(1).sum(0).item()
            hit5+=(hyp[:,:5]==ref[:,None]).sum(1).sum(0).item()
            hit10+=(hyp[:,:10]==ref[:,None]).sum(1).sum(0).item()
            try:
                hyp=hyp.detach().cpu().tolist()
                for i in range(bs):
                    hypothesis.append('<#k#>'.join([knowledge_list[i][min(j,len(knowledge_list[i])-1)]for j in hyp[i]]))
            except:
                logger.info("error when decoding hypothesis")
    if not args.predict:
        priork_model.save_pretrained(os.path.join(args.out_dir,'{}step_priork_model'.format(global_step)))
    logger.info("Saved model checkpoint \n")
    logger.info("hit at 1 is {:.4f}".format(hit1/count))
    logger.info("hit at 2 is {:.4f}".format(hit2/count))
    logger.info("hit at 5 is {:.4f}".format(hit5/count))
    logger.info("hit at 10 is {:.4f}".format(hit10/count))

    if args.eval_origin=='selected':
        with open(os.path.join(args.out_dir,'{}step_knowledge'.format(global_step)),mode='w',encoding='utf-8') as f:
            json.dump(hypothesis,f)


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