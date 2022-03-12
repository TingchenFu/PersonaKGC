import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES']='1'
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
from metric import f1_metric
from transformers import BertForSequenceClassification
from transformers import BertModel
from transformers import BertConfig
from dataset import PersonaDataset
from batcher import PersonaBatcher

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
parser.add_argument("--debug",default=True,help='debug mode, using small dataset',type=str2bool)
parser.add_argument('--predict',type=str2bool,default=False)
parser.add_argument("--task",type=str,default='priorp') # semip priorp priork

#  files
parser.add_argument("--convo_path",type=str,default='/home/futc/persona/convo')
parser.add_argument("--persona_path",type=str,default='/home/futc/persona/history')
parser.add_argument("--knowledge_path",type=str,default='/home/futc/persona/knowledge')
parser.add_argument("--pseudo_path",type=str,default='/home/futc/2021work2/pseudo')

# model 
parser.add_argument("--vocab",type=str,default='/home/futc/bert-base-uncased')
parser.add_argument("--priork_model",type=str,default='/home/futc/bert-base-uncased')
parser.add_argument("--priorp_model",type=str,default='/home/futc/bert-base-uncased')
parser.add_argument("--semip_model",type=str,default='/home/futc/bert-base-uncased')
parser.add_argument("--priork_ablationp_model",type=str,default='/home/futc/bert-base-uncased')
parser.add_argument("--postk_ablationp_model",type=str,default='/home/futc/bert-base-uncased')


# parser.add_argument("--count_path",type=str,default='/home/futc/2021work2/knowledge_count.json')
# parser.add_argument("--label_path",type=str,default='/home/futc/2021work2/label.json')
# training scheme
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--eval_batch_size', type=int, default=4)
parser.add_argument('--num_steps', type=int, default=1000000)
parser.add_argument('--accum_step', type=int, default=16)
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

# length
parser.add_argument("--max_context_length",type=int,default=64)
parser.add_argument("--max_persona_length",type=int,default=64)
parser.add_argument("--max_response_length",type=int,default=64)
parser.add_argument("--max_knowledge_length",type=int,default=64)
parser.add_argument("--n_knowledge",type=int,default=32)
parser.add_argument("--n_persona",type=int,default=32)
# architecture
parser.add_argument("--loss",type=str,default='ce')
parser.add_argument("--n_layer",type=int,default=6)

# gpu
parser.add_argument('--gpu_list', type=str, default='4')
parser.add_argument('--gpu_ratio', type=float, default=0.85)
parser.add_argument('--n_device', type=int, default=8)
parser.add_argument('--no_cuda', type=str2bool, default=False)


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
assert args.task in ['semip','priorp','priork','priork_ablationp','postk_ablationp']
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



# Build dataset
time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.info("Create training dataset begin... | %s " % time_str)

train_dataset=PersonaDataset(args.convo_path,args.persona_path,args.knowledge_path,args.pseudo_path,mode='train',debug=args.debug)
logger.info("train dataset:{}".format(len(train_dataset)))
eval_dataset=PersonaDataset(args.convo_path,args.persona_path,args.knowledge_path,args.pseudo_path,mode='eval',debug=args.debug)
logger.info("eval dataset:{}".format(len(eval_dataset)))
train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=PersonaDataset.collate_fn)
eval_loader=DataLoader(eval_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=PersonaDataset.collate_fn)
train_loader=itertools.cycle(train_loader)

time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.info("Create training dataset end... | %s " % time_str)
tokenizer=BertTokenizer.from_pretrained(args.vocab)
tokenizer.add_special_tokens({'pad_token':'[PAD]','sep_token':'[SEP]'})
batcher = PersonaBatcher(device, tokenizer, args.n_knowledge, args.n_persona, args.max_context_length, args.max_response_length, args.max_knowledge_length,args.max_persona_length)

configuration=BertConfig(num_hidden_layers=args.n_layer)

if args.task=='semip':
    model=BertForSequenceClassification.from_pretrained(args.semip_model,config=configuration)
elif args.task=='priorp':
    model=BertForSequenceClassification.from_pretrained(args.priorp_model,config=configuration)
elif args.task=='priork':
    model=BertForSequenceClassification.from_pretrained(args.priork_model,config=configuration)
# model architecture
elif args.task=='priork_ablationp':
    model=BertForSequenceClassification.from_pretrained(args.priork_ablationp_model,config=configuration)
elif args.task=='postk_ablationp':
    model=BertForSequenceClassification.from_pretrained(args.postk_ablationp_model,config=configuration)

model.resize_token_embeddings(len(tokenizer))


model.to(device)

no_decay = ["bias", "LayerNorm.weight"]
model_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]


model_optimizer = AdamW(model_parameters, lr=args.lr, eps=args.adam_epsilon)
#dualpk_optimizer = AdamW(dualpk_parameters, lr=args.lr, eps=args.adam_epsilon)
total_steps = args.num_epochs * (len(train_dataset) / (args.batch_size * args.accum_step))
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
if args.schedule == 'linear':
    model_scheduler = get_linear_schedule_with_warmup(model_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
elif args.schedule == 'cosine':
    model_scheduler = get_cosine_schedule_with_warmup(model_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)


def train_step(global_step):
    ks_loss_total = 0.0
    for _ in range(args.accum_step):
        context_list,response_list,persona_list,knowledge_list, plabel_list,klabel_list = next(train_loader)
        #The dual learning part
        bs=len(context_list)
        model.train()
        batcher.load(context_list,response_list,persona_list,knowledge_list,plabel_list,klabel_list)
        if args.task=='semip':
            batch_dict=batcher('p|cr',None,None)
            target=torch.tensor([persona_list[i].index(plabel_list[i]) for i in range(bs)],dtype=torch.long,device=device)
        elif args.task=='priorp':
            batch_dict=batcher('p|c',None,None)
            target=torch.tensor([persona_list[i].index(plabel_list[i]) for i in range(bs)],dtype=torch.long,device=device)
        elif args.task=='priork':
            batch_dict=batcher('k|cp',None,plabel_list)
            target=torch.tensor([knowledge_list[i].index(klabel_list[i]) for i in range(bs)],dtype=torch.long,device=device)
        elif args.task=='postk_ablationp':
            batch_dict=batcher('k|cr',None,None)
            target=torch.tensor([knowledge_list[i].index(klabel_list[i]) for i in range(bs)],dtype=torch.long,device=device)
        elif args.task=='priork_ablationp':
            batch_dict=batcher('k|c',None,None)
            target=torch.tensor([knowledge_list[i].index(klabel_list[i]) for i in range(bs)],dtype=torch.long,device=device)

        input_id=batch_dict['input_id']
        segment_id=batch_dict['segment_id']
        assert input_id.dim()==3 and input_id.shape==segment_id.shape
        bs,n_know,seq_len=input_id.shape
        #(bs*n_know,2)
        # if args.task=='priorp' or args.task=='priork':
        #     input_id1,input_id2,input_id3,input_id4,input_id5,input_id6,input_id7,input_id8=torch.chunk(input_id,8,dim=1)
        #     segment_id1,segment_id2,segment_id3,segment_id4,segment_id5,segment_id6,segment_id7,segment_id8=torch.chunk(segment_id,8,dim=1)
        #     logits1=model(input_ids=input_id1.view(-1,seq_len),attention_mask=input_id1.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id1.view(-1,seq_len),return_dict=True)['logits']
        #     logits2=model(input_ids=input_id2.view(-1,seq_len),attention_mask=input_id2.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id2.view(-1,seq_len),return_dict=True)['logits']
        #     logits3=model(input_ids=input_id3.view(-1,seq_len),attention_mask=input_id3.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id3.view(-1,seq_len),return_dict=True)['logits']
        #     logits4=model(input_ids=input_id4.view(-1,seq_len),attention_mask=input_id4.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id4.view(-1,seq_len),return_dict=True)['logits']
        #     logits5=model(input_ids=input_id5.view(-1,seq_len),attention_mask=input_id5.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id5.view(-1,seq_len),return_dict=True)['logits']
        #     logits6=model(input_ids=input_id6.view(-1,seq_len),attention_mask=input_id6.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id6.view(-1,seq_len),return_dict=True)['logits']
        #     logits7=model(input_ids=input_id7.view(-1,seq_len),attention_mask=input_id7.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id7.view(-1,seq_len),return_dict=True)['logits']
        #     logits8=model(input_ids=input_id8.view(-1,seq_len),attention_mask=input_id8.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id8.view(-1,seq_len),return_dict=True)['logits']
        #     logits=torch.torch.cat([logits1,logits2,logits3,logits4,logits5,logits6,logits7,logits8],dim=1)
        # else:
        logits=model(input_ids=input_id.view(-1,seq_len),attention_mask=input_id.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id.view(-1,seq_len),return_dict=True)['logits']        
        #(bs,n_know)
        logits=logits.view(bs,n_know,-1)[:,:,1]
        if args.loss=='ce':
            kloss=F.cross_entropy(logits,target)
        elif args.loss=='mm':
            kloss=F.multi_margin_loss(logits,target)
        ks_loss_total+=kloss.item()
        kloss=kloss/args.accum_step
        kloss.backward()

    grad_norm1 = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip)
    grad_norm2 = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip)
    if grad_norm1 >= 1e2 or grad_norm2 >1e2:
        logger.info('WARNING : Exploding Gradients {:.2f} {:.2f}'.format(grad_norm1,grad_norm2))
    model_optimizer.step()
    model_scheduler.step()
    model_optimizer.zero_grad()

    if global_step % args.print_every == 0 and global_step != 0:
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info("Step: %d \t| ks_loss: %.3f \t| lr: %.8f \t| %s" % (
            global_step, ks_loss_total, model_scheduler.get_lr()[0], time_str
        ))
        # sys.stdout.flush()

def predict_step(global_step):
    model.eval()
    hit1 = 0
    count=0
    if not args.predict:
        model.save_pretrained(os.path.join(args.out_dir,'{}step_{}'.format(global_step,args.task)))
    
    with torch.no_grad():
        for context_list,response_list,persona_list,knowledge_list, plabel_list,klabel_list in eval_loader:
            bs=len(context_list)
            batcher.load(context_list,response_list,persona_list,knowledge_list,plabel_list,klabel_list)
            if args.task=='semip':
                batch_dict=batcher('p|cr',None,None)
                target=torch.tensor([persona_list[i].index(plabel_list[i]) for i in range(bs)],dtype=torch.long,device=device)
            elif args.task=='priorp':
                batch_dict=batcher('p|c',None,None)
                target=torch.tensor([persona_list[i].index(plabel_list[i]) for i in range(bs)],dtype=torch.long,device=device)
            elif args.task=='priork':
                batch_dict=batcher('k|cp',None,plabel_list)
                target=torch.tensor([knowledge_list[i].index(klabel_list[i]) for i in range(bs)],dtype=torch.long,device=device)
            elif args.task=='postk_ablationp':
                batch_dict=batcher('k|cr',None,None)
                target=torch.tensor([knowledge_list[i].index(klabel_list[i]) for i in range(bs)],dtype=torch.long,device=device)
            elif args.task=='priork_ablationp':
                batch_dict=batcher('k|c',None,None)
                target=torch.tensor([knowledge_list[i].index(klabel_list[i]) for i in range(bs)],dtype=torch.long,device=device)            
            input_id=batch_dict['input_id']
            segment_id=batch_dict['segment_id']
            assert input_id.dim()==3 and input_id.shape==segment_id.shape
            bs,n_know,seq_len=input_id.shape            
            logits=model(input_ids=input_id.view(-1,seq_len),attention_mask=input_id.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id.view(-1,seq_len),return_dict=True)['logits']
            
            logits = logits.view(bs,n_know,-1)[:,:,1]
            count += bs
            if count % 1000 == 0:
                logger.info("eval finishing {}".format(count))
            bs=len(context_list)
            hyp=torch.max(logits,dim=1)[1]
            hit1+=torch.sum(hyp==target,dim=0).item()
    
    if not args.predict:
        logger.info("Saved model checkpoint \n")
    if count!=0:
        logger.info("hit at 1 is {:.4f}".format(hit1/count))

best_f1 = -1.
if args.predict:
    predict_step(0)
    exit()
for i in range(args.num_steps):
    torch.cuda.empty_cache()
    train_step(i + 1)
    if (i + 1) % args.valid_every == 0:
        predict_step(i+1)