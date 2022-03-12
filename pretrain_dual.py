
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

#  files
parser.add_argument("--convo_path",type=str,default='/home/futc/persona/convo')
parser.add_argument("--persona_path",type=str,default='/home/futc/persona/history')
parser.add_argument("--knowledge_path",type=str,default='/home/futc/persona/knowledge')
parser.add_argument("--pseudo_path",type=str,default='/home/futc/2021work2/pseudo')

# model 
parser.add_argument("--vocab",type=str,default='/home/futc/bert-base-uncased')
parser.add_argument("--dualkp_model",type=str,default='/home/futc/bert-base-uncased')# (P|CRK) dual learning model
parser.add_argument("--dualpk_model",type=str,default='/home/futc/bert-base-uncased')# (K|CRP) dual learning model

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
logger.addHandler(logging.FileHandler(os.path.join(args.out_dir, "log"), 'w'))
logger.info("\nParameters:")
for attr, value in sorted(vars(args).items()):
    logger.info("{}={}".format(attr.upper(), value))


class PersonaDataset(Dataset):
    def __init__(self,convo_path,persona_path,knowledge_path,pseudo_path,mode,n_knowledge,n_persona,debug=False) -> None:
        super(PersonaDataset,self).__init__()
        self.examples=[]
        self.n_knowledge=n_knowledge
        self.n_persona=n_persona
        assert mode in ['train','eval']
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
            
                # persona=per_dict[author].copy()
                # persona.pop(persona.index(label['plabel']))
                # persona=persona[:self.n_persona-1]
                # persona.append(label['plabel'])
                # random.shuffle(persona)

                # knowledge=know_dict[sid].copy()
                # knowledge.pop(knowledge.index(label['klabel']))
                # knowledge=knowledge[:self.n_knowledge-1]
                # knowledge.append(label['klabel'])
                # random.shuffle(knowledge)

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
        if debug and mode == 'eval':
            self.examples=self.examples[:16]
        logger.info("{} examples {}".format(mode,len(self.examples)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        example=self.examples[i]
        random.shuffle(example['persona'])
        random.shuffle(example['knowledge'])

        # note that the context and persona is a list, but knowledge is a string, a single piece of knowledge
        return example['context'], example['response'],example['persona'], example['knowledge'], example['plabel'],example['klabel']

    @staticmethod
    def collate_fn(batch):
        context_list = [item[0] for item in batch]
        response_list= [item[1] for item in batch]
        persona_list = [item[2] for item in batch]
        knowledge_list = [item[3] for item in batch]
        plabel_list= [item[4] for item in batch]
        klabel_list= [item[5] for item in batch]
        return context_list,response_list,persona_list,knowledge_list,plabel_list,klabel_list

class PersonaBatcher(object):
    def __init__(self, device, tokenizer, n_knowledge, n_persona, max_context_length,max_response_length,max_knowledge_length,max_persona_length):
        self.device=device
        self.tokenizer=tokenizer
        self.n_knowledge=n_knowledge
        self.n_persona=n_persona
        self.max_context_length=max_context_length
        self.max_response_length=max_response_length
        self.max_knowledge_length=max_knowledge_length
        self.max_persona_length=max_persona_length
        #assert glue in ['p|crk','k|crp','p|c','k|cp','p|cr']

    def load(self, context_list, response_list, persona_list, knowledge_list, plabel_list, klabel_list):
        assert len(context_list)==len(response_list)==len(persona_list)==len(knowledge_list)==len(plabel_list)==len(klabel_list)
        bs=len(context_list)
        self.context_id_list=[self.tokenizer.encode(' '.join(context_list[i]),add_special_tokens=False)[:self.max_context_length] for i in range(bs)]
        self.response_id_list=[self.tokenizer.encode(response_list[i],add_special_tokens=False)[:self.max_response_length] for i in range(bs)]
        self.persona_id_list=[]
        self.knowledge_id_list=[]
        longest_persona=0
        longest_knowledge=0
        for i in range(bs):
            persona=persona_list[i]
            knowledge=knowledge_list[i]
            # WARNING: We could not use random shuffle at here 
            # it is a list, every element is a single piece of persona id
            persona_id=[self.tokenizer.encode(p,add_special_tokens=False)[:self.max_persona_length] for p in persona[:self.n_persona] ]
            longest_persona=max(max([len(p) for p in persona_id]),longest_persona)
            knowledge_id=[self.tokenizer.encode(k,add_special_tokens=False)[:self.max_knowledge_length] for k in knowledge[:self.n_knowledge] ]
            longest_knowledge=max(max([len(k) for k in knowledge_id]),longest_knowledge)
            self.persona_id_list.append(persona_id)
            self.knowledge_id_list.append(knowledge_id)
        
        # padding
        longest_context=max([len(self.context_id_list[i]) for i in range(bs)])
        longest_response=max([len(self.response_id_list[i]) for i in range(bs)])
        for i in range(bs):
            padding_length=longest_context-len(self.context_id_list[i])
            self.context_id_list[i].extend([self.tokenizer.pad_token_id]*padding_length)
            padding_length=longest_response-len(self.response_id_list[i])
            self.response_id_list[i].extend([self.tokenizer.pad_token_id]*padding_length)
            for k in self.knowledge_id_list[i]:
                k.extend([self.tokenizer.pad_token_id]*(longest_knowledge-len(k)))
            while len(self.knowledge_id_list[i]) < self.n_knowledge:
                self.knowledge_id_list[i].append([self.tokenizer.pad_token_id]*longest_knowledge)
            for p in self.persona_id_list[i]:
                p.extend([self.tokenizer.pad_token_id]*(longest_persona-len(p)))
            while len(self.persona_id_list[i]) < self.n_persona:
                self.persona_id_list[i].append([self.tokenizer.pad_token_id]*longest_persona)
        
        self.bs=bs
        self.longest_context=longest_context
        self.longest_response=longest_response
        self.longest_persona=longest_persona
        self.longest_knowledge=longest_knowledge


    def __call__(self,glue,golden_knowledge_list=None,golden_persona_list=None):
        assert glue in ['p|crk','k|crp','k|cp','p|cr','p|c']
        batch_input_id=[]
        batch_segment_id=[]
        if glue == 'p|crk':
            assert len(golden_knowledge_list)==self.bs
            bs=len(golden_knowledge_list)
            if isinstance(golden_knowledge_list[0],str):
                golden_knowledge_id_list=[self.tokenizer.encode(golden_knowledge_list[i],add_special_tokens=False)[:self.max_knowledge_length] for i in range(bs)]
                longest_goldenk=max([len(golden_knowledge_id_list[i]) for i in range(bs)])
                for golden_knowedge_id in golden_knowledge_id_list:
                    golden_knowedge_id.extend([self.tokenizer.pad_token_id]*(longest_goldenk-len(golden_knowedge_id)))
            else:
                raise NotImplementedError
            for i in range(bs):
                input_id=[]
                segment_id=[]
                for j in range(self.n_persona):
                    input_id.append([self.tokenizer.cls_token_id]+self.context_id_list[i] +[self.tokenizer.sep_token_id] + self.response_id_list[i] +[self.tokenizer.sep_token_id] + golden_knowledge_id_list[i]+[self.tokenizer.sep_token_id]+self.persona_id_list[i][j]+[self.tokenizer.sep_token_id])
                    segment_id.append([0]*(1 +self.longest_context    +1         + self.longest_response    +1         + longest_goldenk             +1 )+[1]*(self.longest_persona+1))
                batch_input_id.append(input_id)
                batch_segment_id.append(segment_id)

        elif glue == 'k|crp':
            assert len(golden_persona_list)==self.bs
            bs=len(golden_persona_list)
            if isinstance(golden_persona_list[0],str):
                golden_persona_id_list=[self.tokenizer.encode(golden_persona_list[i],add_special_tokens=False)[:self.max_persona_length] for i in range(bs)]
                longest_goldenp=max([len(golden_persona_id_list[i]) for i in range(bs)])
                for golden_persona_id in golden_persona_id_list:
                    golden_persona_id.extend([self.tokenizer.pad_token_id]*(longest_goldenp-len(golden_persona_id)))
            else:
                golden_persona_id_list=golden_persona_list
            for i in range(bs):
                input_id=[]
                segment_id=[]
                for j in range(self.n_knowledge):
                    input_id.append([self.tokenizer.cls_token_id] + self.context_id_list[i] +[self.tokenizer.sep_token_id] + self.response_id_list[i] +[self.tokenizer.sep_token_id]+ golden_persona_id_list[i]  +[self.tokenizer.sep_token_id]+ self.knowledge_id_list[i][j]+[self.tokenizer.sep_token_id])
                    segment_id.append([0]*(self.longest_context+self.longest_response+longest_goldenp+4)+ [1]*(1+self.longest_knowledge))
                batch_input_id.append(input_id)
                batch_segment_id.append(segment_id)
        elif glue=='k|cp':
            assert len(golden_persona_list)==self.bs
            bs=len(golden_persona_list)
            if isinstance(golden_persona_list[0],str):
                golden_persona_id_list=[self.tokenizer.encode(golden_persona_list[i],add_special_tokens=False)[:self.max_persona_length] for i in range(bs)]
                longest_goldenp=max([len(golden_persona_id_list[i]) for i in range(bs)])
                for golden_persona_id in golden_persona_id_list:
                    golden_persona_id.extend([self.tokenizer.pad_token_id]*(longest_goldenp-len(golden_persona_id)))
            else:
                raise NotImplementedError
            for i in range(bs):
                input_id=[]
                segment_id=[]
                for j in range(self.n_knowledge):
                    input_id.append([self.tokenizer.cls_token_id]+ self.context_id_list[i] +[self.tokenizer.sep_token_id]+ golden_persona_id_list[i] +[self.tokenizer.sep_token_id]+self.knowledge_id_list[i][j]+[self.tokenizer.pad_token_id])
                    segment_id.append([0]*(self.longest_context+longest_goldenp+3)+ [1]*(self.longest_knowledge+1))
                batch_input_id.append(input_id)
                batch_segment_id.append(segment_id)
        elif glue=='p|cr':
            bs=self.bs
            for i in range(bs):
                input_id=[]
                segment_id=[]
                for j in range(self.n_persona):
                    input_id.append([self.tokenizer.cls_token_id]+ self.context_id_list[i] +[self.tokenizer.sep_token_id]+ self.response_id_list[i]+ [self.tokenizer.sep_token_id] +self.persona_id_list[i][j]+[self.tokenizer.pad_token_id])
                    segment_id.append([0]*(self.longest_context+self.longest_response+3)+ [1]*(self.longest_persona+1))
                batch_input_id.append(input_id)
                batch_segment_id.append(segment_id)
        elif glue=='p|c':
            bs=self.bs
            for i in range(bs):
                input_id=[]
                segment_id=[]
                for j in range(self.n_persona):
                    input_id.append([self.tokenizer.cls_token_id]+ self.context_id_list[i]+[self.tokenizer.sep_token_id]+self.persona_id_list[i][j]+[self.tokenizer.pad_token_id])
                    segment_id.append([0]*(self.longest_context+2)+ [1]*(self.longest_persona+1))
                batch_input_id.append(input_id)
                batch_segment_id.append(segment_id)
        
        batch_input_id=torch.tensor(batch_input_id,dtype=torch.long,device=self.device)
        batch_segment_id=torch.tensor(batch_segment_id,dtype=torch.long,device=self.device)

        return {
            'input_id':batch_input_id,
            'segment_id':batch_segment_id
        }



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

train_dataset=PersonaDataset(args.convo_path,args.persona_path,args.knowledge_path,args.pseudo_path,mode='train',n_knowledge=args.n_knowledge,n_persona=args.n_persona,debug=args.debug)
eval_dataset=PersonaDataset(args.convo_path,args.persona_path,args.knowledge_path,args.pseudo_path,mode='eval',n_knowledge=args.n_knowledge,n_persona=args.n_persona,debug=args.debug)
train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=PersonaDataset.collate_fn)
eval_loader=DataLoader(eval_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=PersonaDataset.collate_fn)
train_loader=itertools.cycle(train_loader)

time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.info("Create training dataset end... | %s " % time_str)
tokenizer=BertTokenizer.from_pretrained(args.vocab)
tokenizer.add_special_tokens({'pad_token':'[PAD]','sep_token':'[SEP]'})
batcher = PersonaBatcher(device, tokenizer, args.n_knowledge, args.n_persona, args.max_context_length, args.max_response_length, args.max_knowledge_length,args.max_persona_length)

configuration=BertConfig(num_hidden_layers=args.n_layer)
dualkp_model=BertForSequenceClassification.from_pretrained(args.dualkp_model,config=configuration)
dualkp_model.resize_token_embeddings(len(tokenizer))
dualpk_model=BertForSequenceClassification.from_pretrained(args.dualpk_model,config=configuration)
dualpk_model.resize_token_embeddings(len(tokenizer))


dualkp_model.to(device)
dualpk_model.to(device)


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

dualkp_optimizer = AdamW(dualkp_parameters, lr=args.lr, eps=args.adam_epsilon)
dualpk_optimizer = AdamW(dualpk_parameters, lr=args.lr, eps=args.adam_epsilon)
total_steps = args.num_epochs * (len(train_dataset) / (args.batch_size * args.accum_step))
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
if args.schedule == 'linear':
    dualkp_scheduler = get_linear_schedule_with_warmup(dualkp_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    dualpk_scheduler = get_linear_schedule_with_warmup(dualpk_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
elif args.schedule == 'cosine':
    dualkp_scheduler = get_cosine_schedule_with_warmup(dualkp_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    dualpk_scheduler= get_cosine_schedule_with_warmup(dualpk_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)


def train_step(global_step):
    ks_loss_total = 0.0
    for _ in range(args.accum_step):
        context_list, response_list, persona_list, knowledge_list, plabel_list,klabel_list = next(train_loader)
        #The dual learning part
        dualpk_model.train()
        dualkp_model.train()
        batcher.load(context_list,response_list,persona_list,knowledge_list,plabel_list,klabel_list)
        
        batch_dict= batcher('k|crp',None,plabel_list)
        input_id=batch_dict['input_id']
        segment_id=batch_dict['segment_id']
        assert input_id.dim()==3 and input_id.shape==segment_id.shape
        bs,n_know,seq_len=input_id.shape
        #(bs*n_know,2)
        dual_klogits=dualpk_model(input_ids=input_id.view(-1,seq_len),attention_mask=input_id.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id.view(-1,seq_len),return_dict=True)['logits']
        #(bs,n_know)
        dual_klogits=dual_klogits.view(bs,n_know,-1)[:,:,1]
        #(bs)
        targetk=torch.tensor([knowledge_list[i].index(klabel_list[i]) for i in range(bs)],dtype=torch.long,device=dual_klogits.device)
        if args.loss=='ce':
            kloss=F.cross_entropy(dual_klogits,targetk)
        elif args.loss=='mm':
            kloss=F.multi_margin_loss(dual_klogits,targetk)
        ks_loss_total+=kloss.item()
        kloss=kloss/args.accum_step
        kloss.backward()

        batch_dict=batcher('p|crk',klabel_list,None)
        input_id=batch_dict['input_id']
        segment_id=batch_dict['segment_id']
        assert input_id.dim()==3 and input_id.shape==segment_id.shape
        bs,n_per,seq_len=input_id.shape
        dual_plogits=dualkp_model(input_ids=input_id.view(-1,seq_len),attention_mask=input_id.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id.view(-1,seq_len),return_dict=True)['logits']
        dual_plogits=dual_plogits.view(bs,n_per,-1)[:,:,1]
        targetp=torch.tensor([persona_list[i].index(plabel_list[i]) for i in range(bs)],dtype=torch.long,device=device)
        if args.loss=='ce':
            ploss=F.cross_entropy(dual_plogits,targetp)
        elif args.loss=='mm':
            ploss=F.multi_margin_loss(dual_plogits,targetp)
        ks_loss_total+=ploss.item()
        ploss=ploss/args.accum_step
        ploss.backward()

    grad_norm1 = torch.nn.utils.clip_grad_norm_([p for p in dualkp_model.parameters() if p.requires_grad], args.clip)
    grad_norm2 = torch.nn.utils.clip_grad_norm_([p for p in dualpk_model.parameters() if p.requires_grad], args.clip)
    if grad_norm1 >= 1e2 or grad_norm2 >1e2:
        logger.info('WARNING : Exploding Gradients {:.2f} {:.2f}'.format(grad_norm1,grad_norm2))
    dualkp_optimizer.step()
    dualkp_scheduler.step()
    dualkp_optimizer.zero_grad()

    dualpk_optimizer.step()
    dualpk_scheduler.step()
    dualpk_optimizer.zero_grad()

    if global_step % args.print_every == 0 and global_step != 0:
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info("Step: %d \t| ks_loss: %.3f \t| lr: %.8f \t| %s" % (
            global_step, ks_loss_total, dualpk_scheduler.get_lr()[0], time_str
        ))
        # sys.stdout.flush()

def predict_step(global_step):
    #if split == 'test_seen':
    #    test_loader = test_seen_loader
    #else:
    #    raise ValueError
    #dualkp_model.eval()
    dualpk_model.eval()
    all=0
    hit1=0
    count=0
    with torch.no_grad():
        for context_list,response_list,persona_list,knowledge_list, plabel_list,klabel_list in eval_loader:
            batcher.load(context_list,response_list,persona_list,knowledge_list,plabel_list,klabel_list)
            batch_dict=batcher('k|crp',None,plabel_list)
            input_id=batch_dict['input_id']
            segment_id=batch_dict['segment_id']

            assert input_id.dim()==3 and input_id.shape==segment_id.shape
            bs,n_know,seq_len=input_id.shape
            logits = dualpk_model(input_ids=input_id.view(-1,seq_len),attention_mask=input_id.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id.view(-1,seq_len),return_dict=True)['logits']
            logits = logits.view(bs,n_know,-1)[:,:,1]
            count += 1
            if count % 1000 == 0:
                logger.info("eval finishing {}".format(count))
            bs=len(context_list)
            ref=torch.tensor([knowledge_list[i].index(klabel_list[i]) for i in range(bs)],dtype=torch.long,device=device)
            hyp=torch.max(logits,dim=1)[1]
            all+=len(ref)
            hit1+=torch.sum(hyp==ref,dim=0).item()
        
        logger.info("from p infer k, the hit at 1 is {:.4f}".format(hit1/all))

    all=0
    hit1=0
    count=0
    with torch.no_grad():
        for context_list,response_list,persona_list,knowledge_list, plabel_list,klabel_list in eval_loader:
            batcher.load(context_list,response_list,persona_list,knowledge_list,plabel_list,klabel_list)
            batch_dict=batcher('p|crk',klabel_list,None)
            input_id=batch_dict['input_id']
            segment_id=batch_dict['segment_id']

            assert input_id.dim()==3 and input_id.shape==segment_id.shape
            bs,n_per,seq_len=input_id.shape
            logits = dualkp_model(input_ids=input_id.view(-1,seq_len),attention_mask=input_id.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id.view(-1,seq_len),return_dict=True)['logits']
            logits = logits.view(bs,n_per,-1)[:,:,1]
            count += 1
            if count % 1000 == 0:
                logger.info("eval finishing {}".format(count))
            bs=len(context_list)
            ref=torch.tensor([persona_list[i].index(plabel_list[i]) for i in range(bs)],dtype=torch.long,device=device)
            hyp=torch.max(logits,dim=1)[1]
            all+=len(ref)
            hit1+=torch.sum(hyp==ref,dim=0).item()
        
        logger.info("from k infer p, the hit at 1 is {:.4f}".format(hit1/all))

    # with open(os.path.join(args.out_dir, 'score-iter-{}.txt'.format( global_step)), 'w', encoding='utf-8') as f:
    #     for label, score in zip(labels, scores):
    #         f.write('{}\t{}\n'.format(label, score))

   



    dualpk_model.save_pretrained(os.path.join(args.out_dir,'{}step_dualpk_model'.format(global_step)))
    dualkp_model.save_pretrained(os.path.join(args.out_dir,'{}step_dualkp_model'.format(global_step)))
    #torch.save(dualpk_model,os.path.join(args.out_dir,'{}step_dualpk_model'.format(global_step)))
    #checkpoint_dir=os.path.join(args.out_dir,'{}step_model'.format(global_step))
    #torch.save(dualkp_model,os.path.join(args.out_dir,'{}step_dualkp_model'.format(global_step)))
    logger.info("Saved model checkpoint \n")
    #logger.info("hit at 1 is {:.4f}".format(hit1/all))

best_f1 = -1.
if args.predict:
    predict_step(0)
    #logger.info("predict result: the f1 between predict knowledge and response: {:.6f}".format(f1))
    exit()
for i in range(args.num_steps):
    train_step(i + 1)
    if (i + 1) % args.valid_every == 0:
        predict_step(i+1)