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
from transformers import GPT2Tokenizer

from transformers import AdamW
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
parser.add_argument("--selectedk_path",type=str,default='')
parser.add_argument("--selectedp_path",type=str,default='')
parser.add_argument("--pseudo_path",type=str,default='/home/futc/2021work2/pseudo')

# model 
parser.add_argument("--vocab",type=str,default='/home/futc/gpt2')
parser.add_argument("--model",type=str,default='/home/futc/gpt2')

# parser.add_argument("--count_path",type=str,default='/home/futc/2021work2/knowledge_count.json')
# parser.add_argument("--label_path",type=str,default='/home/futc/2021work2/label.json')
# training scheme
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--eval_batch_size', type=int, default=4)
parser.add_argument('--num_steps', type=int, default=1000000)
parser.add_argument('--accum_step', type=int, default=8)
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

parser.add_argument("--decoder",type=str,default='ablationp')
# the constitution of prompt
parser.add_argument("--prompt",type=str,default='pseudo')
# generate
parser.add_argument("--min_generate_length",type=int,default=10)
parser.add_argument("--max_generate_length",type=int,default=25)
parser.add_argument("--beamsize",type=int,default=2)

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
assert args.prompt in ['selected','pseudo']
assert args.decoder in ['v0','v1','ablationp','ablationk']
out_dir = os.path.join(args.dump_path, args.exp_name)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
args.out_dir=out_dir
logger.addHandler(logging.FileHandler(os.path.join(args.out_dir, "log"), 'w'))
logger.info("\nParameters:")
for attr, value in sorted(vars(args).items()):
    logger.info("{}={}".format(attr.upper(), value))



class PersonaDataset(Dataset):
    def __init__(self,convo_path,selectedp_path,selectedk_path,pseudo_path,mode) -> None:
        super(PersonaDataset,self).__init__()
        self.examples=[]
        self.selected_knowledge=[]
        self.selected_persona=[]
        assert mode in ['train','eval']
        self.mode=mode
        for date in os.listdir(convo_path):
            if (date=='2015-05' or date=='2015-06') and mode=='train':
                continue
            if mode=='eval' and date!='2015-05':
                continue
            fconvo=open(os.path.join(convo_path,date),mode='r',encoding='utf-8')
            fpseudo=open(os.path.join(pseudo_path,date),mode='r',encoding='utf-8')
            for line1,line2 in zip(fconvo.readlines(),fpseudo.readlines()):
                data=json.loads(line1)
                author=data['author'][-1]
                sid=data['sid']
                label=json.loads(line2)
                self.examples.append({
                    'context':data['dialog'][:-1],
                    'response':data['dialog'][-1],
                    'klabel':label['klabel'],
                    'plabel':label['plabel']
                })
            if args.debug:
                break
        
        if mode=='eval' and args.prompt=='selected':
            fsk=open(selectedk_path,mode='r',encoding='utf-8')
            for line in fsk.readlines():
                self.selected_knowledge.append(line.strip('\n'))
            fsk.close()

            fsp=open(selectedp_path,mode='r',encoding='utf-8')
            for line in fsp.readlines():
                self.selected_persona.append(line.strip('\n'))
            fsp.close()

        if args.debug and mode == 'eval':
            self.examples=self.examples[:16]
            self.selected_persona=self.selected_persona[:16]
            self.selected_knowledge=self.selected_knowledge[:16]
        logger.info("{} examples {}".format(mode,len(self.examples)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        example=self.examples[i]
        if self.mode=='eval' and args.prompt=='selected':
            return example['context'], example['response'], self.selected_persona[i], self.selected_knowledge[i]
        else:
            assert example['plabel'] is not None
            assert example['klabel'] is not None
            return example['context'], example['response'], example['plabel'],example['klabel']

    @staticmethod
    def collate_fn(batch):
        context_list = [item[0] for item in batch]
        response_list= [item[1] for item in batch]
        persona_list=[item[2] for item in batch]
        knowledge_list=[item[3]for item in batch]
        return context_list,response_list,persona_list,knowledge_list

class PersonaBatcher(object):
    def __init__(self, device, tokenizer, max_context_length,max_response_length,max_knowledge_length,max_persona_length):
        self.device=device
        self.tokenizer=tokenizer
        self.max_context_length=max_context_length
        self.max_response_length=max_response_length
        self.max_knowledge_length=max_knowledge_length
        self.max_persona_length=max_persona_length
    
    # batcher for the v1 and ablation
    def __call__(self, context_list,response_list,persona_list,knowledge_list,mode='train'):
        assert len(context_list)==len(response_list)==len(persona_list)==len(knowledge_list)
        bs=len(context_list)
        batch_input_id=[]
        batch_label=[]
        for i in range(bs):
            input_id=self.tokenizer.encode(' '.join(context_list[i]))[:self.max_context_length]
            label=[-100]*len(input_id)
            if mode=='train':
                input_id+=self.tokenizer.encode(response_list[i])[:self.max_response_length]
                label+=self.tokenizer.encode(response_list[i])[:self.max_response_length]
            batch_input_id.append(input_id)
            batch_label.append(label)
        longest=max([len(batch_input_id[i])for i in range(bs)])
        for i in range(bs):
            batch_input_id[i].extend([self.tokenizer.pad_token_id]*(longest-len(batch_input_id[i])))
            batch_label[i].extend([-100]*(longest-len(batch_label[i])))
        batch_knowledge_id=self.tokenizer.batch_encode_plus(knowledge_list,truncation=True,max_length=self.max_knowledge_length,padding='longest',return_tensors='pt')['input_ids']
        batch_persona_id=self.tokenizer.batch_encode_plus(persona_list,truncation=True,max_length=self.max_persona_length,padding='longest',return_tensors='pt')['input_ids']

        batch_input_id=torch.tensor(batch_input_id,dtype=torch.long,device=self.device)
        batch_label=torch.tensor(batch_label,dtype=torch.long,device=self.device)
        batch_knowledge_id=batch_knowledge_id.cuda()
        batch_persona_id=batch_persona_id.cuda()
        if batch_knowledge_id.shape[-1]>=1024:
            logger.info("too long batch knowledge id!!!")
            batch_knowledge_id=batch_knowledge_id[:,1023]
        if batch_persona_id.shape[-1]>=1024:
            logger.info("too long batch knowledge id!!!")
            batch_persona_id=batch_persona_id[:,1023]

        return{
            'input_id':batch_input_id,
            'knowledge_id':batch_knowledge_id,
            'persona_id':batch_persona_id,
            'label':batch_label
        }
    # the batcher of v0
    # def __call__(self, context_list,response_list,persona_list,knowledge_list):
    #     assert len(context_list)==len(response_list)==len(persona_list)==len(knowledge_list)
    #     bs=len(context_list)
    #     batch_input_id=[]
    #     batch_label=[]
    #     for i in range(bs):
    #         input_id = self.tokenizer.encode(' '.join(context_list[i]) +' '+ persona_list[i].strip(' ') +' '+ knowledge_list[i].strip(' ')+' ',truncation=True,max_length=self.max_context_length+self.max_knowledge_length+self.max_persona_length)
    #         label = [-100]*len(input_id)
    #         input_id.extend(self.tokenizer.encode(response_list[i],truncation=True,max_length=self.max_response_length))
    #         label.extend(self.tokenizer.encode(response_list[i],truncation=True,max_length=self.max_response_length))
    #         batch_input_id.append(input_id)
    #         batch_label.append(label)
    #     longest=max([len(id) for id in batch_input_id])
    #     for i in range(bs):
    #         if len(batch_input_id[i])==longest:
    #             continue
    #         padding_length=longest-len(batch_input_id[i])
    #         batch_input_id[i].extend([self.tokenizer.pad_token_id]*padding_length)
    #         batch_label[i].extend([-100]*padding_length)
    #     batch_input_id=torch.tensor(batch_input_id,dtype=torch.long,device=self.device)
    #     batch_label=torch.tensor(batch_label,dtype=torch.long,device=self.device)
    #     return {
    #         'input_id':batch_input_id,
    #         'label':batch_label
    #     }

def recall_f1(scores,knowledges,responses):
    count=[len(k) for k in knowledges]
    # all equals to the number of context
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

train_dataset=PersonaDataset(args.convo_path,args.selectedp_path,args.selectedk_path,args.pseudo_path,mode='train')
eval_dataset=PersonaDataset(args.convo_path,args.selectedp_path,args.selectedk_path,args.pseudo_path,mode='eval')
train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=PersonaDataset.collate_fn)
eval_loader=DataLoader(eval_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=PersonaDataset.collate_fn)
train_loader=itertools.cycle(train_loader)

time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.info("Create training dataset end... | %s " % time_str)
tokenizer=GPT2Tokenizer.from_pretrained(args.vocab)
tokenizer.add_special_tokens({'pad_token':'[PAD]','sep_token':'[SEP]'})
batcher = PersonaBatcher(device, tokenizer, args.max_context_length, args.max_response_length, args.max_knowledge_length,args.max_persona_length)

if args.decoder=='v1':
    from model.reimpl_gpt2v1 import GPT2LMHeadModel
    model=GPT2LMHeadModel.from_pretrained(args.model)
elif args.decoder=='v0':
    from model.reimpl_gpt2v0 import GPT2LMHeadModel
    model=GPT2LMHeadModel.from_pretrained(args.model)
elif args.decoder=='ablationk':
    from model.reimpl_gpt2_ablationk import GPT2LMHeadModel
    model=GPT2LMHeadModel.from_pretrained(args.model)
elif args.decoder=='ablationp':
    from model.reimpl_gpt2_ablationp import GPT2LMHeadModel
    model=GPT2LMHeadModel.from_pretrained(args.model)
else:
    # TODO: add other tricks and implementation in decoder side
    raise NotImplementedError
model.resize_token_embeddings(len(tokenizer))
# semip_model=BertModel(configuration)
# if args.semip_model:
#     reloaded=torch.load(args.semip_model)['state_dict']
#     semip_model.load_state_dict(reloaded,strict='True')


#priorp_model.to(device)
#priork_model.to(device)
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
    loss_total = 0.0
    for _ in range(args.accum_step):
        context_list,response_list,persona_list,knowledge_list = next(train_loader)
        model.train()
        batch_dict=batcher(context_list,response_list,persona_list,knowledge_list)
        input_id=batch_dict['input_id']
        knowledge_id=batch_dict['knowledge_id']
        persona_id=batch_dict['persona_id']
        label=batch_dict['label']
        if args.decoder=='v1':
            loss=model(input_ids=input_id,attention_mask=(input_id!=tokenizer.pad_token_id),knowledge_id=knowledge_id,knowledge_attention_mask=(knowledge_id!=tokenizer.pad_token_id),persona_id=persona_id,persona_attention_mask=(persona_id!=tokenizer.pad_token_id),labels=label,return_dict=True)['loss']
        elif args.decoder=='ablationp':
            loss=model(input_ids=input_id,attention_mask=(input_id!=tokenizer.pad_token_id),knowledge_id=knowledge_id,knowledge_attention_mask=(knowledge_id!=tokenizer.pad_token_id),labels=label,return_dict=True)['loss']
        elif args.decoder=='ablationk':
            loss=model(input_ids=input_id,attention_mask=(input_id!=tokenizer.pad_token_id),persona_id=persona_id,persona_attention_mask=(persona_id!=tokenizer.pad_token_id),labels=label,return_dict=True)['loss']
        elif args.decoder=='v0':
            loss=model(input_ids=input_id,attention_mask=(input_id!=tokenizer.pad_token_id),labels=label,return_dict=True)['loss']
        loss_total+=loss.item()
        loss=loss/args.accum_step
        loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip)
    if grad_norm >= 1e2:
        logger.info('WARNING : Exploding Gradients {:.2f}'.format(grad_norm))
    model_optimizer.step()
    model_scheduler.step()
    model_optimizer.zero_grad()

    if global_step % args.print_every == 0 and global_step != 0:
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info("Step: %d \t| ks_loss: %.3f \t| lr: %.8f \t| %s" % (
            global_step, loss_total, model_scheduler.get_lr()[0], time_str
        ))
        # sys.stdout.flush()

def predict_step(global_step):
    model.eval()
    hypothesis=[]
    count=0
    for context_list,response_list,persona_list,knowledge_list in eval_loader:
        with torch.no_grad():
            batch_dict=batcher(context_list,response_list,persona_list,knowledge_list,mode='eval')
            input_id=batch_dict['input_id']
            knowledge_id=batch_dict['knowledge_id']
            persona_id=batch_dict['persona_id']
            label=batch_dict['label']
            if args.decoder=='v1':
                modelout=model.generate_beam_search(input_ids=input_id,\
                            attention_mask=(input_id!=tokenizer.pad_token_id), \
                            knowledge_id=knowledge_id,\
                            knowledge_attention_mask=(knowledge_id!=tokenizer.pad_token_id),\
                            persona_id=persona_id,\
                            persona_attention_mask=(persona_id!=tokenizer.pad_token_id), \
                            cur_len=input_id.shape[1], \
                            min_length=input_id.shape[1]+args.min_generate_length, \
                            max_length=input_id.shape[1]+args.max_generate_length, \
                            eos_token_id=tokenizer.eos_token_id, \
                            pad_token_id=tokenizer.pad_token_id, \
                            num_beams=args.beamsize, \
                            vocab_size=len(tokenizer))
            elif args.decoder=='ablationk':
                modelout=model.generate_beam_search(input_ids=input_id,\
                            attention_mask=(input_id!=tokenizer.pad_token_id), \
                            persona_id=persona_id,\
                            persona_attention_mask=(persona_id!=tokenizer.pad_token_id), \
                            cur_len=input_id.shape[1], \
                            min_length=input_id.shape[1]+args.min_generate_length, \
                            max_length=input_id.shape[1]+args.max_generate_length, \
                            eos_token_id=tokenizer.eos_token_id, \
                            pad_token_id=tokenizer.pad_token_id, \
                            num_beams=args.beamsize, \
                            vocab_size=len(tokenizer))
            elif args.decoder=='ablationp':
                modelout=model.generate_beam_search(input_ids=input_id,\
                            attention_mask=(input_id!=tokenizer.pad_token_id), \
                            knowledge_id=knowledge_id,\
                            knowledge_attention_mask=(knowledge_id!=tokenizer.pad_token_id),\
                            cur_len=input_id.shape[1], \
                            min_length=input_id.shape[1]+args.min_generate_length, \
                            max_length=input_id.shape[1]+args.max_generate_length, \
                            eos_token_id=tokenizer.eos_token_id, \
                            pad_token_id=tokenizer.pad_token_id, \
                            num_beams=args.beamsize, \
                            vocab_size=len(tokenizer))
            elif args.decoder=='v0':
                modelout=model.generate_beam_search(input_ids=input_id,\
                            attention_mask=(input_id!=tokenizer.pad_token_id), \
                            cur_len=input_id.shape[1], \
                            min_length=input_id.shape[1]+args.min_generate_length, \
                            max_length=input_id.shape[1]+args.max_generate_length, \
                            eos_token_id=tokenizer.eos_token_id, \
                            pad_token_id=tokenizer.pad_token_id, \
                            num_beams=args.beamsize, \
                            vocab_size=len(tokenizer))
            generated=tokenizer.batch_decode(modelout,skip_special_tokens=True)
            hypothesis.extend(generated)
            if len(hypothesis)%1000==0:
                logger.info("decode finish {}".format(len(hypothesis)))
    f=open(os.path.join(args.out_dir,'{}step_result'.format(global_step)),mode='w',encoding='utf-8')
    for hyp in hypothesis:
        f.write(hyp.replace('\n','')+'\n')
    f.close()
    if not args.predict:
        model.save_pretrained(os.path.join(args.out_dir,'{}step_model'.format(global_step)))

best_f1 = -1.
if args.predict:
    predict_step(0)
    exit()
for i in range(args.num_steps):
    torch.cuda.empty_cache()
    train_step(i + 1)
    if (i + 1) % args.valid_every == 0:
        predict_step(i+1)