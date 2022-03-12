import argparse
import os
from posix import environ
#os.environ['CUDA_VISIBLE_DEVICES']='5'
from numpy.lib.arraypad import pad
from tokenizers import InputSequence
import torch
import torch.nn.functional as F
import numpy as np
import math
import logging
import random
from torch.cuda import check_error
from tqdm import tqdm
from str2bool import str2bool
import itertools
from datetime import datetime

#from zxlmetric import f1_metric
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
from dataset.persona_dataset import PersonaDataset
from batcher.persona_batcher import PersonaBatcher

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
parser.add_argument("--dualkp_model",type=str,default='/home/futc/bert-base-uncased')# (P|CRK) dual learning model
parser.add_argument("--dualpk_model",type=str,default='/home/futc/bert-base-uncased')# (K|CRP) dual learning model

# parser.add_argument("--count_path",type=str,default='/home/futc/2021work2/knowledge_count.json')
# parser.add_argument("--label_path",type=str,default='/home/futc/2021work2/label.json')
# training scheme
parser.add_argument('--batch_size', type=int, default=1)
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

parser.add_argument("--max_context_length",type=int,default=64)
parser.add_argument("--max_persona_length",type=int,default=64)
parser.add_argument("--max_response_length",type=int,default=64)
parser.add_argument("--max_knowledge_length",type=int,default=64)
parser.add_argument("--n_knowledge",default=32,type=int)
parser.add_argument("--n_persona",default=32,type=int)

# gpu
parser.add_argument('--gpu_list', type=str, default='4')
parser.add_argument('--gpu_ratio', type=float, default=0.85)
parser.add_argument('--n_device', type=int, default=8)
parser.add_argument('--no_cuda', type=str2bool, default=False)

parser.add_argument("--n_layer",default=6,type=int)
parser.add_argument("--reward",default='v3',type=str)
# v1: the target logit - the average logit
# v2: the target probability - 0.5
# v3: the target probability

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
logger.info("train examples {}".format(len(train_dataset)))
logger.info("eval examples {}".format(len(eval_dataset)))

time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.info("Create training dataset end... | %s " % time_str)
tokenizer=BertTokenizer.from_pretrained(args.vocab)
tokenizer.add_special_tokens({'pad_token':'[PAD]','sep_token':'[SEP]'})
batcher = PersonaBatcher(device,tokenizer,args.n_knowledge,args.n_persona,args.max_context_length,args.max_response_length,args.max_knowledge_length,args.max_persona_length)


configuration=BertConfig(num_hidden_layers=args.n_layer)
dualkp_model=BertForSequenceClassification.from_pretrained(args.dualkp_model,config=configuration)
dualkp_model.resize_token_embeddings(len(tokenizer))

dualpk_model=BertForSequenceClassification.from_pretrained(args.dualpk_model,config=configuration)
dualpk_model.resize_token_embeddings(len(tokenizer))
# semip_model=BertModel(configuration)
# if args.semip_model:
#     reloaded=torch.load(args.semip_model)['state_dict']
#     semip_model.load_state_dict(reloaded,strict='True')


#priorp_model.to(device)
#priork_model.to(device)
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
    dualloss_total = 0.0
    for _ in range(args.accum_step):
        context_list,response_list,persona_list,knowledge_list, plabel_list,klabel_list= next(train_loader)
        #The dual learning part
        dualpk_model.train()
        dualkp_model.eval()
        batcher.load(context_list,response_list,persona_list,knowledge_list,plabel_list,klabel_list)
        batch_dict=batcher('k|crp',None,plabel_list)
        input_id=batch_dict['input_id']
        segment_id=batch_dict['segment_id']
        assert input_id.dim()==3 and input_id.shape==segment_id.shape
        bs,n_know,seq_len=input_id.shape
        #(bs*n_know,2)
        dual_klogits=dualpk_model(input_ids=input_id.view(-1,seq_len),attention_mask=input_id.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id.view(-1,seq_len),return_dict=True)['logits']
        #(bs,n_know)
        dual_klogits=dual_klogits.view(bs,n_know,-1)[:,:,1]
        #(bs)
        kind=torch.multinomial(torch.softmax(dual_klogits,dim=1),num_samples=1,replacement=True).squeeze(1)
        #kind=torch.max(dual_klogits,dim=1)[1].detach().cpu().tolist()
        selected_know=[knowledge_list[i][min(kind[i].item(),len(knowledge_list[i])-1)] for i in range(bs)]
        
        with torch.no_grad():
            batch_dict=batcher('p|crk',selected_know,None)
            input_id=batch_dict['input_id']
            segment_id=batch_dict['segment_id']
            assert input_id.dim()==3 and input_id.shape==segment_id.shape
            bs,n_per,seq_len=input_id.shape
            # (bs*n_per,2)
            post_plogits=dualkp_model(input_ids=input_id.view(-1,seq_len),attention_mask=input_id.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id.view(-1,seq_len),return_dict=True)['logits']
            # (bs,n_per)
            post_plogits=post_plogits.view(bs,n_per,-1)[:,:,1]
            #(bs)
            target=torch.tensor([persona_list[i].index(plabel_list[i]) for i in range(bs)],dtype=torch.long,device=device)
            #(bs)
            post_pprob=torch.softmax(post_plogits,dim=1)
            # TODO: the reward need to be designed
            reward=torch.gather(post_pprob,dim=1,index=target.unsqueeze(1)).squeeze(1)
        
        # the tensor to backward the gradient 
        tensor1=torch.gather(torch.log_softmax(dual_klogits,dim=1),dim=1,index=kind.unsqueeze(1)).squeeze(1)
        dual_loss1=-torch.sum(tensor1*reward,dim=0)
        dualloss_total+=dual_loss1.item()
        dual_loss1=dual_loss1/args.accum_step
        dual_loss1.backward()

        dualkp_model.train()
        dualpk_model.eval()
        batch_dict=batcher('p|crk',klabel_list,None)
        input_id=batch_dict['input_id']
        segment_id=batch_dict['segment_id']
        assert input_id.dim()==3 and input_id.shape==segment_id.shape
        bs,n_per,seq_len=input_id.shape
        # (bs*n_per,2)
        dual_plogits=dualkp_model(input_ids=input_id.view(-1,seq_len),attention_mask=input_id.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id.view(-1,seq_len),return_dict=True)['logits']
        # (bs,n_per)
        dual_plogits=dual_plogits.view(bs,n_per,-1)[:,:,1]
        pind=torch.multinomial(torch.softmax(dual_plogits,dim=1),num_samples=1,replacement=True).squeeze(1)
        #pind=torch.max(dual_plogits,dim=1)[1].detach().cpu().tolist()
        selected_per=[persona_list[i][min(pind[i].item(), len(persona_list[i])-1)] for i in range(bs)]
        with torch.no_grad():
            batch_dict=batcher('k|crp',None,selected_per)
            input_id=batch_dict['input_id']
            segment_id=batch_dict['segment_id']
            assert input_id.dim()==3 and input_id.shape==segment_id.shape
            bs,n_know,seq_len=input_id.shape
            #(bs*n_know,2)
            post_klogits=dualpk_model(input_ids=input_id.view(-1,seq_len),attention_mask=input_id.view(-1,seq_len)!=tokenizer.pad_token_id,token_type_ids=segment_id.view(-1,seq_len),return_dict=True)['logits']
            #(bs,n_know)
            post_klogits=post_klogits.view(bs,n_know,-1)[:,:,1]
            #(bs)
            target=torch.tensor([knowledge_list[i].index(klabel_list[i]) for i in range(bs)],dtype=torch.long,device=device)
            #(bs)
            post_kprob=torch.softmax(post_klogits,dim=1)
            #reward=torch.gather(post_kprob,dim=1,index=target.unsqueeze(1)).squeeze(1)-torch.tensor([0.5]*bs,dtype=torch.float,device=device)
            reward=torch.gather(post_kprob,dim=1,index=target.unsqueeze(1)).squeeze(1)
            #reward=torch.gather(post_klogits,dim=1,index=target.unsqueeze(1)).squeeze(1)-torch.mean(post_klogits,dim=1)
        tensor2=torch.gather(torch.log_softmax(dual_plogits,dim=1),dim=1,index=pind.unsqueeze(1)).squeeze(1)
        dual_loss2=-torch.sum(tensor2*reward,dim=0)
        dualloss_total+=dual_loss2.item()
        dual_loss2=dual_loss2/args.accum_step
        dual_loss2.backward()
        # if args.dual_loss=='ce':
        #     loss=F.cross_entropy(plogits,target)
        # elif args.dual_loss=='mm':
        #     loss=F.multi_margin_loss(logits,target)
        # label=batch_dict['label']
        # loss=model(input_ids=input_id,attention_mask=attention_mask,token_type_ids=segment_id,labels=label,return_dict=True)['loss']
        # loss.backward()
        # ks_loss_total += loss.item()

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
            global_step, dualloss_total, dualkp_scheduler.get_lr()[0], time_str
        ))
        # sys.stdout.flush()

def predict_step(global_step):
    #if split == 'test_seen':
    #    test_loader = test_seen_loader
    #else:
    #    raise ValueError
    dualkp_model.eval()
    dualpk_model.eval()
    hit1 = 0
    hit2 = 0
    hit5 = 0 
    hit10= 0
    count=0
    with torch.no_grad():
        for context_list,response_list,persona_list,knowledge_list, plabel_list,klabel_list in eval_loader:
            batcher.load(context_list,response_list,persona_list,knowledge_list,plabel_list,klabel_list)
            batch_dict=batcher('k|crp',None,plabel_list)
            input_id=batch_dict['input_id']
            segment_id=batch_dict['segment_id']
            assert input_id.dim()==3 and input_id.shape==segment_id.shape
            bs,n_know,seq_len=input_id.shape
            logits = dualpk_model(input_ids=input_id.view(-1,seq_len),attention_mask=(input_id.view(-1,seq_len)!=tokenizer.pad_token_id),token_type_ids=segment_id.view(-1,seq_len),return_dict=True)['logits']
            logits = logits.view(bs,n_know,-1)[:,:,1]
            count += len(context_list)
            if count % 1000 == 0:
                logger.info("eval finishing {}".format(count))
            bs=len(context_list)
            ref=torch.tensor([knowledge_list[i].index(klabel_list[i]) for i in range(bs)],dtype=torch.long,device=device)
            # hyp=torch.max(logits,dim=1)[1]
            # hit1+=torch.sum(hyp==ref,dim=0).item()
            hyp=torch.topk(logits,k=10)[1]
            hit1+=(hyp[:,:1]==ref[:,None]).sum(1).sum(0).item()
            hit2+=(hyp[:,:2]==ref[:,None]).sum(1).sum(0).item()
            hit5+=(hyp[:,:5]==ref[:,None]).sum(1).sum(0).item()
            hit10+=(hyp[:,:10]==ref[:,None]).sum(1).sum(0).item()
            
    logger.info("knowledge prediction hit1 is {:.4f}".format(hit1/count))
    logger.info("knowledge prediction hit2 is {:.4f}".format(hit2/count))
    logger.info("knowledge prediction hit5 is {:.4f}".format(hit5/count))
    logger.info("knowledge prediction hit10 is {:.4f}".format(hit10/count))

    hit1 = 0
    hit2 = 0
    hit5=0
    hit10=0
    count=0
    with torch.no_grad():
        for context_list,response_list,persona_list,knowledge_list, plabel_list,klabel_list in eval_loader:
            batcher.load(context_list,response_list,persona_list,knowledge_list,plabel_list,klabel_list)
            batch_dict=batcher('p|crk',klabel_list,None)
            input_id=batch_dict['input_id']
            segment_id=batch_dict['segment_id']
            assert input_id.dim()==3 and input_id.shape==segment_id.shape
            bs,n_per,seq_len=input_id.shape
            logits = dualkp_model(input_ids=input_id.view(-1,seq_len),attention_mask=(input_id.view(-1,seq_len)!=tokenizer.pad_token_id),token_type_ids=segment_id.view(-1,seq_len),return_dict=True)['logits']
            logits = logits.view(bs,n_per,-1)[:,:,1]
            count += len(context_list)
            if count % 1000 == 0:
                logger.info("eval finishing {}".format(count))
            bs=len(context_list)
            ref=torch.tensor([persona_list[i].index(plabel_list[i]) for i in range(bs)],dtype=torch.long,device=device)
            # hyp=torch.max(logits,dim=1)[1]
            # hit1+=torch.sum(hyp==ref,dim=0).item()
            hyp=torch.topk(logits,k=10)[1]
            hit1+=(hyp[:,:1]==ref[:,None]).sum(1).sum(0).item()
            hit2+=(hyp[:,:2]==ref[:,None]).sum(1).sum(0).item()
            hit5+=(hyp[:,:5]==ref[:,None]).sum(1).sum(0).item()
            hit10+=(hyp[:,:10]==ref[:,None]).sum(1).sum(0).item()
            
    logger.info("persona prediction hit1 is {:.4f}".format(hit1/count))
    logger.info("persona prediction hit2 is {:.4f}".format(hit2/count))
    logger.info("persona prediction hit5 is {:.4f}".format(hit5/count))
    logger.info("persona prediction hit10 is {:.4f}".format(hit10/count))

    if args.predict:
        exit()
    # with open(os.path.join(args.out_dir, 'score-iter-{}.txt'.format( global_step)), 'w', encoding='utf-8') as f:
    #     for label, score in zip(labels, scores):
    #         f.write('{}\t{}\n'.format(label, score))

    # time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # logger.info("**********************************")
    # logger.info("test results..........")
    # logger.info("Step: %d \t|  %s" % (global_step, time_str))

    #model_to_save = model.module if hasattr(model, "module") else model
    #checkpoint_dir=os.path.join(args.out_dir,'{}step_model'.format(global_step))
    dualpk_model.save_pretrained(os.path.join(args.out_dir,'{}step_dualpk_model'.format(global_step)))
    dualkp_model.save_pretrained(os.path.join(args.out_dir,'{}step_dualkp_model'.format(global_step)))
    #torch.save(dualpk_model,os.path.join(args.out_dir,'{}step_dualpk_model'.format(global_step)))
    #checkpoint_dir=os.path.join(args.out_dir,'{}step_model'.format(global_step))
    #torch.save(dualkp_model,os.path.join(args.out_dir,'{}step_dualkp_model'.format(global_step)))
    logger.info("Saved model checkpoint \n")
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