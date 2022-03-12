import torch
from torch.utils.data import Dataset
import os
import json
import random
class PersonaDataset(Dataset):
    def __init__(self,convo_path,persona_path,knowledge_path,pseudo_path,mode,debug=False) -> None:
        super(PersonaDataset,self).__init__()
        self.examples=[]
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