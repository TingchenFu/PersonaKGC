import torch

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
        assert len(context_list)==len(response_list)==len(persona_list)==len(knowledge_list)
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
        assert glue in ['p|crk','k|crp','k|cp','p|cr','p|c','k|cr','k|c']
        # Note that k|cr and k|c are for ablation only
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
                    input_id.append([self.tokenizer.cls_token_id]+ self.context_id_list[i] +[self.tokenizer.sep_token_id]+ golden_persona_id_list[i] +[self.tokenizer.sep_token_id]+self.knowledge_id_list[i][j]+[self.tokenizer.sep_token_id])
                    segment_id.append([0]*(self.longest_context+longest_goldenp+3)+ [1]*(self.longest_knowledge+1))
                batch_input_id.append(input_id)
                batch_segment_id.append(segment_id)
        elif glue=='p|cr':
            bs=self.bs
            for i in range(bs):
                input_id=[]
                segment_id=[]
                for j in range(self.n_persona):
                    input_id.append([self.tokenizer.cls_token_id]+ self.context_id_list[i] +[self.tokenizer.sep_token_id]+ self.response_id_list[i]+ [self.tokenizer.sep_token_id] +self.persona_id_list[i][j]+[self.tokenizer.sep_token_id])
                    segment_id.append([0]*(self.longest_context+self.longest_response+3)+ [1]*(self.longest_persona+1))
                batch_input_id.append(input_id)
                batch_segment_id.append(segment_id)
                
        elif glue=='k|cr':
            bs=self.bs
            for i in range(bs):
                input_id=[]
                segment_id=[]
                for j in range(self.n_knowledge):
                    input_id.append([self.tokenizer.cls_token_id]+ self.context_id_list[i] + [self.tokenizer.sep_token_id] + self.response_id_list[i] + [self.tokenizer.sep_token_id] + self.knowledge_id_list[i][j] + [self.tokenizer.sep_token_id])
                    segment_id.append([0]*(self.longest_context+self.longest_response+3) + [1]*(self.longest_knowledge+1))
                batch_input_id.append(input_id)
                batch_segment_id.append(segment_id)

        elif glue=='k|c':
            bs=self.bs
            for i in range(bs):
                input_id=[]
                segment_id=[]
                for j in range(self.n_knowledge):
                    input_id.append([self.tokenizer.cls_token_id]+ self.context_id_list[i] + [self.tokenizer.sep_token_id] + self.knowledge_id_list[i][j] + [self.tokenizer.sep_token_id])
                    segment_id.append([0]*(self.longest_context+2)+[1]*(self.longest_knowledge+1))
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