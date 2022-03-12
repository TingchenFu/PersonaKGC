from transformers import BertPreTrainedModel
from transformers import BertConfig
from transformers import BertModel
import torch
import torch.nn as nn
from model.mlp import mlp
class BertPosterior(BertPreTrainedModel):
    base_model_prefix='bert'
    def __init__(self,config:BertConfig, n_label=2 ) -> None:
        super(BertPosterior,self).__init__(config)
        self.bert=BertModel(config)
        self.n_label=n_label
        self.classifier=mlp(config.hidden_size,self.num_label)

    def resize_token_embeddings(self, new_num_tokens) -> torch.nn.Embedding:
        return super().resize_token_embeddings(new_num_tokens=new_num_tokens)

    def forward(self, input_id,attention_mask=None):

        assert input_id.shape[0]==attention_mask.shape[0]
        bs=input_id.shape[0]
        # (bs,hidden_size)
        logits=self.bert(input_ids=input_id,attention_mask=attention_mask,return_dict='True')['logits']
        #(bs,2)
        return self.classifier(logits)

        context_mask=attention_mask[:,:self.max_query_length]
        knowledge_mask=attention_mask[:,self.max_query_length:]

        #(bs,slen,hidden_dim)
        enc=self.bert(input_id,attention_mask,return_dict=True).last_hidden_state
        #(bs,crlen,hidden_dim)
        context_enc=enc[:,:self.max_query_length,:]
        context_enc=context_enc*context_mask.unsqueeze(2).expand_as(context_enc).float()
        #(bs)
        context_realength=torch.sum(attention_mask,dim=1)
        #(bs,klen,hidden_dim)
        knowledge_enc=enc[:,self.max_query_length:,:]
        # (bs,hidden_dim) 
        # average pooling
        context_rep=torch.sum(context_enc,dim=1)/context_realength[:,None].float()
        # (bs,klen,hidden_dim*2)
        mlp_in=torch.cat([context_rep.unsqueeze(1).expand_as(knowledge_enc),knowledge_enc],dim=-1)
        # (bs,klen)
        start_mlp_out=self.start_mlp(mlp_in).squeeze()
        end_mlp_out=self.end_mlp(mlp_in).squeeze()
        Zs=torch.masked_fill(start_mlp_out,knowledge_mask==0,1e-18)
        Ze=torch.masked_fill(end_mlp_out,knowledge_mask==0,1e-18)

        # (bs,hidden_dim)
        start_bag_logits=self.start_bag_mlp(torch.matmul(Zs.unsqueeze(1),knowledge_enc).squeeze(1))
        end_bag_logits=self.end_bag_mlp(torch.matmul(Ze.unsqueeze(1),knowledge_enc).squeeze(1))
        return Zs,Ze,start_bag_logits,end_bag_logits