import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

#模块只有两部分 一个是BertModel 一个是分类头
class myBertModel(nn.Module):
    def __init__(self):
        super(myBertModel, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_path)
        # config = BertConfig.from_pretrained(bert_path)
        # self.bert = BertModel(config)

        self.cls_head = nn.linear(768, num_class)
        #分词器
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
    
    def forward(self, text):
        input = self.tokenizer(text, truncation=True, padding="max_length", max_length=128)
        input_ids = input["input_ids"].to(self.device)
        token_type_ids = input["token_type_ids"].to(self.device)
        attention_mask = input["attention_mask"].to(self.device)
        sequence_out, pooler_out = self.bert(input_ids=input_ids, 
                        token_type_ids=token_type_ids, 
                        attention_mask=attention_mask,
                        return_dict=False)  #return_dict=False表示返回的是一个元组
        
        pred = self.cls_head(pooler_out)
        return pred

model = myBertModel("../bert-base-uncased", 2)
pred= model("Hello, my dog is cute") 
    