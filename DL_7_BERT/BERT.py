from transformers import BertModel, BertTokenizer

bert = BertModel.from_pretrained('bert-base-uncased')
#激活函数gelu


#计算模型参数函数
# def get_parameter_number(model):
#     total_num = sum(p.numel() for p in model.parameters())
#     trainable_num = sum(p.numel() for p in model.parameters() is p.requires_grad)
#     return {'Total':total_num,'Trainable':trainable_num}

# print(get_parameter_number(bert))

# emb_num = 21128*768 + 2*768 + 512*768 #忽略了bias

# self_att_num = 768*768*3 + 768*768 + 768+3072 + 768*3072
# all_att_num = 12*self_att_num

# print(emb_num+all_att_num)

# for name,para in bert.named_parameters():
#     print(name,para.shape)

#分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

input ="Hello, my dog is cute"
out = tokenizer(input, truncation=True, padding="max_length", max_length=128)
print(out)