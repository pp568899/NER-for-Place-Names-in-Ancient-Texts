import torch
import torch.nn as nn
from transformers import BertModel
from TorchCRF import CRF

class BertBiLSTMMHACRF(nn.Module):
    def __init__(self, bert_model_name, num_tags, hidden_size=768, 
                 lstm_hidden_size=128, num_lstm_layers=1, dropout_rate=0.2):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # 简化为单个LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_tags)
        
        # 添加转移矩阵约束
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        # 设置非法转移为很大的负数
        self.transitions.data[0, 2] = -10000  # O到I-LOC的转移
        self.crf = CRF(num_tags, batch_first=True)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        lstm_output, _ = self.lstm(sequence_output)
        sequence_output = self.dropout(lstm_output)
        emissions = self.classifier(sequence_output)
        
        if labels is not None:
            # 计算类别权重
            label_counts = torch.bincount(labels[labels != -100].flatten(), minlength=3)
            total = label_counts.sum()
            weights = total / (label_counts + 1e-10)
            weights = weights / weights.sum()
            
            loss = -self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            loss = loss * weights[1]  # 给予B-LOC更高的权重
            return loss
        else:
            return self.crf.decode(emissions, mask=attention_mask.byte())

class AncientNERDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        if len(label) > self.max_length:
            label = label[:self.max_length]
        else:
            label = label + [0] * (self.max_length - len(label))
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }