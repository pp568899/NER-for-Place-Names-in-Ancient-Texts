import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

def read_excel_data(file_path):
    """读取Excel文件并处理数据"""
    df = pd.read_excel(file_path)
    df = df.drop_duplicates(subset=['事件'])
    return df['事件'].tolist()

def create_label_dict():
    """创建标签字典"""
    return {
        'O': 0,     # 非实体
        'B-LOC': 1, # 地名开始
        'I-LOC': 2  # 地名内部
    }

class AncientTextProcessor:
    def __init__(self, label_dict=None):
        self.label_dict = label_dict if label_dict else create_label_dict()
        self.id2label = {v: k for k, v in self.label_dict.items()}
        
    def process_text(self, text):
        """处理单个文本"""
        chars = list(text)
        return chars

    def create_labels(self, text, entities):
        """根据实体创建标签序列"""
        labels = ['O'] * len(text)
        for entity, entity_type, start, end in entities:
            labels[start] = f'B-{entity_type}'
            for i in range(start + 1, end):
                labels[i] = f'I-{entity_type}'
        return [self.label_dict[label] for label in labels]

class AncientNERDataset(Dataset):
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
        
        # 使用tokenizer处理文本
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 确保标签长度与输入一致
        padded_labels = self._pad_labels(label, self.max_length)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(padded_labels)
        }
    
    def _pad_labels(self, labels, max_length):
        """填充标签序列到指定长度"""
        if len(labels) > max_length:
            return labels[:max_length]
        return labels + [0] * (max_length - len(labels))
