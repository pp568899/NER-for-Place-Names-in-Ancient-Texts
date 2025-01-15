import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import logging
import json
from models.bert_bilstm_crf import BertBiLSTMCRF
from config import Config
import os
from evaluate import evaluate_model, print_metrics
from data_augmentation import augment_data

# 获取项目根目录的绝对路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class AncientNERDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
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

def setup_logging():
    log_file = os.path.join(PROJECT_ROOT, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_training_data(file_path):
    """加载训练数据"""
    abs_file_path = os.path.join(PROJECT_ROOT, file_path)
    print(f"Loading data from: {abs_file_path}")
    
    with open(abs_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [item['text'] for item in data], [item['labels'] for item in data]

def validate_samples(model, tokenizer, config):
    """验证多个样本"""
    test_cases = [
        "巡按御史唐龍檄南昌知府吴嘉聰修南昌府志，開館於白鹿洞中",
        "至廣信，有書致唐龍，自贛還南昌",
        "遊西湖，至孤山、放鶴亭，訪林和靖故居"
    ]
    
    model.eval()
    print("\n验证样本:")
    for text in test_cases:
        encoding = tokenizer(
            text,
            max_length=config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(config.device)
        attention_mask = encoding['attention_mask'].to(config.device)
        
        with torch.no_grad():
            predictions = model(input_ids, attention_mask=attention_mask)
        
        pred_labels = predictions[0][:len(text)]
        
        print(f"\n测试文本: {text}")
        print("预测结果:")
        entities = []
        current_entity = None
        
        for i, (char, label) in enumerate(zip(text, pred_labels)):
            label_name = "O" if label == 0 else "B-LOC" if label == 1 else "I-LOC"
            print(f"{char}: {label_name}")
            
            if label == 1:  # B-LOC
                if current_entity:
                    entities.append(current_entity)
                current_entity = {'text': char}
            elif label == 2 and current_entity:  # I-LOC
                current_entity['text'] += char
            elif current_entity:
                entities.append(current_entity)
                current_entity = None
        
        if current_entity:
            entities.append(current_entity)
            
        if entities:
            print("识别的地名:", ", ".join(e['text'] for e in entities))
        else:
            print("未识别出地名")

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

def train():
    setup_logging()
    logging.info(f"Project root directory: {PROJECT_ROOT}")
    
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
    
    # 加载并增强训练数据
    texts, labels = load_training_data('data/train_data.json')
    augmented_texts, augmented_labels = augment_data(texts, labels)
    logging.info(f"原始样本数: {len(texts)}, 增强后样本数: {len(augmented_texts)}")
    
    # 创建数据集
    dataset = AncientNERDataset(augmented_texts, augmented_labels, tokenizer, config.max_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    model = BertBiLSTMCRF(
        config.bert_model_name,
        config.num_tags,
        hidden_size=config.hidden_size,
        lstm_hidden_size=config.lstm_hidden_size,
        num_lstm_layers=config.num_lstm_layers,
        dropout_rate=config.dropout_rate
    )
    model.to(config.device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    best_loss = float('inf')
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        verbose=True
    )
    
    # 添加warmup
    num_training_steps = len(dataloader) * config.num_epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            optimizer.zero_grad()
            loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 更新学习率
            if batch_idx < num_warmup_steps:
                lr_scale = min(1., float(batch_idx + 1) / num_warmup_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * config.learning_rate
        
            current_loss = loss.item()
            total_loss += current_loss
            progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        logging.info(f'Epoch {epoch+1}/{config.num_epochs}, Average Loss: {avg_loss:.4f}')
        
        validate_samples(model, tokenizer, config)
        
        # 在每个epoch结束后评估模型
        metrics = evaluate_model(model, dataloader, config)
        print_metrics(metrics)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.model_save_path)
            logging.info(f'New best model saved with loss: {best_loss:.4f}')
        
        if early_stopping(avg_loss):
            logging.info(f'Early stopping triggered after {epoch+1} epochs')
            break
        
        # 每个epoch结束后更新学习率
        scheduler.step(avg_loss)

if __name__ == '__main__':
    train()
