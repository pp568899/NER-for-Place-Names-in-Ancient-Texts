import torch
import os

class Config:
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 模型参数
    bert_model_name = 'bert-base-chinese'
    max_length = 128
    hidden_size = 768
    
    # 训练参数
    batch_size = 2
    num_epochs = 50
    learning_rate = 5e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    num_tags = 3  # O, B-LOC, I-LOC
    
    # 早停参数
    patience = 10
    min_delta = 1e-3
    
    # 路径
    data_path = os.path.join(project_root, 'data/train_data.json')
    model_save_path = os.path.join(project_root, 'models/ancient_ner_model.pt')
    
    # 设备
    device = torch.device("cpu")
    
    # 多头注意力头数
    num_heads = 8
    
    # LSTM 相关参数
    lstm_hidden_size = 128
    num_lstm_layers = 1
    dropout_rate = 0.2

    
    # 优化训练参数
    batch_size = 2
    num_epochs = 50
    learning_rate = 5e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    
    # 添加学习率调度器参数
    lr_scheduler_factor = 0.1
    lr_scheduler_patience = 3
    
    # 添加梯度裁剪
    max_grad_norm = 1.0
    
    # 添加早停改进
    patience = 10
    min_delta = 1e-3