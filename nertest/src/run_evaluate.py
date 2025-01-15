import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import json
import os
from models.bert_bilstm_crf import BertBiLSTMCRF, AncientNERDataset
from config import Config
from evaluate import evaluate_model, print_metrics

def load_test_data(file_path):
    """加载测试数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [item['text'] for item in data], [item['labels'] for item in data]

def main():
    # 1. 加载配置
    config = Config()
    print("配置加载完成")

    # 2. 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
    print("Tokenizer 初始化完成")

    # 3. 加载测试数据
    test_file = os.path.join(config.project_root, 'data/test_data.json')
    if not os.path.exists(test_file):
        print(f"测试数据文件不存在: {test_file}")
        # 如果没有测试数据，使用训练数据的一部分作为测试
        train_file = os.path.join(config.project_root, 'data/train_data.json')
        texts, labels = load_test_data(train_file)
        # 使用20%的数据作为测试集
        split_idx = int(len(texts) * 0.8)
        test_texts = texts[split_idx:]
        test_labels = labels[split_idx:]
        print(f"使用训练数据的20%作为测试集，共 {len(test_texts)} 个样本")
    else:
        test_texts, test_labels = load_test_data(test_file)
        print(f"测试数据加载完成，共 {len(test_texts)} 个样本")

    # 4. 创建测试数据集和数据加载器
    test_dataset = AncientNERDataset(test_texts, test_labels, tokenizer, config.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    print("数据加载器创建完成")

    # 5. 初始化模型
    model = BertBiLSTMCRF(
        config.bert_model_name,
        config.num_tags,
        hidden_size=config.hidden_size,
        lstm_hidden_size=config.lstm_hidden_size,
        num_lstm_layers=config.num_lstm_layers,
        dropout_rate=config.dropout_rate
    )
    print("模型初始化完成")

    # 6. 加载训练好的模型
    model_path = os.path.join(config.project_root, 'models/ancient_ner_model.pt')
    if not os.path.exists(model_path):
        print(f"错误：找不到训练好的模型文件: {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.to(config.device)
    print("模型加载完成")

    # 7. 运行评估
    print("\n开始评估...")
    model.eval()
    metrics = evaluate_model(model, test_dataloader, config)
    
    # 8. 打印评估结果
    print("\n评估完成！")
    print_metrics(metrics)

    # 9. 保存评估结果
    results_path = os.path.join(config.project_root, 'evaluation_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("=== 评估结果 ===\n")
        for label in ['O', 'B-LOC', 'I-LOC']:
            f.write(f"\n{label}:\n")
            f.write(f"Precision: {metrics[label]['precision']:.4f}\n")
            f.write(f"Recall: {metrics[label]['recall']:.4f}\n")
            f.write(f"F1: {metrics[label]['f1']:.4f}\n")
        
        f.write("\n总体指标:\n")
        f.write(f"Macro Precision: {metrics['macro_avg']['precision']:.4f}\n")
        f.write(f"Macro Recall: {metrics['macro_avg']['recall']:.4f}\n")
        f.write(f"Macro F1: {metrics['macro_avg']['f1']:.4f}\n")
        
        f.write("\n混淆矩阵:\n")
        f.write(str(metrics['confusion_matrix']))
    
    print(f"\n评估结果已保存到: {results_path}")

if __name__ == '__main__':
    main() 