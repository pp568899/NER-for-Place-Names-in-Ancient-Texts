import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix  # sklearn 是 scikit-learn 的别名，这样写也可以
import numpy as np



def calculate_metrics(all_predictions, all_labels):
    """计算详细的评估指标"""
    # 将预测和标签展平
    predictions_flat = [p for preds in all_predictions for p in preds]
    labels_flat = [l for labels in all_labels for l in labels]
    
    # 计算各项指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_flat, 
        predictions_flat, 
        average=None, 
        labels=[0, 1, 2]  # O, B-LOC, I-LOC
    )
    
    # 创建混淆矩阵
    conf_matrix = confusion_matrix(labels_flat, predictions_flat)
    
    # 计算每个类别的指标
    metrics = {
        'O': {'precision': precision[0], 'recall': recall[0], 'f1': f1[0]},
        'B-LOC': {'precision': precision[1], 'recall': recall[1], 'f1': f1[1]},
        'I-LOC': {'precision': precision[2], 'recall': recall[2], 'f1': f1[2]},
        'confusion_matrix': conf_matrix
    }
    
    # 计算总体指标
    metrics['macro_avg'] = {
        'precision': np.mean(precision),
        'recall': np.mean(recall),
        'f1': np.mean(f1)
    }
    
    return metrics

def evaluate_model(model, dataloader, config):
    """评估模型性能"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            # 获取预测结果
            predictions = model(input_ids, attention_mask=attention_mask)
            
            # 只使用有效的标签（去除padding）
            for i, mask in enumerate(attention_mask):
                valid_length = mask.sum().item()
                all_predictions.append(predictions[i][:valid_length])
                all_labels.append(labels[i][:valid_length].cpu().numpy())
    
    # 计算指标
    metrics = calculate_metrics(all_predictions, all_labels)
    return metrics

def print_metrics(metrics):
    """打印评估指标"""
    print("\n=== 评估结果 ===")
    for label in ['O', 'B-LOC', 'I-LOC']:
        print(f"\n{label}:")
        print(f"Precision: {metrics[label]['precision']:.4f}")
        print(f"Recall: {metrics[label]['recall']:.4f}")
        print(f"F1: {metrics[label]['f1']:.4f}")
    
    print("\n总体指标:")
    print(f"Macro Precision: {metrics['macro_avg']['precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_avg']['recall']:.4f}")
    print(f"Macro F1: {metrics['macro_avg']['f1']:.4f}")
    
    print("\n混淆矩阵:")
    print(metrics['confusion_matrix']) 