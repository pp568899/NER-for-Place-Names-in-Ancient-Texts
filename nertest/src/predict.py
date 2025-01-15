import torch
from transformers import BertTokenizer
from models.bert_bilstm_crf import BertBiLSTMCRF
from config import Config
import os
import pandas as pd
import json
from datetime import datetime

def load_model(config):
    """加载训练好的模型"""
    model = BertBiLSTMCRF(
        config.bert_model_name,
        config.num_tags,
        hidden_size=config.hidden_size,
        lstm_hidden_size=config.lstm_hidden_size,
        num_lstm_layers=config.num_lstm_layers,
        dropout_rate=config.dropout_rate
    )
    
    # 添加错误处理
    try:
        model.load_state_dict(torch.load(config.model_save_path))
    except FileNotFoundError:
        raise FileNotFoundError(f"模型文件未找到：{config.model_save_path}")
    except Exception as e:
        raise Exception(f"加载模型时发生错误：{str(e)}")
        
    model.to(config.device)
    model.eval()
    return model

def predict_entities(text, model, tokenizer, config):
    """预测单个文本中的实体"""
    encoding = tokenizer(
        text,
        max_length=config.max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(config.device)
    attention_mask = encoding['attention_mask'].to(config.device)
    
    # 预测
    with torch.no_grad():
        predictions = model(input_ids, attention_mask=attention_mask)
    
    # 打印详细的预测信息
    pred_labels = predictions[0][:len(text)]
    print("\n预测详情:")
    for i, (char, label) in enumerate(zip(text, pred_labels)):
        label_name = "O" if label == 0 else "B-LOC" if label == 1 else "I-LOC"
        print(f"字符: {char}, 预测标签: {label_name} ({label})")
    
    # 提取实体
    entities = []
    current_entity = None
    
    for i, label_id in enumerate(pred_labels):
        if label_id == 1:  # B-LOC
            if current_entity:
                entities.append(current_entity)
            current_entity = {'text': text[i], 'start': i, 'end': i + 1}
        elif label_id == 2 and current_entity:  # I-LOC
            current_entity['text'] += text[i]
            current_entity['end'] = i + 1
        elif current_entity:
            entities.append(current_entity)
            current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities

def process_excel_file(file_path, model, tokenizer, config):
    """处理Excel文件并输出预测结果"""
    try:
        # 读取Excel文件
        print(f"正在读取Excel文件: {file_path}")
        df = pd.read_excel(file_path)
        
        # 确保"事件"列存在
        if "事件" not in df.columns:
            raise ValueError("Excel文件中未找到'事件'列")
        
        # 存储结果的列表
        results = []
        
        # 处理每一行
        for index, row in df.iterrows():
            text = str(row["事件"])
            if pd.isna(text) or text.strip() == "":
                continue
                
            print(f"\n处理第 {index + 1} 行:")
            print(f"原文: {text}")
            
            # 预测实体
            entities = predict_entities(text, model, tokenizer, config)
            
            # 在原文中标记地名
            marked_text = list(text)
            for entity in reversed(entities):
                marked_text.insert(entity['end'], '】')
                marked_text.insert(entity['start'], '【')
            marked_text = ''.join(marked_text)
            
            # 收集该行的结果
            result = {
                "行号": index + 1,
                "原文": text,
                "标记文本": marked_text,
                "识别地名": [entity['text'] for entity in entities]
            }
            results.append(result)
            
            print(f"标记后文本: {marked_text}")
            print(f"识别的地名: {', '.join(result['识别地名'])}")
        
        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(file_path), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存JSON结果
        json_output = os.path.join(output_dir, f"地名识别结果_{timestamp}.json")
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 创建新的DataFrame并保存为Excel
        excel_output = os.path.join(output_dir, f"地名识别结果_{timestamp}.xlsx")
        output_df = pd.DataFrame(results)
        output_df.to_excel(excel_output, index=False)
        
        print(f"\n处理完成！")
        print(f"JSON结果已保存至: {json_output}")
        print(f"Excel结果已保存至: {excel_output}")
        
    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")
        raise

def main():
    # 加载配置
    config = Config()
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
    
    # 加载模型
    print("正在加载模型...")
    model = load_model(config)
    print("模型加载成功！")
    
    # 设置输入文件路径
    input_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "足迹20241128.xls")
    
    if not os.path.exists(input_file):
        print(f"错误：找不到输入文件 {input_file}")
        return
    
    # 处理Excel文件
    process_excel_file(input_file, model, tokenizer, config)

if __name__ == '__main__':
    main()
