import random

def replace_with_synonyms(text, labels):
    """使用同义词替换文本中的某些字符，同时保持标签对应关系"""
    synonyms = {
        '遊': ['游', '遊'],
        '寺': ['寺院', '禪寺', '精舍'],
        '山': ['嶺', '峰', '岳'],
        '江': ['河', '溪', '川'],
        '湖': ['湖', '潭', '池'],
        '城': ['府', '邑', '鎮'],
        '道': ['路', '徑', '街']
    }
    
    new_text = list(text)
    new_labels = labels.copy()  # 创建标签的副本
    
    for i, char in enumerate(text):
        # 确保索引在有效范围内
        if i >= len(labels):
            break
            
        if char in synonyms and labels[i] != 0:  # 只替换地名中的字
            # 随机决定是否替换（50%的概率）
            if random.random() < 0.5:
                replacement = random.choice(synonyms[char])
                # 如果是单字替换
                if len(replacement) == 1:
                    new_text[i] = replacement
                # 如果是多字替换，需要特殊处理
                else:
                    # 插入新字符
                    new_text[i] = replacement[0]
                    for j, c in enumerate(replacement[1:], 1):
                        new_text.insert(i + j, c)
                        # 扩展标签序列
                        new_labels.insert(i + j, 2)  # 将新增的字符标记为地名内部(I-LOC)
    
    return ''.join(new_text), new_labels

def swap_positions(text, labels):
    """调换位置
    例如：将"遊西湖"变为"西湖遊"
    """
    if len(text) < 4:
        return text, labels
        
    # 找到动词+地名的模式
    patterns = ['遊', '至', '過', '游']
    for i, char in enumerate(text[:-2]):
        if char in patterns and labels[i+1] == 1:  # 后面是地名开始
            # 交换动词和地名位置
            new_text = list(text)
            new_labels = list(labels)
            
            # 找到地名结束位置
            end = i + 2
            while end < len(text) and labels[end] == 2:
                end += 1
                
            # 调换位置
            place = list(text[i+1:end])
            new_text[i:end] = place + [char]
            new_labels[i:end] = labels[i+1:end] + [labels[i]]
            
            return ''.join(new_text), new_labels
            
    return text, labels

def add_noise(text, labels):
    """添加噪声
    在非地名部分随机插入常见字符
    """
    noise_chars = ['，', '。', '而', '乃', '即', '又']
    new_text = list(text)
    new_labels = list(labels)
    
    # 在非地名位置随机插入字符
    positions = []  # 先收集所有可能的插入位置
    for i, label in enumerate(labels):
        if label == 0:
            positions.append(i)
    
    # 随机选择10%的位置插入噪声
    num_insertions = max(1, int(len(positions) * 0.1))
    insert_positions = random.sample(positions, min(num_insertions, len(positions)))
    
    # 从后往前插入，避免位置变化影响
    for pos in sorted(insert_positions, reverse=True):
        new_text.insert(pos, random.choice(noise_chars))
        new_labels.insert(pos, 0)
    
    return ''.join(new_text), new_labels

def augment_data(texts, labels):
    """数据增强主函数"""
    augmented_texts = []
    augmented_labels = []
    
    for text, label in zip(texts, labels):
        # 确保文本和标签长度匹配
        if len(text) != len(label):
            print(f"警告：文本和标签长度不匹配，跳过该样本。文本：{text}")
            continue
            
        # 原始数据
        augmented_texts.append(text)
        augmented_labels.append(label)
        
        # 同义词替换
        new_text, new_label = replace_with_synonyms(text, label)
        augmented_texts.append(new_text)
        augmented_labels.append(new_label)
        
        # 添加噪声
        noisy_text, noisy_label = add_noise(text, label)
        augmented_texts.append(noisy_text)
        augmented_labels.append(noisy_label)
    
    return augmented_texts, augmented_labels