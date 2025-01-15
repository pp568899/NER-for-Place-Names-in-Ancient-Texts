import warnings
import torch
import spacy
import pandas as pd
import re
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

class AncientLocationExtractor:
    def __init__(self):
        try:
            try:
                self.nlp = spacy.load("zh_core_web_trf")
            except:
                self.nlp = spacy.load("zh_core_web_sm")
            print("成功加载中文模型")
        except OSError:
            print("错误：未能加载中文模型，请确保已经运行：python -m spacy download zh_core_web_sm")
            raise

    def _is_location_by_pattern(self, text):
        """基于语言模式判断是否是地名"""
        # 地名后缀特征
        location_suffixes = {
            '州', '府', '县', '郡', '国', '城', '关', '山', '水', '河', '湖', '洞', 
            '寺', '观', '庙', '桥', '池', '亭', '楼', '堂', '阁', '斋', '堡', '塘', 
            '峰', '岭', '台', '园', '宫', '院', '坊', '巷', '路', '街', '门'
        }
        
        # 地名前缀词
        location_prefixes = {
            '东', '西', '南', '北', '上', '下', '内', '外', '前', '后',
            '大', '小', '新', '旧', '左', '右'
        }
        
        # 方位词和介词
        position_words = {
            '至', '在', '往', '从', '自', '到', '去', '经', '由',
            '赴', '抵', '出', '入', '过'
        }

        # 1. 检查是否以地名后缀结尾
        for suffix in location_suffixes:
            if text.endswith(suffix) and len(text) > len(suffix):
                return True

        # 2. 检查是否有典型的地名前缀
        for prefix in location_prefixes:
            if text.startswith(prefix) and len(text) > len(prefix):
                return True

        # 3. 检查上下文模式（如果有上下文）
        context_pattern = f"({'|'.join(position_words)})[一-龥]{{1,5}}"
        if re.search(context_pattern, text):
            return True

        return False

    def _clean_location(self, text):
        """清理和标准化地名"""
        # 移除非地名部分
        remove_patterns = [
            r'^於', r'^有書致', r'^至', r'^往', r'^在', r'^居', r'^自',
            r'^遂有', r'^經', r'^登', r'^訪', r'^遊', r'^道', r'^抵',
            r'^開館於', r'^設席於', r'^窆王越於', r'^作遊', r'^闢', r'^請築',
            r'^集門人於', r'之$', r'也$', r'矣$', r'上$', r'中$', r'內$', r'外$'
        ]
        
        result = text
        for pattern in remove_patterns:
            result = re.sub(pattern, '', result)
        
        # 如果清理后长度太短，可能不是地名
        if len(result) < 2 and not self._is_special_location(result):
            return None
            
        return result if self._is_location_by_pattern(result) else None

    def _is_special_location(self, text):
        """判断是否是特殊的地名（如单字地名）"""
        # 单字地名的上下文模式
        if len(text) == 1:
            return text in {'京', '都', '塘', '山', '湖'}
        return False

    def _extract_by_context(self, text):
        """基于上下文提取地名"""
        locations = set()
        
        # 1. 基于位置词提取
        position_patterns = [
            r'[至在往从自到去经由赴抵出入过][一-龥]{1,5}(?:[州府县郡国城关山水河湖洞寺观庙])',
            r'[一-龥]{1,5}(?:[州府县郡国城关山水河湖洞寺观庙])[之中]?',
            r'(?:東|西|南|北|上|下|内|外|大|小|新|旧)[一-龥]{1,4}',
        ]
        
        for pattern in position_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                loc = self._clean_location(match.group())
                if loc:
                    locations.add(loc)
        
        return locations

    def extract_locations(self, text):
        """提取文本中的地名"""
        locations = set()
        
        # 1. 使用上下文规则提取
        context_locations = self._extract_by_context(text)
        locations.update(context_locations)
        
        # 2. 使用 spaCy 模型识别
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == 'LOC':
                cleaned_loc = self._clean_location(ent.text)
                if cleaned_loc:
                    locations.add(cleaned_loc)

        # 3. 处理特殊情况
        for word in text.split():
            if self._is_special_location(word):
                locations.add(word)

        return sorted(list(locations))

def process_excel_sample(file_path, sample_size=50):
    """处理Excel文件的前N行"""
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        
        # 先去重
        df = df.drop_duplicates(subset=['事件'])
        
        # 重置索引，确保索引连续
        df = df.reset_index(drop=True)
        
        # 再取前sample_size行
        df = df.head(sample_size)
        
        extractor = AncientLocationExtractor()
        
        results = []
        for idx, row in df.iterrows():
            event_text = str(row['事件'])
            locations = extractor.extract_locations(event_text)
            result = {
                '序号': len(results) + 1,  # 使用结果列表长度+1作为序号
                '原文': event_text,
                '识别的地名': '、'.join(locations) if locations else '无'
            }
            results.append(result)
            print(f"\n序号 {result['序号']}:")
            print(f"原文: {event_text}")
            print(f"识别的地名: {result['识别的地名']}")
            
        # 将结果保存到新的Excel文件
        output_df = pd.DataFrame(results)
        output_path = Path(file_path).parent / '地名提取结果_样本.xlsx'
        output_df.to_excel(output_path, index=False)
        print(f"\n结果已保存到：{output_path}")
        print(f"\n总共处理了 {len(results)} 条记录")
        
    except Exception as e:
        print(f"处理文件时发生错误：{str(e)}")

if __name__ == "__main__":
    file_path = "足迹20241128.xls"
    process_excel_sample(file_path, 50)
