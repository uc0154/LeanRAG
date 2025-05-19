import json
import re
import jieba
import os

def create_if_not_exist(path):
    if not os.path.exists(path):  # 如果目录不存在，递归创建该目录
        os.makedirs(path, exist_ok=True)

def dicts_almost_equal(dict1, dict2, tolerance=1e-6):
    # 比较字典时允许浮动误差
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        value1 = dict1[key]
        value2 = dict2[key]

        # 如果值是列表，逐个元素比较
        if isinstance(value1, list) and isinstance(value2, list):
            if len(value1) != len(value2):
                return False
            for v1, v2 in zip(value1, value2):
                if isinstance(v1, float) and isinstance(v2, float):
                    if abs(v1 - v2) > tolerance:  # 浮动容差
                        return False
                elif v1 != v2:
                    return False
        # 如果值是浮点数，直接比较
        elif isinstance(value1, float) and isinstance(value2, float):
            if abs(value1 - value2) > tolerance:  # 浮动容差
                return False
        # 其他类型的值直接比较
        elif value1 != value2:
            return False
    return True


def custom_lower_fast(s):
    """中英文兼容的小写转换"""
    return s.lower() if s.isascii() else s  # 中文保持原样


def is_word_boundary(text, start, end):
    """自适应中英文词边界检测"""
    # 判断文本是否包含中文（包括扩展CJK字符）
    has_chinese = re.search(r"[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df]", text)

    if has_chinese:
        # 中文模式：使用jieba分词检测词边界
        words = list(jieba.cut(text))
        current_pos = 0
        boundaries = set()

        # 构建词边界集合
        for word in words:
            boundaries.add(current_pos)  # 词开始位置
            boundaries.add(current_pos + len(word))  # 词结束位置
            current_pos += len(word)

        # 检查输入位置是否在分词边界上
        return start in boundaries or end in boundaries
    else:
        # 英文模式：使用正则表达式检测单词边界
        word_chars = r"\w"  # 仅字母、数字、下划线

        # 前字符检查
        prev_is_word = False
        if start > 0:
            prev_char = text[start - 1]
            prev_is_word = re.match(f"[{word_chars}]", prev_char, re.UNICODE)

        # 后字符检查
        next_is_word = False
        if end < len(text):
            next_char = text[end]
            next_is_word = re.match(f"[{word_chars}]", next_char, re.UNICODE)

        return not prev_is_word and not next_is_word


def read_jsonl(file_path):
    """
    读取jsonl文件，并返回包含每行JSON对象的列表。
    
    :param file_path: .jsonl文件的路径
    :return: 包含每行JSON对象的列表
    """
    data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 去除空行
                if line.strip():
                    json_obj = json.loads(line.strip())  # 解析每一行的JSON对象
                    data.append(json_obj)
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def write_jsonl(data, path, mode="a",encoding='utf-8'):
    with open(path, mode, encoding=encoding) as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
def write_jsonl_force(data, path, mode="w+",encoding='utf-8'):
    with open(path, mode, encoding=encoding) as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

