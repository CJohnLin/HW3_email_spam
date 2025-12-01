import re
import string

def clean_text(text: str):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # 移除網址
    text = re.sub(r"\d+", "", text)              # 移除數字
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)             # 多空白變一個
    return text.strip()
