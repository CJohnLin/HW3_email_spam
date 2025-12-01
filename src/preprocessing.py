import re, string
def clean_text(text:str):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+","",text)
    text = re.sub(r"\d+","",text)
    text = text.translate(str.maketrans("","",string.punctuation))
    text = re.sub(r"\s+"," ",text).strip()
    return text
