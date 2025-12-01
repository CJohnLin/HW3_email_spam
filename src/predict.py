import argparse
import joblib
import pandas as pd
from src.preprocessing import clean_text

def predict_single(model_path, text):
    model = joblib.load(model_path)
    clean = clean_text(text)
    pred = model.predict([clean])[0]
    prob = None
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba([clean])[0][1]
    return pred, prob

def predict_csv(model_path, csv_path, out_path='prediction_output.csv'):
    model = joblib.load(model_path)
    df = pd.read_csv(csv_path)
    if 'text' not in df.columns:
        raise ValueError('CSV must contain text column')
    df['text_clean'] = df['text'].astype(str).apply(clean_text)
    df['pred'] = model.predict(df['text_clean'])
    if hasattr(model, 'predict_proba'):
        df['spam_prob'] = model.predict_proba(df['text_clean'])[:,1]
    df.to_csv(out_path, index=False)
    return out_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--text', help='single text to classify')
    parser.add_argument('--csv', help='csv file with text column for batch predict')
    args = parser.parse_args()
    if args.text:
        p,prob = predict_single(args.model, args.text)
        print('pred',p,'prob',prob)
    if args.csv:
        out = predict_csv(args.model, args.csv)
        print('saved',out)
