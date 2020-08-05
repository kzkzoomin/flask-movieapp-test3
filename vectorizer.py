from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

cur_dir = os.path.dirname(__file__)

# ストップワード読み込み
stop = pickle.load(open(
                    os.path.join(cur_dir,
                                'pkl_objects',
                                'stopwords.pkl'), 'rb'))

# 前処理用関数
def tokenizer(text):
    # マークアップ削除
    text = re.sub('<[^>]*>', '', text)
    # 絵文字リスト
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    # 余計な記号を空白に置換、絵文字リストを結合、絵文字の鼻を削除で統一
    text = re.sub('[\W]+', ' ', text.lower()) \
        + ' '.join(emoticons).replace('-', '')
    # ストップワードを除外してトークン化
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)