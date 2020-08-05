from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np

# ローカルディレクトリからHashingVectorizerをインポート
from vectorizer import vect

app = Flask(__name__)


##################
### 分類器の準備 ###
##################
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'reviews.sqlite')

# 文書に対して予測されたクラスラベルとその予測確率を返す
def classify(document):
    label = {0:'negative', 1:'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = clf.predict_proba(X).max()
    return label[y], proba

# 指定された文書とクラスラベルに基づいて分類器を更新
def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])

# 送信された映画レビュー、クラスラベル、タイムスタンプをDBに格納
def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review, sentiment, date)" \
              "VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()


#############
### Flask ###
#############

# TextAreaFieldをインスタンス化
class ReviewForm(Form):
    moviereview = TextAreaField('',
                                [validators.DataRequired(),
                                 validators.length(min=15)])  # レビューは15文字以上でなければならない

# ランディングページreviewform.htmlの表示
@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

# フォームの送信内容を分類器（classify関数）に渡す
@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y, proba = classify(review)
        return render_template('results.html',
                                content=review,
                                prediction=y,
                                probability=round(proba*100, 2))
    return render_template('reviewform.html', form=form)

# ユーザーがフィードバックボタンを押したときの挙動（Incorrectのとき正しい評価に更新）
@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']

    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))  # Incorrectのとき0,1を反転
    train(review, y)  # 分類器を更新
    sqlite_entry(db, review, y)  # 更新内容をDBに反映
    return render_template('thanks.html')

if __name__ == '__main__':
    app.run()