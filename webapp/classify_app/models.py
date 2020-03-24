from django.db import models

import re
import sys
import math
import nagisa
import numpy as np
import urllib.request
from bs4 import BeautifulSoup

# ナイーブベイズ分類器
class NaiveBayes:
  def in_category(self, word, category):
    if word in self.word_ct[category]:
        return float(self.word_ct[category][word])# カテゴリー内の単語出現回数
    return 0.0
    
  def word_prob(self, word, category):
    # カテゴリ内の対象単語の出現率
    # 単語のカテゴリー内出現回数 + 1 / カテゴリー内単語数 + 学習データの全単語数 (加算スムージング)
    prob = (self.in_category(word, category) + 1.0) / (sum(self.word_ct[category].values())+ len(self.vocabularies) * 1.0)
    return prob
        
  def score(self, words, category):
    # カテゴリー出現率P(C)を取得 (アンダーフロー対策で対数をとった後で加算)
    score = math.log(float(self.category_ct[category] / sum(self.category_ct.values())))
      
    # カテゴリー内の単語出現率を文書内のすべての単語で求める
    for word in words:
        # カテゴリー内の単語出現率P(Wn|C)を計算 (アンダーフロー対策で対数をとった後で加算)
        score += math.log(self.word_prob(word, category))
    return score
    
  def classify(self,link):
    # 対象記事の形態素解析
    html = urllib.request.urlopen(link)
    soup = BeautifulSoup(html, 'html.parser')

    text = soup.find_all("p")
    text = [t.text for t in text]
    text = ','.join(text)

    words = nagisa.tagging(text)
    words = nagisa.filter(text, filter_postags=['助詞', '助動詞','接頭辞','接尾辞','補助記号','URL','空白']).words
    words = list(filter(lambda x: len(x) != 1, words)) # 一文字の単語を削除
    words = list(filter(lambda x: x != '', [re.sub(r'[0-9]', "", s) for s in words]))# 数字を削除

    # 学習データをloadする
    self.word_ct = np.load('./classify_app/data/word_ct.npy',allow_pickle=True).tolist()
    self.category_ct = np.load('./classify_app/data/category_ct.npy',allow_pickle=True).tolist()
    self.vocabularies = np.load('./classify_app/data/vocabularies.npy',allow_pickle=True).tolist()

    best_category = None # もっとも近いカテゴリ

    max_prob = -sys.maxsize# 最小整数値を設定

    # カテゴリ毎に対象文書（単語）のカテゴリー出現率P(C|W)を求める
    for category in self.category_ct.keys():
      # 文書内のカテゴリー出現率P(C|W)を求める
      prob = self.score(words, category)
      if prob > max_prob:
        max_prob = prob
        best_category = category

    return best_category