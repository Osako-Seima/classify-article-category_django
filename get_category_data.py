import re
import time
import sys
import math
import nagisa
import numpy as np
import urllib.request
from bs4 import BeautifulSoup
from tqdm import tqdm

vocabularies = set() # 学習データの全単語の集合(加算スムージング用)
word_ct = {} # 学習データのカテゴリー毎の単語数セット用
category_ct = {} # 学習データのカテゴリー毎の記事数セット用

# 対象記事の形態素解析メソッド
def Morphological_analysis(link):
    html = urllib.request.urlopen(link)
    soup = BeautifulSoup(html, 'html.parser')

    text = soup.find_all("p")
    text = [t.text for t in text]
    text = ','.join(text)

    words = nagisa.tagging(text)
    words = nagisa.filter(text, filter_postags=['助詞', '助動詞','接頭辞','接尾辞','補助記号','URL','空白']).words
    words = list(filter(lambda x: len(x) != 1, words)) # 一文字の単語を削除
    words = list(filter(lambda x: x != '', [re.sub(r'[0-9]', "", s) for s in words]))# 数字を削除
    
    return words

def word_count_up(word, category):
    word_ct.setdefault(category, {}) # 新カテゴリーなら追加
    word_ct[category].setdefault(word, 0) # カテゴリー内で新単語なら追加
    word_ct[category][word] += 1 # カテゴリー内の単語出現回数をカウント
    vocabularies.add(word) # 学習データの全単語集合に加える(重複排除)

def category_count_up(category):
    category_ct.setdefault(category, 0) # 新カテゴリーなら追加
    category_ct[category] += 1 # カテゴリー記事数をカウント

# 検証用データの空のリスト
verification_link = []
verification_category = []

for i, category in tqdm(zip([1,2,3,4,5,6,7,8],['エンタメ','スポーツ','おもしろ','国内','海外','コラム','IT・科学','グルメ']), desc='各カテゴリの記事収集'):
    for j in tqdm(range(1,6), desc=f'{category}'):
        # 最後のページに遷移した時に表示されたリンクを検証データとする。
        if j == 5:
            url = f'https://gunosy.com/categories/{i}?page={j}'
            html = urllib.request.urlopen(url)
            soup = BeautifulSoup(html, 'html.parser')

            # 各カテゴリのリンクを収集
            elements = soup.find_all('div', class_="list_title")
            link_list = [element.find('a')['href'] for element in elements]
            for link in link_list:
                verification_link.append(link)
                verification_category.append(category)

        else:
            url = f'https://gunosy.com/categories/{i}?page={j}'
            html = urllib.request.urlopen(url)
            soup = BeautifulSoup(html, 'html.parser')
            elements = soup.find_all('div', class_="list_title")
            link_list = [element.find('a')['href'] for element in elements]

            for link in link_list:
                words = Morphological_analysis(link)
                for word in words:
                    # カテゴリー内の単語出現回数をカウント
                    word_count_up(word, category)
                # カテゴリーの記事数をカウント
                category_count_up(category)
    
    time.sleep(1)

# 学習データ用のカテゴリごとの単語データ基盤
np.save('word_ct.npy',word_ct)
np.save('category_ct.npy',category_ct)
np.save('vocabularies.npy',vocabularies)

# 検証用データ
verification_data = dict(zip(verification_link, verification_category))
np.save('verification_data.npy', verification_data)