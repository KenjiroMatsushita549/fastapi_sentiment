from fastapi import FastAPI
from logging import getLogger
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import functools
from ja_sentence_segmenter.common.pipeline import make_pipeline
from ja_sentence_segmenter.concatenate.simple_concatenator import concatenate_matching
from ja_sentence_segmenter.normalize.neologd_normalizer import normalize
from ja_sentence_segmenter.split.simple_splitter import split_newline, split_punctuation


app = FastAPI()

# Plutchikの8つの基本感情
emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', "Disgust", 'Trust']
emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']

# 使用するモデルを指定して、トークナイザとモデルを読み込む
tokenizer = AutoTokenizer.from_pretrained('tohoku-nlp/bert-base-japanese-whole-word-masking', resume_download=None)
model = AutoModelForSequenceClassification.from_pretrained('omatsu/tohoku-nlp_wrime', resume_download=None).to('cpu')

# softmax関数を定義する
def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

# 感情推定モデルを定義する
def analyze_emotion(text: str):
    # 推論モードを有効化
    model.eval()

    # 入力データ変換 + 推論
    tokens = tokenizer(text, truncation=True, return_tensors="pt").to('cpu')
    # tokens.to(model.device)
    preds = model(**tokens)
    prob = np_softmax(preds.logits.cpu().detach().numpy()[0])

    return prob


@app.get("/")
def root():
    return {"message": "Hello World"}


# 推定した確率値を返すAPI
@app.post("/predict")
async def predict(text: str):
    prediction = analyze_emotion(text)

    return prediction.tolist()


# 歌詞データをスクレイピングし、文ごとにリストへ変換する関数を定義する
def scraping_lylick(artist: str, title: str):

    # ウタテンのurlをbase_urlに入力する
    base_url = "https://utaten.com/"

    # サーチするためのURLを入力する
    serch_url = "search?sort=popular_sort_asc&artist_name=" + artist + "&title=" + title + "&beginning=&body=&lyricist=&composer=&sub_title=&tag=&show_artists=1"

    # 指定したURLのHTMLを取得する
    time.sleep(1)
    url = base_url + serch_url
    response = requests.get(url)

    # BeautifulSoupでHTMLの解析を行う
    soup = BeautifulSoup(response.text, "html.parser")

    # 検索した曲のurlを取得する
    sub_urls = soup.find_all("a")
    sub_url_list = []

    for sub_url in sub_urls:
        href = sub_url.get('href')
        # 相対URLを絶対URLに変換します
        absolute_url = urljoin(base_url, href)
        # URLをリストに追加します
        sub_url_list.append(absolute_url)

    sub_url = [s for s in sub_url_list if '/lyric/' in s]

    # 歌詞を取得する
    # 取得した歌詞ページのURLのHTMLを取得する
    time.sleep(1)
    try:
        response = requests.get(sub_url[0])

        # BeautifulSoupで歌詞ページのHTMLの解析を行う
        soup = BeautifulSoup(response.text, "html.parser")

        # 歌詞が入力されているクラスを取得
        div = soup.find("div", class_="hiragana")

        # 不要なクラスを除外する
        for d in div.find_all("span", class_= "rt"):
            d.extract()

        # 歌詞データを取得する
        lylick = div.get_text()
    except IndexError:
        print("IndexError")
    except AttributeError:
        print("AttributeError")

    # よくわからんけどこれで、改行や空白を除外して、文ごとにリストへ変換してくれる
    split_punc2 = functools.partial(split_punctuation, punctuations=r"。!?")
    concat_tail_no = functools.partial(concatenate_matching, former_matching_rule=r"^(?P<result>.+)(の)$", remove_former_matched=False)
    segmenter = make_pipeline(normalize, split_newline, concat_tail_no, split_punc2)

    return list(segmenter(lylick))


# 取得した歌詞を返すAPI
@app.post("/scraping")
async def scraping(artist: str, title: str):
    lylick = scraping_lylick(artist, title)
    
    return lylick
