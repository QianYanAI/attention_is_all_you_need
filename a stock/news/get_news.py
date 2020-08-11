import jqdatasdk as jq
import pandas as pd
import jieba
import jieba.analyse
import codecs
from gensim.models import word2vec


def get_stock():
    jq.auth('13345624026', 'WY192837465')
    get_security = jq.get_all_securities(types=['stock'], date=None)
    sec_list_chinese = get_security['display_name'].to_list()
    return sec_list_chinese


def get_news(num, stock):
    data_tickernews = pd.DataFrame()
    segments = []
    for i in range(1, num):
        data = pd.read_csv('/Users/alex/Downloads/'
                           'Sentiment-Analysis-in-Event-Driven-Stock-Price-Movement-Prediction-master/'
                           'a stock/thefile{}.csv'.format(i))
        data = data.dropna(0)
        data_tickernews = pd.concat([data_tickernews, data[data['newsSummary'].str.contains(stock)]], ignore_index=True)
        data_tickernews.drop_duplicates('newsTitle', 'first')
    stopwords = [line.strip() for line in codecs.open('/Users/alex/Downloads/'
                                                      'Sentiment-Analysis-in-Event-Driven-Stock-Price-Movement-Prediction-master/'
                                                      'a stock/news/scu_stopwords.txt', 'r', 'utf-8').readlines()]
    for index, row in data_tickernews.iterrows():
        content = row[2]
        # TextRank 关键词抽取，只获取固定词性
        words = jieba.cut(content, cut_all=False)
        splitedStr = ''
        for word in words:
            # 停用词判断，如果当前的关键词不在停用词库中才进行记录
            if word not in stopwords:
                # 记录全局分词
                splitedStr += word + ' '
                segments.append("".join(list(splitedStr)))
    output_list = pd.Series(segments)
    output_list.to_csv('news.txt', encoding='utf-8')


def word2vec_news(modelpath):
    fileTrainRead = pd.read_csv('news.txt')
    train_sentences = pd.Series(fileTrainRead.iloc[:, 1])
    f = lambda x: str(x).split(" ")
    train_sentences = train_sentences.apply(f)

    model = word2vec.Word2Vec(train_sentences, size=300)
    model.save(modelpath)
    print(model.wv['茅台'])


if __name__ == "__main__":
    # stock_list = get_stock()
    get_news(30, '茅台')
    word2vec_news('model.bin')
