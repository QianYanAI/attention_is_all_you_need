import jqdatasdk as jq
import pandas as pd
import jieba
import jieba.analyse
import codecs


def get_stock():
    jq.auth('', '')
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
        words = jieba.cut(content)
        splitedStr = ''
        for word in words:
            # 停用词判断，如果当前的关键词不在停用词库中才进行记录
            if word not in stopwords:
                # 记录全局分词
                segments.append({'word': word, 'count': 1})
                splitedStr += word + ' '
    dfSg = pd.DataFrame(segments)
    # 词频统计
    dfWord = dfSg.groupby('word')['count'].sum()
    return dfWord


if __name__ == "__main__":
    stock_list = get_stock()
    dfword = get_news(10, '茅台')
    print(dfword)
