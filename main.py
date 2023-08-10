import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.feature_extraction import text

from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import gensim.downloader as api
from nltk.corpus import stopwords

import re
import string
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from wordcloud import WordCloud

# alter the width for inspection
pd.set_option('max_colwidth', 150)
# read json into dataframe
df = pd.read_json('10000_ai_ml_dp_nlp.json')

# select main indexes
df = df[['title', 'abstract']]
# set index to title
df = df.set_index('title')


def clean_text_round1(abstract):
    """Remove HTML labels,
    and remove redundant words and line breaks"""
    abstract = re.sub('<[^>]+>', '', str(abstract))
    abstract = re.sub('Abstract', '', str(abstract))
    abstract = re.sub('\n', '', str(abstract))
    return abstract.strip()


# Clean Speech Text
df["abstract"] = df["abstract"].apply(lambda x: clean_text_round1(x))


# Noun extract and lemmatize function
def nouns(abstract):
    """Given a string of text, tokenize the text
    and pull out only the nouns."""
    # create mask to isolate words that are nouns
    is_noun = lambda pos: pos[:2] == 'NN'
    # store function to split string of words
    # into a list of words (tokens)
    tokenized = word_tokenize(abstract)
    # store function to lemmatize each word
    wordnet_lemmatizer = WordNetLemmatizer()
    # use list comprehension to lemmatize all words
    # and create a list of all nouns
    all_nouns = [wordnet_lemmatizer.lemmatize(word) for (word, pos) in pos_tag(tokenized) if is_noun(pos)]

    # return string of joined list of nouns
    return ' '.join(all_nouns)


# Create dataframe of only nouns from speeches
data_nouns = pd.DataFrame(df.abstract.apply(nouns))

stop_noun = ['ai', 'intelligence', 'research', 'technology', 'learning', 'machine', 'technique', 'article',
             'development', 'paper', 'application', 'method', 'deep', 'algorithm', 'problem', 'model']
# Store TF-IDF Vectorizer
tv_noun = TfidfVectorizer(stop_words=stopwords.words('english') + stop_noun, ngram_range=(1, 1), max_df=.8, min_df=.01)
# Fit and Transform speech noun text to a TF-IDF Doc-Term Matrix
data_tv_noun = tv_noun.fit_transform(data_nouns.abstract)
# Create data-frame of Doc-Term Matrix with nouns as column names
data_dtm_noun = pd.DataFrame(data_tv_noun.toarray(), columns=tv_noun.get_feature_names_out())
# Set President's Names as Index
data_dtm_noun.index = df.index
# Visually inspect Document Term Matrix
data_dtm_noun.head()


def display_topics(model, feature_names, num_top_words, topic_names=None):
    # iterate through topics in topic-term matrix, 'H' aka
    # model.components_
    for ix, topic in enumerate(model.components_):
        # print topic, topic number, and top words
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '", topic_names[ix], "'")
        print(", ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))


nmf_model = NMF(14)
# Learn an NMF model for given Document Term Matrix 'V'
# Extract the document-topic matrix 'W'
doc_topic = nmf_model.fit_transform(data_dtm_noun)
# Extract top words from the topic-term matrix 'H'
display_topics(nmf_model, tv_noun.get_feature_names_out(), 5)


# # 1. 话题分布的可视化
# def plot_topic_distribution(doc_topic):
#     topic_dist = doc_topic.sum(axis=0)
#     plt.bar(range(len(topic_dist)), topic_dist)
#     plt.title("Topic Distribution")
#     plt.xlabel("Topic Index")
#     plt.ylabel("Number of Documents")
#     plt.show()
#
#
# plot_topic_distribution(doc_topic)
# # 使用肘部法则来确定最佳话题数量
# def find_optimal_topics(data_dtm, n_topics_range):
#     errors = []
#     for n_topics in n_topics_range:
#         nmf_model = NMF(n_topics)
#         nmf_model.fit(data_dtm)
#         # 通常，NMF的误差可以通过模型的reconstruction_err_属性获得
#         error = nmf_model.reconstruction_err_
#         errors.append(error)
#
#     return errors
#
# # 设定话题数量范围，如从2到20
# n_topics_range = range(2, 21)
# errors = find_optimal_topics(data_dtm_noun, n_topics_range)

# # 可视化肘部法则
# def plot_elbow(n_topics_range, errors):
#     plt.figure()
#     plt.plot(n_topics_range, errors, marker='o')
#     plt.title('Elbow Method for Optimal Topics')
#     plt.xlabel('Number of Topics')
#     plt.ylabel('Reconstruction Error')
#     plt.grid(True)
#     plt.show()
#
#
# plot_elbow(n_topics_range, errors)

# 2. 使用t-SNE对文档-话题矩阵进行降维，并在二维空间中可视化
# def visualize_docs_tsne(doc_topic):
#     tsne = TSNE(n_components=2, random_state=42)
#     tsne_embedding = tsne.fit_transform(doc_topic)
#
#     plt.figure(figsize=(10, 10))
#     plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1])
#     plt.title("t-SNE visualization of documents")
#     plt.show()
#
#
# visualize_docs_tsne(doc_topic)
def visualize_docs_tsne_colored(doc_topic):
    # 使用t-SNE对文档-话题矩阵进行降维
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embedding = tsne.fit_transform(doc_topic)

    # 根据最有可能的话题为每个文档分配颜色
    dominant_topic = np.argmax(doc_topic, axis=1)
    colors = plt.cm.jet(np.linspace(0, 1, len(set(dominant_topic))))

    plt.figure(figsize=(10, 10))

    # 为每个话题创建一个散点图
    for topic_num, color in enumerate(colors):
        indices = np.where(dominant_topic == topic_num)
        plt.scatter(tsne_embedding[indices, 0], tsne_embedding[indices, 1], color=color, label=f"Topic {topic_num}",
                    s=60)

    plt.title("t-SNE visualization of documents")
    plt.legend(loc="best")
    plt.show()


visualize_docs_tsne_colored(doc_topic)


# 3. 为每个话题创建词云
def display_wordclouds(model, feature_names, num_top_words):
    for ix, topic in enumerate(model.components_):
        wc = WordCloud(background_color="white", max_words=num_top_words)
        word_dict = {feature_names[i]: topic[i] for i in topic.argsort()[:-num_top_words - 1:-1]}
        wc.generate_from_frequencies(word_dict)
        plt.figure()
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Topic {ix}")
        plt.show()


display_wordclouds(nmf_model, tv_noun.get_feature_names_out(), 10)