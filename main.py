import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from wordcloud import WordCloud

# alter the width for inspection
pd.set_option('max_colwidth', 150)
# read json into dataframe
df = pd.read_json('10000_ai_ml_dp_nlp.json')

# select main indexes
df = df[['title', 'abstract', '_id']]
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

stop_noun = ['ai', 'artificial', 'intelligence', 'research', 'technology', 'learning', 'machine', 'technique',
             'article', 'development', 'paper', 'application', 'method', 'deep', 'algorithm', 'problem', 'model',
             'approach', 'data', 'set', 'science', 'industry', 'ml', 'dl', 'field', 'use', 'study', 'analysis',
             'amp', 'gt', 'lt', 'us', 'nan', 'x0d', 'concept', 'task', 'issue', 'computer', 'knowledge', 'management',
             'information', 'author', 'solution', 'design', 'performance', 'result', 'search', 'function', 'process',
             'impact', 'challenge', 'review', 'agent', 'framework']
# Store TF-IDF Vectorizer
tv_noun = TfidfVectorizer(stop_words=stopwords.words('english') + stop_noun, ngram_range=(1, 2), max_df=.8, min_df=.01)
# Fit and Transform speech noun text to a TF-IDF Doc-Term Matrix
data_tv_noun = tv_noun.fit_transform(data_nouns.abstract)
# Create data-frame of Doc-Term Matrix with nouns as column names
data_dtm_noun = pd.DataFrame(data_tv_noun.toarray(), columns=tv_noun.get_feature_names_out())
# Set President's Names as Index
data_dtm_noun.index = df.index
# Visually inspect Document Term Matrix
data_dtm_noun.head()


def display_topics(model, feature_names, num_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        topic_words = [(feature_names[i], topic[i]) for i in topic.argsort()[:-num_top_words * 3 - 1:-1]]

        # Check for duplicates and sub-strings in bigrams
        to_remove = set()
        for i, (word_i, _) in enumerate(topic_words):
            components_i = word_i.split()

            # remove exact duplicates like "system system"
            if len(components_i) == 2 and components_i[0] == components_i[1]:
                to_remove.add(word_i)
                continue

            for j, (word_j, _) in enumerate(topic_words):
                if i != j:
                    if len(components_i) == 1 and word_i in word_j:
                        # if word_i is a substring of word_j and word_i is not a bigram, then word_i should be removed
                        to_remove.add(word_i)

        # 只保存那些没有被标记为移除的单词
        topic_words = [(word, importance) for word, importance in topic_words if word not in to_remove]

        # 最后，为了确保输出的单词数量，我们再次筛选
        top_words = sorted(topic_words, key=lambda x: x[1], reverse=True)[:num_top_words]
        words = [word for word, _ in top_words]

        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '", topic_names[ix], "'")
        print(", ".join(words))


nmf_model = NMF(13)
# Learn an NMF model for given Document Term Matrix 'V'
# Extract the document-topic matrix 'W'
doc_topic = nmf_model.fit_transform(data_dtm_noun)
# Extract top words from the topic-term matrix 'H'
display_topics(nmf_model, tv_noun.get_feature_names_out(), 5)


# 使用t-SNE对文档-话题矩阵进行降维，并在二维空间中可视化
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
        topic_words = [(feature_names[i], topic[i]) for i in topic.argsort()[:-num_top_words * 3 - 1:-1]]

        # Check for duplicates and sub-strings in bigrams
        to_remove = set()
        for i, (word_i, _) in enumerate(topic_words):
            components_i = word_i.split()

            # remove exact duplicates like "system system"
            if len(components_i) == 2 and components_i[0] == components_i[1]:
                to_remove.add(word_i)
                continue

            for j, (word_j, _) in enumerate(topic_words):
                if i != j:
                    if len(components_i) == 1 and word_i in word_j:
                        # if word_i is a substring of word_j and word_i is not a bigram, then word_i should be removed
                        to_remove.add(word_i)

        # 只保存那些没有被标记为移除的单词
        topic_words = [(word, importance) for word, importance in topic_words if word not in to_remove]

        # 为了确保单词数量，我们再次筛选
        top_words = sorted(topic_words, key=lambda x: x[1], reverse=True)[:num_top_words]
        word_dict = {word: importance for word, importance in top_words}

        wc = WordCloud(background_color="white", max_words=num_top_words)
        wc.generate_from_frequencies(word_dict)
        plt.figure()
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Topic {ix}")
        plt.show()


display_wordclouds(nmf_model, tv_noun.get_feature_names_out(), 10)


def topic_to_keywords(model, feature_names, num_top_words):
    topic_map = {}
    for ix, topic in enumerate(model.components_):
        topic_words = [(feature_names[i], topic[i]) for i in topic.argsort()[:-num_top_words * 3 - 1:-1]]

        # Check for duplicates and sub-strings in bigrams
        to_remove = set()
        for i, (word_i, _) in enumerate(topic_words):
            components_i = word_i.split()

            # remove exact duplicates like "system system"
            if len(components_i) == 2 and components_i[0] == components_i[1]:
                to_remove.add(word_i)
                continue

            for j, (word_j, _) in enumerate(topic_words):
                if i != j:
                    if len(components_i) == 1 and word_i in word_j:
                        # if word_i is a substring of word_j and word_i is not a bigram, then word_i should be removed
                        to_remove.add(word_i)

        # 只保存那些没有被标记为移除的单词
        topic_words = [(word, importance) for word, importance in topic_words if word not in to_remove]

        # 最后，为了确保输出的单词数量，我们再次筛选
        top_words = sorted(topic_words, key=lambda x: x[1], reverse=True)[:num_top_words]
        words = [word for word, _ in top_words]
        topic_map[ix] = ' - '.join(words)
    return topic_map


#
#
# topic_keywords_map = topic_to_keywords(nmf_model, tv_noun.get_feature_names_out(), 10)
# df['dominant_topic'] = np.argmax(doc_topic, axis=1)
# df['dominant_topic_keywords'] = df['dominant_topic'].apply(lambda x: topic_keywords_map[x])
# df_out = df.reset_index()[['_id', 'dominant_topic_keywords']]
# df_out.columns = ['_id', 'topic']
# df_out.to_csv('topics_output.csv', index=False)


df['dominant_topic'] = np.argmax(doc_topic, axis=1)
topic_counts = df['dominant_topic'].value_counts().reset_index()
topic_counts.columns = ['Topic', 'Num_Documents']
topic_keywords_map = topic_to_keywords(nmf_model, tv_noun.get_feature_names_out(), 5)
topic_counts['Topic_Keywords'] = topic_counts['Topic'].apply(lambda x: topic_keywords_map[x])

# 绘制条形图
plt.figure(figsize=(15, 10))
plt.barh(topic_counts['Topic_Keywords'], topic_counts['Num_Documents'], color='skyblue')
plt.xlabel('Number of Documents', fontsize=14)
plt.ylabel('Topics', fontsize=14)
plt.title('Number of Documents per Topic', fontsize=16)
plt.gca().invert_yaxis()  # 这样第一个话题将会在顶部显示
plt.show()
