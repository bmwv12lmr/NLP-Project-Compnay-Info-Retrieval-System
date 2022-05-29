'''
CMPE-257 Final Exam
Allen Wu 015292667 yanshiun.wu@sjsu.edu

Requirements:
1. Get access to crunchbase API
    Use BeautifulSoup to scrape crunchbase instead of crunchbase API.
    Due to unavailability of crunchbase API since crunchbase didn't asssign the API Key for me after filing the request for a long tiem.

2. accept the name of a company
    Please type the company name in variable "COMPANY_NAME" below

3. search for the company on crunchbase
    Use function "get_url" to get the page of company on crunchbase

4. search for company in the news
    Use function "get_google_news" can perform news searching and return news list

5. POS and NER for building a graph of relationships
    Use POS to find the cos sim form company's webpage and use NER to find the entities from company and keymans' news to build relationship in grape

6. lookup the main people (NER) from crunchbase using search api (e.g., google search api)
    Use NER to find the entities from keymans' Google News Search

7. get top topics in the news about the company
    Use BERTopic to find the topic of company news and add it to the graph

8. show graph using networkx api in python
    Use networkx API to build graph in function plot_company_info and save the graph in IMG folder with COMPANY_NAME.jpg

9. deposit a text file containing research about that company
    Use function output_company_info to save the info in RPT folder with COMPANY_NAME.txt

PLEASE NOTE: If you want to scrape data from CrunchBase, you MUST ENABLE variable "scrape_mode" to avoid blocked by crunchbase anti-bot tools

'''


# User-Defined Company
company_list = ["giftpack"]
COMPANY_NAME = company_list[0]

# MUST ENABLE if you want to scrape from crunchbase
scrape_mode = False

import gensim
import requests
import spacy as spacy
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from xml.etree.ElementTree import XML
import pandas as pd
import os
import hashlib
import time
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as bs
import sys
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer, util
from spacy import displacy
from transformers import pipeline
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from gensim.utils import simple_preprocess
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Line Debugging
def line():
    return sys._getframe(1).f_lineno

# Enable to turn on debug mode
debug_mode = False

# Generate Hashed TimeStamp
def get_hash_timestamp():
    hash_stamp = hashlib.sha1()
    hash_stamp.update(str(time.time()).encode('utf-8'))
    return hash_stamp.hexdigest()[:10]

# Pre-Define Parameters
WORKING_DIR = os.getcwd() + "/"
CHROMEDRIVER_PATH = WORKING_DIR + 'chromedriver'
CRUNCHBASE_SITE = "www.crunchbase.com" + "%2F" + "organization" + "%2F"
LEN_THRESHOLD = 20
SIM_THRESHOLD = 0.8

if scrape_mode:
    SLEEP = 20
else:
    SLEEP = 2


# Selenium Parameters
def get_browser_option():
    browser = Service(CHROMEDRIVER_PATH)
    option = webdriver.ChromeOptions()
    option.add_argument('--disable-blink-features=AutomationControlled')
    option.add_argument("window-size=1280,800")
    option.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/74.0.3729.169 Safari/537.36")
    if scrape_mode != True:
        option.add_argument("--headless")
    return browser, option


# Get the src of web
def get_src(url_list):
    chrome = get_browser_option()
    driver = webdriver.Chrome(service=chrome[0], options=chrome[1])
    if scrape_mode and len(url_list) != 0:
        driver.get(url_list[0])
        time.sleep(SLEEP)
    src_list = list()
    for url in url_list:
        file = WORKING_DIR + "WEB/" + url.replace("https://", "").replace("/", "_") + ".pkl"
        source_code = None
        if os.path.exists(file):
            with open(file, "rb") as fp:
                source_code = pickle.load(fp)
            src_list.append(source_code)
            continue
        driver.get(url)
        time.sleep(2)
        if driver.current_url == url:
            source_code = driver.page_source
            with open(file, "wb") as fp:
                pickle.dump(source_code, fp)
        src_list.append(source_code)
    return src_list


# Get the Target URL from Search Engine
def get_url(company_name, site=""):
    if site != "":
        site = "site%3A" + site + "+"
    url = "https://duckduckgo.com/?q=%5C" + site + company_name
    chrome = get_browser_option()
    driver = webdriver.Chrome(service=chrome[0], options=chrome[1])
    driver.get(url)
    time.sleep(2)
    return driver.current_url


# Get the RSS and Save it
def get_rss(url):
    file = WORKING_DIR + 'RSS/' + get_hash_timestamp() + '.xml'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    resp = requests.get(url)
    with open(file, 'wb') as f:
        f.write(resp.content)
    return file


# Parse Google RSS from file
def parse_google_rss(file):
    xml_data = open(file, 'r').read()  # Read file
    root = XML(xml_data)  # Parse XML
    data = list()
    for i, root_1 in enumerate(root):
        for j, child in enumerate(root_1):
            if child.tag != 'item':
                continue
            data.append([sub_child.text for sub_child in child])
    return pd.DataFrame(data, columns=['title', 'link', 'guid', 'pubDate', 'description', 'source'])


# Get Google News Titles and Links
def get_google_news(keyword):
    google_rss_url = "https://news.google.com/rss/search?q=%22" + keyword + "%22"
    df = parse_google_rss(get_rss(google_rss_url))[['title', 'link']]
    return df


# bs object iterator for debug
def obj_iter(obj):
    for i in range(len(obj)):
        print(obj[i].text)

# Scrape crunchbase data
def get_crunchbase_information(crunchbase_url):
    crunchbase_summary_url = crunchbase_url
    crunchbase_keyman_url = crunchbase_url + "/people"
    crunchbase_finance_url = crunchbase_url + "/company_financials"
    url_list = [crunchbase_summary_url, crunchbase_keyman_url, crunchbase_finance_url]
    src_list = get_src(url_list)

    summary = None
    if src_list[0] is not None:
        soup = bs(src_list[0], 'lxml')

        web_link = None
        try:
            web_link = soup.find('a', class_='component--field-formatter link-accent ng-star-inserted').text
        except:
            web_link = None

        about = None
        try:
            about = soup.find('span', class_='description').text
        except:
            about = None

        industry_list = list()
        try:
            tag_list = soup.find_all('div', class_='cb-overflow-ellipsis')
            for tag in tag_list:
                industry_list.append(tag.text)
        except:
            industry_list = None
        if len(industry_list) == 0:
            industry_list = None
        summary = [web_link, about, industry_list]
        if web_link == None and about == None and industry_list == None:
            summary = None

    keyman = None
    if src_list[1] is not None:
        soup = bs(src_list[1], 'lxml')
        keyman_df = pd.DataFrame(columns=['title', 'full_name', 'news'])
        try:
            tag_list = soup.find_all('li', class_='ng-star-inserted')
            for tag in tag_list:
                name = tag.find('a').text
                try:
                    title = tag.find('span',
                                     class_="component--field-formatter field-type-text_short ng-star-inserted").text
                except:
                    title = tag.find('span',
                                     class_="component--field-formatter field-type-enum ng-star-inserted").text
                news = get_google_news(name).title
                keyman_df = pd.concat(
                    [keyman_df,
                     pd.DataFrame.from_records([{'title': title, 'full_name': name, 'news': news}])]).reset_index(
                    drop=True)
        except:
            keyman_df = pd.DataFrame(columns=['title', 'full_name', 'news'])
        keyman = keyman_df

    finance = None
    if src_list[2] is not None:
        soup = bs(src_list[2], 'lxml')
        investor_list = list()
        try:
            card_list = soup.find_all('row-card', class_='ng-star-inserted')
            for card in card_list:
                if card.find('h2').text != "Investors":
                    continue
                row_list = card.find_all('tr', class_="ng-star-inserted")
                for row in row_list:
                    investor_list.append(row.find('td', class_="ng-star-inserted").text)
        except:
            investor_list = None
        if len(investor_list) == 0:
            investor_list = None

        partner_list = list()
        try:
            card_list = soup.find_all('row-card', class_='ng-star-inserted')
            for card in card_list:
                if card.find('h2').text != "Investors":
                    continue
                row_list = card.find_all('tr', class_="ng-star-inserted")
                for row in row_list:
                    parner = row.find_all('td', class_="ng-star-inserted")[3].text
                    if parner != " — ":
                        partner_list.append(parner)
        except:
            partner_list = None
        if len(partner_list) == 0:
            partner_list = None

        round_type = None
        try:
            round_type = soup.find(
                class_='component--field-formatter field-type-enum link-accent ng-star-inserted').text
        except:
            round_type = None

        fund_raised = None
        try:
            fund_raised = soup.find(
                class_="component--field-formatter field-type-money link-accent ng-star-inserted").text
        except:
            fund_raised = None

        round_num = None
        try:
            card_list = soup.find_all('a', class_='link-primary')
            for card in card_list:
                title = card.find('span', class_="wrappable-label-with-info ng-star-inserted").text
                if title != "Funding Rounds ":
                    continue
                round_num = card.find('span',
                                      class_="component--field-formatter field-type-integer ng-star-inserted").text
        except:
            round_num = None
        finance = [round_num, round_type, fund_raised, partner_list, investor_list]
        if round_num == None and round_type == None and fund_raised == None and partner_list == None and investor_list == None:
            finance = None
    return summary, keyman, finance

# Generate Company Relation Graph using networkx API
def plot_company_info(company_name, keyman=None, industry=None, finance=None, news=None, news_topic=None,
                      entity_relationship=None, cos_relationship=None):
    company_graph = nx.Graph()

    if keyman is not None:
        for index, row in keyman.iterrows():
            company_graph.add_edges_from([(company_name, "Keyman"),
                                          (row.title, "Keyman"),
                                          (row.title, row.full_name)])
            for index2, row2 in row.news.iteritems():
                # Lookup the relationship of entities from the keymans' News
                ner_news = ner(row2)
                person_list = [ent for ent in ner_news.ents if ent.label_ == "PERSON"]
                org_list = [ent for ent in ner_news.ents if ent.label_ == "ORG"]
                for person in person_list:
                    for org in org_list:
                        company_graph.add_edges_from([(person, org)])
                        company_graph.add_edges_from([(row2, person)])
                        company_graph.add_edges_from([(row2, org)])
                company_graph.add_edges_from([(row.full_name, row2)])

    if industry is not None:
        for tag in industry:
            company_graph.add_edges_from([(company_name, "Industry"),
                                          (tag, "Industry")])

    if finance is not None:
        company_graph.add_edges_from([(company_name, "Finance")])
        fin_title = ["Total Round: ", "", "Fund Raised: "]
        for i in range(3):
            if finance[i] is not None:
                company_graph.add_edges_from([(fin_title[i] + str(finance[i]), "Finance")])
        if finance[3] is not None:
            for partner in finance[3]:
                company_graph.add_edges_from([("Finance", "Partner"),
                                              (partner, "Partner")])
        if finance[4] is not None:
            for investor in finance[4]:
                company_graph.add_edges_from([("Finance", "Investor"),
                                              (investor, "Investor")])

    if news is not None:
        for index, row in news.iterrows():
            company_graph.add_edges_from([(company_name, "News"),
                                          (row.title, "News")])
            if news_topic != None:
                company_graph.add_edges_from([(news_topic[0], row.title),
                                              (news_topic[1], row.title)])

    if entity_relationship is not None:
        for edge in entity_relationship:
            company_graph.add_edges_from([(edge[0], edge[1])])

    if cos_relationship is not None:
        for edge in cos_relationship:
            company_graph.add_edges_from([(edge[0], edge[1])])

    val_map = {COMPANY_NAME: 1.0,
               "Keyman": 0.75,
               "Industry": 0.5,
               "Finance": 0.25,
               "News": 0.0}
    values = [val_map.get(node, 0.25) for node in company_graph.nodes()]
    pos = nx.spring_layout(company_graph)
    plt.figure(figsize=(60, 40))
    nx.draw(company_graph, pos, node_size=2000, cmap=plt.get_cmap('Pastel1'), node_color=values)
    nx.draw_networkx_labels(company_graph, pos, font_size=12)

    img = WORKING_DIR + "IMG/" + company_name.replace(" ", "_") + ".jpg"
    os.makedirs(os.path.dirname(img), exist_ok=True)
    plt.savefig(img)
    return img


# Get Text Output Information
def output_company_info(company_name, company_summary=None, summary=None, keyman=None, finance=None, news=None, img=None):
    txt = WORKING_DIR + "/RPT/" + company_name.replace(" ", "_") + ".txt"
    os.makedirs(os.path.dirname(txt), exist_ok=True)
    with open(txt, 'w') as fp:
        print('Company Name:', company_name, file=fp)

        if company_summary is not None:
            print('Summary:', company_summary, file=fp)

        if img is not None:
            print('Relation Graph:', img, file=fp)

        print("\n======Detail Report======", file=fp)

        if summary is not None:
            print("-----Company Information-----", file=fp)
            if summary[0] is not None:
                print('Web Link:', summary[0], file=fp)
            if summary[1] is not None:
                print('About:', summary[1], file=fp)
            if summary[2] is not None:
                print('Industry:', end=' ', file=fp)
                for tag in industry:
                    print(tag, end=' ', file=fp)
                print('', file=fp)
            print("-----Company Information-----\n\n", file=fp)

        if keyman is not None:
            print("-----Keyman Listing-----", file=fp)
            for index, row in keyman.iterrows():
                print("Name:", row.full_name, file=fp)
                print("Title:", row.title, file=fp)
                for index2, row2 in row.news.iteritems():
                    print('News', index2+1, ":", row2, file=fp)
                print("", file=fp)
            print("-----Keyman Listing-----\n\n", file=fp)

        if finance is not None:
            print("-----Financial Information-----", file=fp)
            if finance[0] is not None:
                print("Total Round:", finance[0], file=fp)

            if finance[1] is not None:
                print("Current Round:", finance[1], file=fp)

            if finance[2] is not None:
                print("Fund Raised:", finance[2], file=fp)

            if finance[3] is not None:
                print('Partner:', end=' ', file=fp)
                for partner in finance[3]:
                    print(partner, end=' ', file=fp)
                print("", file=fp)

            if finance[4] is not None:
                print('Investor:', end=' ', file=fp)
                for investor in finance[4]:
                    print(investor, end=' ', file=fp)
                print("", file=fp)
            print("-----Financial Information-----\n\n", file=fp)

        if news is not None:
            print("-----Company News Listing-----", file=fp)
            for index, row in news.iterrows():
                print(index, row.title, file=fp)
            print("-----Company News Listing-----\n\n", file=fp)
        print("======Detail Report======", file=fp)

    return txt


# GET CRUNCHBASE INFO
cb_file = WORKING_DIR + "CRUNCHBASE/" + COMPANY_NAME + ".pkl"
if os.path.exists(cb_file):
    with open(cb_file, "rb") as fp:
        cb_info = pickle.load(fp)
else:
    os.makedirs(os.path.dirname(cb_file), exist_ok=True)
    cb_info = get_crunchbase_information(get_url(COMPANY_NAME, site=CRUNCHBASE_SITE))
    with open(cb_file, "wb") as fp:
        pickle.dump(cb_info, fp)
summary, keyman, finance = cb_info

#####
# Get web text from company page
link = summary[0]
if link != None:
    web_text = bs(get_src([get_url(link)])[0], 'lxml').text.lower()


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def preprocess(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace('{html}', "")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '', cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words = [stemmer.stem(w) for w in filtered_words]
    lemma_words = [lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)


web_text_processed = preprocess(web_text)


# Get Part Of Speech for NLP analysis
def get_pos(text):
    word_list = word_tokenize(text)
    pos_value = nltk.pos_tag(word_list)

    vrb = set([word for (word, pos) in pos_value if (pos.startswith('VB'))])
    nns = set([word for (word, pos) in pos_value if (pos.startswith('NN'))])
    adj = set([word for (word, pos) in pos_value if (pos.startswith('JJ'))])
    return vrb, nns, adj


verb_set, noun_set, adj_set = get_pos(web_text)
noun_list = list(noun_set)

i = 0
while 1:
    if i == len(noun_list):
        break
    word = noun_list[i]
    if len(word) <= LEN_THRESHOLD:
        noun_list.remove(word)
        i -= 1
    i += 1

# Calculation COS similarity
def get_cos_similarity(p1, p2, model):
    embedding1 = model.encode(p1, convert_to_tensor=True)
    embedding2 = model.encode(p2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding1, embedding2).item()


# Get relationship of each others
model = SentenceTransformer('stsb-roberta-large')
pos_list = list()
for i in range(len(noun_list)):
    for j in range(i + 1, len(noun_list)):
        sim = get_cos_similarity(noun_list[i], noun_list[j], model)
        if sim >= SIM_THRESHOLD:
            pos_list.append([i, j])

# Summary the text
def summarize(text):
    if len(text) > 1024:
        text = text[:1024]
    summarizer = pipeline("summarization")
    summary_text = summarizer(text, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
    return summary_text

company_summary = summarize(web_text)

# Get NER
def ner(text):
    return spacy.load("en_core_web_sm")(text)

def remove_stopwords(texts):
    stopword_list = nltk.corpus.stopwords.words('english')
    return [[word for word in simple_preprocess(str(doc)) if word not in stopword_list] for doc in texts]


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# SEARCH GOOGLE NEWS
news = get_google_news(COMPANY_NAME)

# Get NER Relationship in News
ner_list = list()
for index, row in news.iterrows():
    ner_news = ner(row.title)
    if debug_mode:
        displacy.serve(ner_news, style="ent")
    person_list = [ent for ent in ner_news.ents if ent.label_ == "PERSON"]
    org_list = [ent for ent in ner_news.ents if ent.label_ == "ORG"]
    for person in person_list:
        for org in org_list:
            ner_list.append([person, org])
            ner_list.append([row.title, person])
            ner_list.append([row.title, org])

# Get Company News Topic
headline_list = list()
topic_model = BERTopic(embedding_model=model)
for index, row in news.iterrows():
    data_words = list(sent_to_words(row.title.split()))
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    docs = remove_stopwords(row.title.split())
    docs = make_bigrams(docs)
    docs = make_trigrams(docs)
    docs = lemmatization(docs)
    d2tod1 = list()
    for i in docs:
        for j in i:
            d2tod1.append(j)
    processed_headline = " ".join(d2tod1)
    headline_list.append(processed_headline)

topics, probabilities = topic_model.fit_transform(headline_list)
top_topic_important_1 = topic_model.get_topic(topic_model.get_topic_freq().Topic[0])[0][0]
top_topic_important_2 = topic_model.get_topic(topic_model.get_topic_freq().Topic[0])[1][0]

top_topic = [top_topic_important_1, top_topic_important_2]

# Generate Tree Graph
industry = None
if summary != None:
    industry = summary[2]
image_link = plot_company_info(COMPANY_NAME, keyman=keyman, industry=industry, finance=finance, news=get_google_news(COMPANY_NAME),
                  news_topic=top_topic, entity_relationship=ner_list, cos_relationship=pos_list)
output_company_info(COMPANY_NAME, company_summary=company_summary, summary=summary, keyman=keyman, finance=finance, news=get_google_news(COMPANY_NAME), img=image_link)
