import os
import time
import re
import numpy as np
import pywikibot as wiki
from nltk import sent_tokenize, word_tokenize
from sqlalchemy import create_engine, Column, Integer, String, Table, ForeignKey, Text, text
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


engine = session = model = None
STOP_WORDS = {"the", "is", "in", "at", "of", "on", "and", "a", "to", "with", "for", "from", "by", "an", "as", "that",
              "it", "this", "these", "those", "are", "be", "has", "have", "was", "were", "or", "not", "but", "if", "so",
              "then", "when", "up", "down", "out", "about", "into", "over", "after", "before", "above", "below",
              "under", "between", "against", "during", "without", "within", "near", "along", "through", "to", "no",
              "yes", "its", "-"}

Base = declarative_base()

trends_keywords_table = Table(
    'trends_keywords', Base.metadata,
    Column('trend_id', Integer, ForeignKey('trends.id'), primary_key=True),
    Column('keyword_id', Integer, ForeignKey('keywords.id'), primary_key=True)
)


class Trend(Base):
    __tablename__ = 'trends'

    id = Column(Integer, primary_key=True)
    title = Column(String, unique=True, nullable=False)
    has_content = Column(Integer, default=-1)

    keywords = relationship(
        'Keyword',
        secondary=trends_keywords_table,
        back_populates='trends'
    )


class Keyword(Base):
    __tablename__ = 'keywords'

    id = Column(Integer, primary_key=True)
    keyword = Column(String, unique=True, nullable=False)
    content = Column(Text)
    article_title = Column(String)
    article_url = Column(String)

    trends = relationship(
        'Trend',
        secondary=trends_keywords_table,
        back_populates='keywords'
    )


def create_fts_tables(engine):
    with engine.connect() as conn:
        conn.execute(text('CREATE VIRTUAL TABLE IF NOT EXISTS trends_fts USING fts5(title);'))
        conn.execute(text('CREATE VIRTUAL TABLE IF NOT EXISTS keywords_fts USING fts5(keyword, content);'))


def init_db():
    global engine, session

    engine = create_engine('sqlite:///pages.db')

    if not os.path.exists('pages.db'):
        with engine.connect() as conn:
            conn.execute(text('CREATE VIRTUAL TABLE IF NOT EXISTS trends_fts USING fts5(title);'))
            conn.execute(text('CREATE VIRTUAL TABLE IF NOT EXISTS keywords_fts USING fts5(keyword, content);'))
            conn.commit()

    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()


def create_trend(title):
    new_trend = Trend(title=title, has_content=-1)
    session.add(new_trend)
    session.commit()

    with engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(
                text("INSERT INTO trends_fts(rowid, title) VALUES (:rowid, :title);"),
                {"rowid": new_trend.id, "title": title}
            )

            verify_query = text("SELECT rowid, title FROM trends_fts WHERE rowid = :rowid;")
            verification_results = conn.execute(verify_query, {"rowid": new_trend.id}).fetchall()

            trans.commit()
        except Exception as e:
            print("Error during FTS insertion:", e)
            trans.rollback()


def init_model():
    global model
    if not os.path.exists('search.model'):
        if model is None:
            model = Word2Vec(vector_size=100, window=10, min_count=1, workers=4, epochs=30, alpha=0.03)
    if model is None:
        model = Word2Vec.load('search.model')


def title_exist(title):
    record = session.query(Trend).filter_by(title=title).first()
    return True if record else False


def update_trends():
    url = 'https://trends.google.com/trending?geo=US&hl=en-US'
    total_pages = 5

    driver = webdriver.Chrome()
    driver.get(url)
    time.sleep(2)

    try:
        for i in range(total_pages):
            table = driver.find_element(By.TAG_NAME, 'table')
            tbodies = table.find_elements(By.TAG_NAME, 'tbody')
            if len(tbodies) < 2:
                print("The table doesn't have a second tbody.")
            else:
                tbody = tbodies[1]
                for tr in tbody.find_elements(By.TAG_NAME, 'tr'):
                    tds = tr.find_elements(By.TAG_NAME, 'td')
                    if len(tds) >= 2:
                        div = tds[1].find_element(By.CLASS_NAME, 'mZ3RIc')
                        title = div.text
                        if not title_exist(title):
                            create_trend(title)
            next_page_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//button[@aria-label="Go to next page"]'))
            )
            next_page_button.click()
            time.sleep(1)
    except Exception as e:
        print('Error occurred when getting trends', e)
    finally:
        driver.quit()


def load_trends():
    update_time = 1000
    updated = False
    if int(time.time()) - int(os.path.getmtime('./pages.db')) > update_time:
        updated = True
        update_trends()

    return updated


def get_new_trends():
    records = session.query(Trend).filter_by(has_content=-1).all()
    new_trends = [record.title for record in records]

    return records


def filter_keywords(title):
    title_no_punct = re.sub(r"[^\w\s]", '', title)
    words = title_no_punct.lower().split()
    keywords = [word for word in words if word not in STOP_WORDS]
    return keywords


def get_html(keyword):
    try:
        site = wiki.Site('en', 'wikipedia')
        page = wiki.Page(site, keyword)
        time.sleep(1)
        if page.exists():
            html = page.extract('plain')
            title = page.title()
            url = page.full_url()
            return html, title, url
    except Exception as e:
        print('Error occurred when searching wikipedia', e)

    return None


def update_keyword(keyword, html, title, url, trend):
    new_keyword = Keyword(
        keyword=keyword,
        content=html,
        article_title=title,
        article_url=url
    )
    trend.keywords.append(new_keyword)

    session.add(new_keyword)
    session.commit()

    with engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(
                text("INSERT INTO keywords_fts(rowid, keyword, content) VALUES (:rowid, :keyword, :content);"),
                {"rowid": new_keyword.id, "keyword": keyword, "content": html}
            )
            trans.commit()
        except Exception as e:
            print("Error during FTS insertion:", e)
            trans.rollback()


def get_htmls(trend):
    keywords = filter_keywords(trend.title)

    compiled_html = []

    for keyword in keywords:
        existing_keyword = session.query(Keyword).filter_by(keyword=keyword).first()

        if not existing_keyword:
            result = get_html(keyword)

            if result:
                html, title, url = result

                update_keyword(keyword, html, title, url, trend)
                trend.has_content = 1
                compiled_html.append(html)
        else:
            compiled_html.append(existing_keyword.content)

        if len(compiled_html) == 0:
            trend.has_content = 0
    return compiled_html


def clean_text(text):
    text = text.encode('ascii', errors='ignore').decode()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    return text.lower()


def process_text(text):
    sentences = []
    text = clean_text(text)
    tokenize_sentences = sent_tokenize(text)
    for sentence in tokenize_sentences:
        words = word_tokenize(sentence)
        words = [word for word in words if word]
        sentences.append(words)

    return sentences


def lookup(trend):
    cleaned_text = None
    htmls = get_htmls(trend)
    if len(htmls) > 0:
        compiled_text = ' '.join(htmls)
        cleaned_text = process_text(compiled_text)

    return cleaned_text


def train_model():
    new_trends = get_new_trends()[:20]
    global model
    for trend in new_trends:
        data = lookup(trend)

        if data:
            if model.wv.key_to_index:
                model.build_vocab(data, update=True)
            else:
                model.build_vocab(data)

            model.train(data, total_examples=len(data), epochs=10)

    model.save('search.model')
    model = Word2Vec.load('search.model')


def search_articles(query):
    results = []

    with engine.connect() as conn:
        trends_query = text("SELECT rowid, title FROM trends_fts WHERE trends_fts MATCH :query;")
        trends_results = conn.execute(trends_query, {"query": query}).fetchall()

        for row in trends_results:
            trend_id, trend_title = row
            results.append({
                'type': 'trend',
                'id': trend_id,
                'title': trend_title
            })

        keywords_query = text("SELECT rowid, keyword, content FROM keywords_fts WHERE keywords_fts MATCH :query;")
        keywords_results = conn.execute(keywords_query, {"query": query}).fetchall()

        for row in keywords_results:
            keyword_id, keyword, content = row
            keyword_record = session.query(Keyword).filter_by(keyword=keyword).first()
            results.append({
                'type': 'keyword',
                'id': keyword_id,
                'keyword': keyword,
                'article_url': keyword_record.article_url,
                'article_title': keyword_record.article_title
            })

            associated_trends_query = text("""
                SELECT trends.id, trends.title
                FROM trends
                JOIN trends_keywords ON trends.id = trends_keywords.trend_id
                WHERE trends_keywords.keyword_id = :keyword_id
            """)
            associated_trends_results = conn.execute(associated_trends_query, {"keyword_id": keyword_id}).fetchall()

            for trend_row in associated_trends_results:
                trend_id, trend_title = trend_row
                if not any(r['id'] == trend_id and r['type'] == 'trend' for r in results):
                    results.append({
                        'type': 'trend',
                        'id': trend_id,
                        'title': trend_title
                    })

    return results


def get_similar_words(model, user_query):
    if user_query in model.wv.key_to_index:
        return model.wv.most_similar(user_query)
    return None


def predict_next_word(model, user_query, stop_words):
    valid_context = [word for word in user_query if word in model.wv.key_to_index]

    if not valid_context:
        print('All words in query not in vocab')
        return None

    possible_words = []

    avg_vec = np.mean([model.wv[word] / np.linalg.norm(model.wv[word]) for word in valid_context], axis=0)
    avg_words = model.wv.similar_by_vector(avg_vec, topn=20)

    next_words = [word for word, similarity in avg_words if word not in valid_context and word not in stop_words and word not in user_query and word not in STOP_WORDS]
    for word in next_words:
        possible_words.append(word)
    return possible_words


def get_predictions(user_query):
    stop_words = set(stopwords.words('english'))
    similar_words = [word[0] for word in get_similar_words(model, user_query)]
    predicted_words = predict_next_word(model, user_query, stop_words)
    return similar_words, predicted_words


def search_and_suggest(query):
    search_results = search_articles(query)
    similar_words, predicted_words = get_predictions(query)
    predictions = predicted_words + similar_words

    return search_results, predictions


if __name__ == '__main__':
    init_db()
    init_model()
    updated = load_trends()
    if updated:
        train_model()
    query = input('Enter search query: ')
    results, predictions = search_and_suggest(query)
    print('-'*20)
    for result in results:
        if result['type'] == 'trend':
            print(f"Trend: {result['title']} (ID: {result['id']})")
        elif result['type'] == 'keyword':
            print(f"Keyword: {result['keyword']} {result['article_url']}")
    print('Suggestions: ')
    print(predictions)
    print('-'*20)
