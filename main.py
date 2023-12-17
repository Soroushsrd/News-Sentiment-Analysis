import requests
from transformers import pipeline
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

API_KEY = 'your_api'
endpoint = 'https://newsdata.io/api/1/news'

keywords = ['BTC', 'bitcoin', 'crypto']

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def save_to_text_file(title, summary):
    with open('news_articles.txt', 'a', encoding='utf-8') as file:
        file.write(f"Title: {title}\n")
        file.write(f"Summary: {summary}\n\n")


def preprocess_text(text):
    # Tokenization, removing punctuation, converting to lowercase
    tokens = nltk.word_tokenize(text.lower())
    # Remove stopwords (sample list, you might want to use a complete set)
    stopwords = set(nltk.corpus.stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stopwords]
    return ' '.join(filtered_tokens)



results = []
processed_articles = []
all_summaries = []  # List to store all summaries

for keyword in keywords:
    params = {
        'apikey': API_KEY,
        'q': f'"{keyword}"',
        'language': 'en',
    }

    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        news_data = response.json()
        results.extend(news_data['results'])
    else:
        print(f"Request for keyword '{keyword}' failed with status code: {response.status_code}")

for article_index, article in enumerate(results):
    title = article['title']
    article_text = article['content']
    if article_text:
        max_chunk_length = 800  # Define a maximum length for each chunk
        article_chunks = [article_text[i:i + max_chunk_length] for i in
                          range(0, len(article_text), max_chunk_length)]
        preprocessed_articles = preprocess_text(article_text)
        processed_articles.append(preprocessed_articles)
        for chunk in article_chunks:
            summarized_text = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
            summarized_text = summarized_text[0]['summary_text']
            all_summaries.append(summarized_text)

        # After processing all chunks, save the article title and its corresponding summary
        save_to_text_file(title, all_summaries[article_index])
    else:
        print(f"No content found for article '{title}'")
# Sentiment analysis using NLTK's VADER
sid = SentimentIntensityAnalyzer()
sentiments = [sid.polarity_scores(article) for article in processed_articles]

# Extract compound scores for mean calculation and visualization
compound_scores = [sentiment['compound'] for sentiment in sentiments]
if len(compound_scores) > 0:
    sentiment_mean = sum(compound_scores) / len(compound_scores)
else:
    sentiment_mean = 0  # or any default value you prefer
print(f"Mean Compound Sentiment: {sentiment_mean}")
print("Titles and summaries saved in 'news_articles.txt'")
