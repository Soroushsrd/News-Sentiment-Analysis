import requests
from transformers import pipeline
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime


API_KEY = 'Your_API_Key'
endpoint = 'https://newsdata.io/api/1/news'

keywords = ['BTC']
today_date = datetime.today().strftime('%Y-%m-%d-%H-%M')

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def save_to_text_file(title, summary):
    with open(f'news_articles_{today_date}.txt', 'a', encoding='utf-8') as file:
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
sid = SentimentIntensityAnalyzer()
sentiments=[]

#add in category and country
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
        max_chunk_length = 500  # Define a maximum length for each chunk
        article_chunks = [article_text[i:i + max_chunk_length] for i in
                          range(0, len(article_text), max_chunk_length)]
        all_article_summaries = []  # Create a list for each article's summaries
        combined_summary=''
        for chunk in article_chunks:
            preprocessed_chunk = preprocess_text(chunk)  # Preprocess the chunk
            summarized_text = summarizer(preprocessed_chunk, max_length=100, min_length=30, do_sample=False)
            summarized_text = summarized_text[0]['summary_text']
            combined_summary += summarized_text + ' '  # Concatenate each chunk's summary

        # After processing all chunks, save the article title and its corresponding summary
        save_to_text_file(title, combined_summary)
        # Sentiment analysis using NLTK's VADER
        sentiments.append(sid.polarity_scores(combined_summary))

    else:
        print(f"No content found for article '{title}'")

# Extract compound scores for mean calculation and visualization
compound_scores = [sentiment['compound'] for sentiment in sentiments]
if len(compound_scores) > 0:
    sentiment_mean = sum(compound_scores) / len(compound_scores)
else:
    sentiment_mean = 0  # or any default value you prefer
print(f"Mean Compound Sentiment: {sentiment_mean}")
print("Titles and summaries saved in 'news_articles.txt'")
