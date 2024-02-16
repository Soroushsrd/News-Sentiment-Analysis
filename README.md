# News Sentiment Analysis

I wrote this code using [NewsData](https://newsdata.io/) API. You can sign up and use their free package to access recent news.
You can also get their premium package which allows you to access historical data and more options.

This code uses [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) for summarization.
We first request the title and content of the news from NewsData and then cut the content into smaller chunks as this summarizer does not support texts longer than 1024 tokens. These chunks are later summarized and added to the overal summary of the content.

Each news is later analyzed for sentiment using NLTK Vader sentiment anlyzer and the compound sentiment is calculated.
A text format file is also put out with the titles and their content summary.


## Usage

It can be used to get the most recent news about any subject. All you need is an API key, your keywords as a list and the language you need your news to be in.
You can use this code for you fundamental analysis regarding specific stocks or cryptocurrencies.

Different querys can be made to NewsData website and you can change the [parameters](https://newsdata.io/documentation) this code uses.

## Caution

Any usage of this code for trading should be done at your own discretion and i'm not responsible for your losses!
