!pip install vaderSentiment
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Import Data
df = pd.read_csv(r"/content/drive/MyDrive/Colab Notebooks/COVID19_news_final/COVID19_news.csv")
print("Original Data Size:", df.shape[0])

scores =[]
sentences = df['text']

# Calculate Vader Score
analyser = SentimentIntensityAnalyzer()
for sentence in sentences:
    score = analyser.polarity_scores(sentence)
    scores.append(score)
    
#Converting List of Dictionaries into Dataframe
dataFrame= pd.DataFrame(scores)
print(dataFrame)

dataFrame.to_csv(r"/content/drive/MyDrive/Colab Notebooks/50state_vader_gdelt.csv", header=True)
