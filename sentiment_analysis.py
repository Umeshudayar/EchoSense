from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, method="vader"):
        """Initialize sentiment analyzer with either VADER or TextBlob."""
        self.method = method
        
        if method == "vader":
            self.vader = SentimentIntensityAnalyzer()
    
    def analyze_vader(self, text):
        """Analyze sentiment using VADER."""
        try:
            sentiment_scores = self.vader.polarity_scores(text)
            
            # Determine sentiment category
            compound_score = sentiment_scores["compound"]
            
            if compound_score >= 0.05:
                sentiment = "positive"
            elif compound_score <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return {
                "sentiment": sentiment,
                "scores": {
                    "positive": sentiment_scores["pos"],
                    "negative": sentiment_scores["neg"],
                    "neutral": sentiment_scores["neu"],
                    "compound": sentiment_scores["compound"]
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment with VADER: {e}")
            return {"sentiment": "unknown", "scores": {}}
    
    def analyze_textblob(self, text):
        """Analyze sentiment using TextBlob."""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Determine sentiment category
            if polarity > 0.1:
                sentiment = "positive"
            elif polarity < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return {
                "sentiment": sentiment,
                "scores": {
                    "polarity": polarity,
                    "subjectivity": subjectivity
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment with TextBlob: {e}")
            return {"sentiment": "unknown", "scores": {}}
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using the selected method."""
        if not text or len(text.strip()) == 0:
            return {"sentiment": "unknown", "scores": {}}
            
        if self.method == "vader":
            return self.analyze_vader(text)
        else:
            return self.analyze_textblob(text)