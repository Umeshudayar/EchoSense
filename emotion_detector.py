import os
import logging
import pickle
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.preprocessing import LabelEncoder
from config import EMOTION_MODEL_PATH

logger = logging.getLogger(__name__)

class EmotionDetector:
    def __init__(self, model_type="distilbert"):
        """Initialize emotion detector with either DistilBERT or custom model."""
        self.model_type = model_type
        
        if model_type == "distilbert":
            try:
                # Load DistilBERT tokenizer and model
                self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
                
                # Load emotion classifier
                if os.path.exists(EMOTION_MODEL_PATH):
                    with open(EMOTION_MODEL_PATH, 'rb') as f:
                        self.classifier = pickle.load(f)
                    
                    # Get emotion labels
                    self.labels = self.classifier.classes_
                else:
                    # Fallback emotion classifier if file not found
                    logger.warning(f"Emotion model not found at {EMOTION_MODEL_PATH}")
                    logger.warning("Using fallback emotion detection")
                    self.classifier = None
                    self.labels = ["anger", "fear", "joy", "love", "sadness", "surprise"]
            except Exception as e:
                logger.error(f"Error initializing DistilBERT emotion detector: {e}")
                # Fallback to simple keyword-based detection
                self.model_type = "keyword"
        else:
            # Simple keyword-based emotion detection
            self.model_type = "keyword"
    
    def detect_with_distilbert(self, text):
        """Detect emotions using DistilBERT."""
        try:
            # Tokenize input text
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Get model output
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use [CLS] token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            
            if self.classifier:
                # Use trained classifier to predict emotion
                emotion_probs = self.classifier.predict_proba(embeddings)[0]
                emotion_idx = np.argmax(emotion_probs)
                emotion = self.labels[emotion_idx]
                confidence = emotion_probs[emotion_idx]
            else:
                # Fallback if no classifier is available
                emotion = "neutral"
                confidence = 1.0
            
            return {"emotion": emotion, "confidence": float(confidence)}
        
        except Exception as e:
            logger.error(f"Error detecting emotion with DistilBERT: {e}")
            return {"emotion": "unknown", "confidence": 0.0}
    
    def detect_with_keywords(self, text):
        """Simple keyword-based emotion detection."""
        text = text.lower()
        
        # Simple emotion keyword mapping
        emotion_keywords = {
            "joy": ["happy", "happiness", "joyful", "glad", "delighted", "pleased", "excited", "cheerful"],
            "sadness": ["sad", "unhappy", "depressed", "miserable", "gloomy", "heartbroken", "tearful"],
            "anger": ["angry", "furious", "annoyed", "irritated", "mad", "outraged", "frustrated"],
            "fear": ["afraid", "scared", "frightened", "terrified", "anxious", "worried", "nervous"],
            "surprise": ["surprised", "shocked", "amazed", "astonished", "startled", "unexpected"],
            "disgust": ["disgusted", "revolted", "repulsed", "sickened", "distaste"],
            "neutral": ["normal", "fine", "okay", "alright"]
        }
        
        # Count occurrences of emotion keywords
        emotion_counts = {emotion: 0 for emotion in emotion_keywords}
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    emotion_counts[emotion] += text.count(keyword)
        
        # Find most common emotion
        max_count = max(emotion_counts.values())
        if max_count > 0:
            # Get emotion with highest count
            emotion = max(emotion_counts, key=emotion_counts.get)
            # Simple confidence score based on count
            confidence = min(max_count / 10, 1.0)  # Cap at 1.0
        else:
            emotion = "neutral"
            confidence = 0.5
        
        return {"emotion": emotion, "confidence": confidence}
    
    def detect_emotion(self, text):
        """Detect emotion in text using the selected method."""
        if not text or len(text.strip()) == 0:
            return {"emotion": "unknown", "confidence": 0.0}
            
        if self.model_type == "distilbert":
            return self.detect_with_distilbert(text)
        else:
            return self.detect_with_keywords(text)