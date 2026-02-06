import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoTokenizer
import numpy as np
import os
import sys

# --- 1. Define Architecture (Must match training exactly) ---

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, lstm_output, mask=None):
        energy = torch.tanh(self.W(lstm_output))
        scores = self.v(energy).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=1)
        attention_weights = self.dropout(attention_weights)
        context = torch.sum(attention_weights.unsqueeze(-1) * lstm_output, dim=1)
        return context, attention_weights

class EnhancedSentimentModel(nn.Module):
    """Enhanced BERT-base + CNN + BiLSTM + Attention"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize BERT 
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        bert_size = 768
        
        # CNN Layers
        self.convs = nn.ModuleList([
            nn.Conv1d(bert_size, 256, 3),
            nn.Conv1d(bert_size, 512, 4),
            nn.Conv1d(bert_size, 1024, 5)
        ])
        cnn_out = 256 + 512 + 1024  # 1792
        
        # BiLSTM
        self.lstm = nn.LSTM(
            bert_size, 512, 
            num_layers=3, 
            bidirectional=True, 
            dropout=0.2, 
            batch_first=True
        )
        lstm_out = 512 * 2  # 1024
        
        # Attention
        self.attention = SelfAttention(lstm_out, 0.1)
        
        # Classifiers
        combined = cnn_out + lstm_out
        
        self.sentiment_clf = nn.Sequential(
            nn.Linear(combined, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3)
        )
        
        self.emotion_clf = nn.Sequential(
            nn.Linear(combined, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 6)
        )
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids, attention_mask).last_hidden_state
        
        # CNN
        x = bert_out.transpose(1, 2)
        cnn_feats = []
        for conv in self.convs:
            h = torch.relu(conv(x))
            h = torch.max_pool1d(h, h.size(2)).squeeze(2)
            cnn_feats.append(h)
        cnn_out = torch.cat(cnn_feats, 1)
        cnn_out = self.dropout(cnn_out)
        
        # BiLSTM + Attention
        lstm_out, _ = self.lstm(bert_out)
        lstm_out = self.dropout(lstm_out)
        context, attn_weights = self.attention(lstm_out, attention_mask)
        
        # Combined
        combined = torch.cat([cnn_out, context], 1)
        
        # Classify
        sentiment = self.sentiment_clf(combined)
        emotion = self.emotion_clf(combined)
        
        return sentiment, emotion, attn_weights

# --- 2. Inference Runner Class ---

class ModelRunner:
    def __init__(self, model_path='enhanced_sentiment_model.pth'):
        self.device = torch.device('cpu')
        print(f"Loading model from {model_path} to {self.device}...")
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Initialize Architecture
        self.model = EnhancedSentimentModel()
        
        # Load Weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")

    def predict(self, text):
        # Preprocessing
        encoding = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        mask = encoding['attention_mask'].to(self.device)
        
        # Inference
        with torch.no_grad():
            sent_logits, emo_logits, _ = self.model(input_ids, mask)
        
        # Post-processing
        sent_probs = torch.softmax(sent_logits, 1)[0]
        emo_probs = torch.sigmoid(emo_logits)[0]
        
        # Definitions
        sent_labels = ['Negative', 'Neutral', 'Positive']
        emo_labels = ['Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Neutral']
        
        # Get Top Sentiment
        pred_sent_idx = torch.argmax(sent_probs).item()
        pred_sent = sent_labels[pred_sent_idx]
        confidence = sent_probs[pred_sent_idx].item()
        
        # Get Top 3 Emotions
        top_emo_indices = torch.topk(emo_probs, 3).indices
        top_emotions = [
            (emo_labels[idx], emo_probs[idx].item()) 
            for idx in top_emo_indices
        ]
        
        return {
            "sentiment": pred_sent,
            "confidence": confidence,
            "emotions": top_emotions
        }

# --- 3. Interactive Main Loop ---

if __name__ == "__main__":
    try:
        # Initialize the model once
        runner = ModelRunner('enhanced_sentiment_model.pth')
        
        print("\n" + "="*50)
        print(" INTERACTIVE SENTIMENT ANALYSIS ")
        print(" Type 'quit', 'exit', or 'q' to stop.")
        print("="*50 + "\n")

        while True:
            # Get user input
            try:
                user_input = input("Enter text > ").strip()
            except EOFError:
                break # Handle Ctrl+D gracefully

            # Check exit conditions
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nExiting...")
                break
            
            # Skip empty inputs
            if not user_input:
                continue

            # Run prediction
            try:
                result = runner.predict(user_input)
                
                # Format Output
                print(f"\n   Sentiment:  {result['sentiment']} ({result['confidence']:.1%})")
                print(f"   Emotions:   {', '.join([f'{e} ({p:.2f})' for e, p in result['emotions']])}")
                print("-" * 50)
                
            except Exception as e:
                print(f"Error processing text: {e}")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please download the 'enhanced_sentiment_model.pth' file from your Colab notebook.")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
