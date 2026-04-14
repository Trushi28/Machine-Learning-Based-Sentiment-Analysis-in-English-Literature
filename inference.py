import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoTokenizer
import numpy as np
import os
import sys

# ── Labels ────────────────────────────────────────────────────────────────────
EMOTION_LABELS   = ['Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust', 'Neutral']
SENTIMENT_LABELS = ['Negative', 'Neutral', 'Positive']
NUM_EMOTIONS     = len(EMOTION_LABELS)
NUM_SENTIMENTS   = len(SENTIMENT_LABELS)
MAX_LEN          = 256

# Default thresholds (fallback if not saved in checkpoint)
DEFAULT_THRESHOLDS = [0.5] * NUM_EMOTIONS


# ── Model Architecture (must match training exactly) ──────────────────────────

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.W       = nn.Linear(hidden_size, hidden_size)
        self.v       = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, lstm_output, mask=None):
        energy = torch.tanh(self.W(lstm_output))
        scores = self.v(energy).squeeze(-1)
        if mask is not None:
            scores = scores.float().masked_fill(mask.float() == 0, -1e4)
        attention_weights = F.softmax(scores, dim=1)
        attention_weights = self.dropout(attention_weights)
        context = torch.sum(attention_weights.unsqueeze(-1) * lstm_output.float(), dim=1)
        return context, attention_weights


class EnhancedSentimentModel(nn.Module):
    def __init__(self, num_emotions=NUM_EMOTIONS, num_sentiments=NUM_SENTIMENTS):
        super().__init__()

        self.bert     = BertModel.from_pretrained('bert-base-uncased')
        bert_size     = 768

        # CNN — balanced filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(bert_size, 512, 3),
            nn.Conv1d(bert_size, 512, 4),
            nn.Conv1d(bert_size, 512, 5),
        ])
        cnn_out = 512 * 3  # 1536

        # BiLSTM
        self.lstm = nn.LSTM(
            bert_size, 512,
            num_layers=3,
            bidirectional=True,
            dropout=0.2,
            batch_first=True
        )
        lstm_out = 512 * 2  # 1024

        self.attention = SelfAttention(lstm_out, 0.1)

        # LayerNorm before fusion
        self.cnn_norm  = nn.LayerNorm(cnn_out)
        self.lstm_norm = nn.LayerNorm(lstm_out)

        combined = cnn_out + lstm_out  # 2560

        self.sentiment_clf = nn.Sequential(
            nn.Linear(combined, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_sentiments)
        )
        self.emotion_clf = nn.Sequential(
            nn.Linear(combined, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_emotions)
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids, attention_mask).last_hidden_state

        # CNN path
        x = bert_out.float().transpose(1, 2)
        cnn_feats = []
        for conv in self.convs:
            h = torch.relu(conv(x))
            h = torch.max_pool1d(h, h.size(2)).squeeze(2)
            cnn_feats.append(h)
        cnn_out = self.cnn_norm(torch.cat(cnn_feats, 1))
        cnn_out = self.dropout(cnn_out)

        # BiLSTM + Attention path
        lstm_out, _ = self.lstm(bert_out.float())
        lstm_out     = self.dropout(lstm_out)
        context, attn_weights = self.attention(lstm_out, attention_mask)
        context = self.lstm_norm(context)

        combined   = torch.cat([cnn_out, context], 1)
        sentiment  = self.sentiment_clf(combined)
        emotion    = self.emotion_clf(combined)
        return sentiment, emotion, attn_weights


# ── Model Runner ──────────────────────────────────────────────────────────────

class ModelRunner:
    def __init__(self, model_path='enhanced_emotion_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        print(f"Loading model from: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Download enhanced_emotion_model.pth from your Colab notebook first."
            )

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model     = EnhancedSentimentModel()

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        # Load calibrated thresholds if saved, else fall back to defaults
        self.thresholds = checkpoint.get('best_thresholds', DEFAULT_THRESHOLDS)

        # Override labels from checkpoint if present (handles future class changes)
        self.emotion_labels   = checkpoint.get('emotion_labels',   EMOTION_LABELS)
        self.sentiment_labels = checkpoint.get('sentiment_labels', SENTIMENT_LABELS)

        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
        print(f"Emotion classes  : {self.emotion_labels}")
        print(f"Thresholds       : {[f'{t:.2f}' for t in self.thresholds]}")

    def predict(self, text: str) -> dict:
        """
        Returns:
            {
                'sentiment': str,
                'confidence': float,
                'all_sentiment_probs': dict,
                'emotions': list of (label, prob),
                'all_emotion_probs': dict
            }
        """
        enc = self.tokenizer(
            text,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].to(self.device)
        mask      = enc['attention_mask'].to(self.device)

        with torch.no_grad():
            sent_logits, emo_logits, _ = self.model(input_ids, mask)

        # Sentiment
        sent_probs    = torch.softmax(sent_logits.float(), 1)[0].cpu().numpy()
        pred_sent_idx = int(sent_probs.argmax())
        pred_sent     = self.sentiment_labels[pred_sent_idx]
        confidence    = float(sent_probs[pred_sent_idx])

        # Emotions — use calibrated per-emotion thresholds
        emo_probs = torch.sigmoid(emo_logits.float())[0].cpu().numpy()
        detected  = [
            (self.emotion_labels[i], float(emo_probs[i]))
            for i, t in enumerate(self.thresholds)
            if emo_probs[i] >= t
        ]
        # Always return at least one emotion
        if not detected:
            detected = [(self.emotion_labels[int(emo_probs.argmax())], float(emo_probs.max()))]
        detected.sort(key=lambda x: x[1], reverse=True)

        return {
            'sentiment':            pred_sent,
            'confidence':           confidence,
            'all_sentiment_probs':  dict(zip(self.sentiment_labels, sent_probs.tolist())),
            'emotions':             detected,
            'all_emotion_probs':    dict(zip(self.emotion_labels, emo_probs.tolist())),
        }


# ── Interactive CLI ───────────────────────────────────────────────────────────

def print_result(result: dict):
    print(f"\n  Sentiment : {result['sentiment']} ({result['confidence']:.1%})")
    emo_str = ', '.join(f"{e} ({p:.2f})" for e, p in result['emotions'])
    print(f"  Emotions  : {emo_str}")
    print("-" * 55)


if __name__ == "__main__":
    try:
        runner = ModelRunner('enhanced_emotion_model.pth')

        print("\n" + "=" * 55)
        print("  Sentiment & Emotion Analysis")
        print("  Type 'quit' or press Ctrl+C to exit.")
        print("=" * 55 + "\n")

        while True:
            try:
                user_input = input("Enter text > ").strip()
            except EOFError:
                break

            if user_input.lower() in ('quit', 'exit', 'q'):
                print("\nExiting...")
                break

            if not user_input:
                continue

            try:
                result = runner.predict(user_input)
                print_result(result)
            except Exception as e:
                print(f"Error: {e}")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
