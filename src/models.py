import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Ép hệ thống dùng Keras 2 (Tương thích với HuggingFace)
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tf_keras
from tf_keras.models import Sequential
from tf_keras.layers import Input, Embedding, LSTM, Dropout, Dense, TextVectorization
from tf_keras.callbacks import EarlyStopping

# Scikit-learn & SciPy
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from scipy.sparse import hstack, csr_matrix

# Transformers
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, TFAutoModel
from sentence_transformers import SentenceTransformer

def compute_comprehensive_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return {
        "Accuracy": round(acc, 4),
        "F1_Macro": round(f1, 4),
        "Precision_Macro": round(precision, 4),
        "Recall_Macro": round(recall, 4)
    }

def balance_data(df, text_col, label_col='label', random_state=42):
    # Tìm số lượng của class ít nhất
    min_class_size = df[label_col].value_counts().min()

    # Cân bằng từng class
    df_balanced = pd.concat([
        df[df[label_col] == label].sample(n=min_class_size, random_state=random_state)
        for label in df[label_col].unique()
    ])

    # Trộn đều lại dữ liệu
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"   -> [Dữ liệu đã cân bằng] Tổng: {len(df_balanced)} mẫu. (Mỗi class có {min_class_size} mẫu)")
    return df_balanced

def evaluate_model(y_test, y_pred, model_name, class_names=None):
    """Hàm đánh giá in ra Classification Report và vẽ Confusion Matrix"""
    print("\n" + "="*40)
    print(f"MODEL RESULTS: {model_name}")
    print("="*40)

    # 1. Print Classification Report
    print("--- Classification Report ---")
    if class_names is not None:
        print(classification_report(y_test, y_pred, target_names=[str(c) for c in class_names]))
    else:
        print(classification_report(y_test, y_pred))

    # 2. Plotting Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)

    if class_names is None:
        class_names = sorted(list(set(y_test)))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=class_names, yticklabels=class_names,
                cbar=False)
    plt.title(f'Confusion Matrix: {model_name}', fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()


class CombinedExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=self.max_features)
        self.bert = SentenceTransformer('all-MiniLM-L6-v2')

    def fit(self, texts, y=None):
        self.tfidf.fit(texts)
        return self

    def transform(self, texts):
        tfidf_feat = self.tfidf.transform(texts)
        bert_feat = csr_matrix(self.bert.encode(texts, show_progress_bar=False))
        return hstack([tfidf_feat, bert_feat])

    def fit_transform(self, texts, y=None):
        tfidf_feat = self.tfidf.fit_transform(texts)
        bert_feat = csr_matrix(self.bert.encode(texts, show_progress_bar=False))
        return hstack([tfidf_feat, bert_feat])
    
class SklearnPipeline:
    def __init__(self, extractor, model, text_col='processed_text', scaler=None):
        if hasattr(extractor, 'encode'):
            self.extractor = extractor
        else:
            self.extractor = clone(extractor)

        self.model = clone(model)
        self.scaler = clone(scaler) if scaler else None
        self.text_col = text_col

    def fit(self, df_train, y_train, df_val=None, y_val=None):
        # KHÔNG GỌI HÀM balance_data NỮA -> Dùng 100% dữ liệu gốc
        texts = df_train[self.text_col].tolist()

        if hasattr(self.extractor, 'encode'):
            X_features = self.extractor.encode(texts, show_progress_bar=False)
        else:
            X_features = self.extractor.fit_transform(texts)

        if self.scaler:
            X_features = self.scaler.fit_transform(X_features)

        self.model.fit(X_features, y_train)

    def predict(self, df_test):
        texts = df_test[self.text_col].tolist()
        if hasattr(self.extractor, 'encode'):
            X_features = self.extractor.encode(texts, show_progress_bar=False)
        else:
            X_features = self.extractor.transform(texts)

        if self.scaler:
            X_features = self.scaler.transform(X_features)

        return self.model.predict(X_features)
    
class LSTMPipeline:
    def __init__(self, max_vocab=20000, max_len=128, embed_dim=128, text_col='processed_text'):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.text_col = text_col
        self.vectorizer = TextVectorization(max_tokens=max_vocab, output_mode='int', output_sequence_length=max_len)

    def fit(self, df_train, y_train, df_val, y_val):
        self.num_classes = len(np.unique(y_train))
        texts_train = df_train[self.text_col].to_numpy()
        texts_val = df_val[self.text_col].to_numpy()

        self.vectorizer.adapt(texts_train)

        y_train_idx = y_train - 1
        y_val_idx = y_val.values - 1

        # TÍNH TRỌNG SỐ LỚP ĐỂ XỬ LÝ MẤT CÂN BẰNG
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_idx),
            y=y_train_idx
        )
        class_weight_dict = dict(enumerate(weights))

        self.model = Sequential([
            Input(shape=(1,), dtype=tf.string),
            self.vectorizer,
            Embedding(input_dim=self.max_vocab, output_dim=self.embed_dim, mask_zero=True),
            LSTM(128, return_sequences=False),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        self.model.fit(
            texts_train, y_train_idx,
            validation_data=(texts_val, y_val_idx),
            epochs=10, batch_size=32,
            class_weight=class_weight_dict, # <-- Đưa trọng số vào huấn luyện
            callbacks=[early_stopping], verbose=1
        )

    def predict(self, df_test):
        texts_test = df_test[self.text_col].to_numpy()
        preds = self.model.predict(texts_test, verbose=0)
        return np.argmax(preds, axis=1) + 1

class BERT_LSTMPipeline:
    def __init__(self, model_name='distilbert-base-uncased', max_len=128, text_col='text'):
        self.model_name = model_name
        self.max_len = max_len
        self.text_col = text_col
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def fit(self, df_train, y_train, df_val, y_val):
        self.num_classes = len(np.unique(y_train))
        y_train_idx = y_train - 1
        y_val_idx = y_val.values - 1

        # Trọng số cân bằng lớp
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_idx),
            y=y_train_idx
        )
        class_weight_dict = dict(enumerate(weights))

        print("Đang Tokenize dữ liệu cho Hybrid BERT + LSTM...")
        train_encodings = self.tokenizer(df_train[self.text_col].tolist(), truncation=True, padding=True, max_length=self.max_len, return_tensors='tf')
        val_encodings = self.tokenizer(df_val[self.text_col].tolist(), truncation=True, padding=True, max_length=self.max_len, return_tensors='tf')

        train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train_idx)).shuffle(10000).batch(16)
        val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), y_val_idx)).batch(16)

        # 1. KIẾN TRÚC HYBRID
        input_ids = Input(shape=(self.max_len,), dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=(self.max_len,), dtype=tf.int32, name="attention_mask")

        # Load BERT gốc (không có đầu phân loại)
        bert_model = TFAutoModel.from_pretrained(self.model_name, use_safetensors=False)
        
        # ĐÓNG BĂNG BERT
        bert_model.trainable = False 

        # Lấy BERT Sequence (Vector của từng từ)
        bert_output = bert_model(input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state # Shape: (batch_size, max_len, 768)

        # Đẩy Sequence vào LSTM
        lstm_out = LSTM(128, return_sequences=False)(sequence_output)
        dropout_out = Dropout(0.5)(lstm_out)
        final_output = Dense(self.num_classes, activation='softmax')(dropout_out)

        self.model = tf_keras.Model(inputs=[input_ids, attention_mask], outputs=final_output)

        optimizer = tf_keras.optimizers.Adam(learning_rate=1e-3) # Tốc độ học lớn hơn một chút vì BERT đã đóng băng
        loss = tf_keras.losses.SparseCategoricalCrossentropy()

        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        early_stopping = tf_keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

        self.model.fit(
            train_dataset, validation_data=val_dataset,
            epochs=5,
            class_weight=class_weight_dict,
            callbacks=[early_stopping], verbose=1
        )

    def predict(self, df_test):
        test_encodings = self.tokenizer(df_test[self.text_col].tolist(), truncation=True, padding=True, max_length=self.max_len, return_tensors='tf')
        test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings))).batch(32)
        preds = self.model.predict(test_dataset)
        return np.argmax(preds, axis=1) + 1
    

class TFBERTPipeline:
    def __init__(self, model_name='distilbert-base-uncased', max_len=128, text_col='text'):
        self.model_name = model_name
        self.max_len = max_len
        self.text_col = text_col
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def fit(self, df_train, y_train, df_val, y_val):
        self.num_classes = len(np.unique(y_train))
        y_train_idx = y_train - 1
        y_val_idx = y_val.values - 1

        # TÍNH TRỌNG SỐ LỚP CHO BERT
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_idx),
            y=y_train_idx
        )
        class_weight_dict = dict(enumerate(weights))

        print("Đang Tokenize dữ liệu cho HuggingFace...")
        train_encodings = self.tokenizer(df_train[self.text_col].tolist(), truncation=True, padding=True, max_length=self.max_len, return_tensors='tf')
        val_encodings = self.tokenizer(df_val[self.text_col].tolist(), truncation=True, padding=True, max_length=self.max_len, return_tensors='tf')

        train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train_idx)).shuffle(10000).batch(16)
        val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), y_val_idx)).batch(16)

        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_classes, use_safetensors=False
        )

        import tf_keras
        optimizer = tf_keras.optimizers.Adam(learning_rate=2e-5)
        loss = tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        early_stopping = tf_keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

        self.model.fit(
            train_dataset, validation_data=val_dataset,
            epochs=3,
            class_weight=class_weight_dict, # <-- Đưa trọng số vào huấn luyện
            callbacks=[early_stopping], verbose=1
        )

    def predict(self, df_test):
        test_encodings = self.tokenizer(df_test[self.text_col].tolist(), truncation=True, padding=True, max_length=self.max_len, return_tensors='tf')
        test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings))).batch(32)
        preds = self.model.predict(test_dataset).logits
        return np.argmax(preds, axis=1) + 1