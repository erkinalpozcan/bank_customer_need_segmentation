import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import re


def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text



file_path = r"C:\Users\Erkinalp\Desktop\piton\NLP\bank_user_data.csv"
df = pd.read_csv(file_path)

mesaj=input("Yapmak İstediniz İşlemi Giriniz.")

mesajdf=pd.DataFrame({"sorgu":mesaj,"label":0},index=[42])

df=pd.concat([df,mesajdf],ignore_index=True)

df['Sorular'] = df['Sorular'].apply(preprocess_text)

# Stopword'ler çıkarılıyor
stopwords = ['fakat', 'lakin', 'ancak', 'acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey',
             'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 
             'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 
             'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 
             'veya', 'ya', 'yani']
for word in stopwords:
    word = " " + word + " "
    df['Sorular'] = df['Sorular'].str.replace(word, " ", regex=False)


label_encoder = LabelEncoder()
df['Kategori'] = label_encoder.fit_transform(df['Kategori'])


X = df['Sorular']
y = df['Kategori']


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128

def tokenize_texts(texts, max_len):
    return tokenizer(
        list(texts),
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

X_tokenized = tokenize_texts(X, max_len)


X_input_ids = X_tokenized['input_ids'].numpy()
X_attention_mask = X_tokenized['attention_mask'].numpy()


X_train_input_ids, X_test_input_ids, X_train_attention_mask, X_test_attention_mask, y_train, y_test = train_test_split(
    X_input_ids, X_attention_mask, y, test_size=0.2, random_state=42
)

X_train_input_ids = tf.convert_to_tensor(X_train_input_ids)
X_test_input_ids = tf.convert_to_tensor(X_test_input_ids)
X_train_attention_mask = tf.convert_to_tensor(X_train_attention_mask)
X_test_attention_mask = tf.convert_to_tensor(X_test_attention_mask)


y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)


model = TFBertForSequenceClassification.from_pretrained(
    'dbmdz/bert-base-turkish-cased', num_labels=len(label_encoder.classes_)
)


optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


history = model.fit(
    [X_train_input_ids, X_train_attention_mask],
    y_train,
    validation_split=0.2,
    epochs=8,
    batch_size=16
)


y_pred_logits = model.predict([X_test_input_ids, X_test_attention_mask]).logits
y_pred = tf.argmax(y_pred_logits, axis=1).numpy()


accuracy = accuracy_score(y_test.numpy(), y_pred)
print("Doğruluk Skoru:", accuracy)
print("\nSınıflandırma Raporu:\n", classification_report(y_test.numpy(), y_pred, target_names=label_encoder.classes_))
