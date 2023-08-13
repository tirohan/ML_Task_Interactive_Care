import os
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFDistilBertForSequenceClassification
import re
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification
tf.keras.utils.get_custom_objects()["TFDistilBertForSequenceClassification"] = TFDistilBertForSequenceClassification
model = tf.keras.models.load_model('/content/resume_parser.h5')

tokenizer = AutoTokenizer.from_pretrained("manishiitg/distilbert-resume-parts-classify")
bert_model = TFDistilBertForSequenceClassification.from_pretrained("manishiitg/distilbert-resume-parts-classify", from_pt=True)

def clean_html(text):
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', text)

def remove_digits(text):
    return re.sub(r'\d+', '', text)

def remove_links(text):
    return re.sub(r'http\S+', '', text)

def remove_special_characters(text):
    return re.sub(r'[^\w\s]', '', text)

def punct(text):
    return re.sub(r'[!"#$%&()*+,-/:;<=>?@[\]^_`{|}~]', '', text)

def non_ascii(text):
    return ''.join(char for char in text if ord(char) < 128)

def email_address(text):
    return re.sub(r'\S*@\S*\s?', '', text)

def lower(text):
    return text.lower()

def preprocess_text(text):
    text = clean_html(text)
    text = remove_special_characters(text)
    text = remove_digits(text)
    text = remove_links(text)
    text = email_address(text)
    text = punct(text)
    text = non_ascii(text)
    text = lower(text)
    return text

def categorize_and_move(input_csv, output_dir):
    categorized_resumes = []

    resume_df = pd.read_csv(input_csv)
    
    for index, row in resume_df.iterrows():
        resume_text = row['Resume']
        category = row['Category']
        
        preprocessed_resume = preprocess_text(resume_text)
        encoded_resume = tokenizer(preprocessed_resume, padding=True, truncation=True, max_length=max_resume_len, return_tensors='tf')
        category_prediction = model.predict({'input_ids': encoded_resume['input_ids'], 'attention_mask': encoded_resume['attention_mask']})
        predicted_category = np.argmax(category_prediction, axis=1)[0]
        category_folder = os.path.join(output_dir, str(predicted_category))
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)
        new_file_path = os.path.join(category_folder, f"{index}_{row['Category']}.txt")
        with open(new_file_path, 'w') as file:
            file.write(resume_text)

        categorized_resumes.append({'filename': f"{index}_{row['Category']}.txt", 'category': category})

    return categorized_resumes

input_csv = '/content/drive/MyDrive/Resume Data/UpdatedResumeDataSet.csv'
output_directory = '/content/drive/MyDrive/Resume Data'
max_resume_len = 200

categorized_resumes = categorize_and_move(input_csv, output_directory)
categorized_df = pd.DataFrame(categorized_resumes)
categorized_df.to_csv('categorized_resumes.csv', index=False)