# Resume Classification Project

Welcome to the Resume Classification project! This project focuses on classifying resumes into different categories using a Fine-Tuned BERT model for text classification. The goal is to automate the categorization process and provide insights into the selection of the model, preprocessing techniques, and overall project flow.

## Table of Contents
- [Introduction](#introduction)
- [Chosen Model](#chosen-model-fine-tuned-bert-model)
- [Explanations for Not Choosing Other Models](#explanations-for-not-choosing-other-models)
- [Preprocessing and Feature Extraction](#preprocessing-and-feature-extraction)
- [Instructions for Running the Script](#instructions-for-running-the-script)
- [Expected Outputs](#expected-outputs)
- [Conclusion](#conclusion)

## Introduction

This project aims to categorize resumes into specific categories using advanced natural language processing techniques. By leveraging a Fine-Tuned BERT model, the project achieves high accuracy in classifying resumes. The README provides an in-depth overview of the chosen model, explanations for not selecting alternative models, preprocessing methods, feature extraction details, script execution instructions, expected outputs, and project conclusions.

## Chosen Model: Fine-Tuned BERT Model

The chosen model for this project is the Fine-Tuned BERT (Bidirectional Encoder Representations from Transformers) model. BERT's state-of-the-art performance in various text-related tasks, contextual understanding, and pre-trained language representation make it an ideal choice for accurate text classification.

## Explanations for Not Choosing Other Models

Several alternative models were considered, including Random Forest, Gradient Boosting Classifier, LightGBM, Voting Classifiers, and Bidirectional LSTM. The README explains the reasons for not selecting each of these models, providing insights into the decision-making process.

## Preprocessing and Feature Extraction

The preprocessing phase involves a series of steps to clean and prepare the text data for model input. These steps include HTML tag removal, special character elimination, digit removal, link removal, email address elimination, punctuation removal, non-ASCII character elimination, and text conversion to lowercase. The README highlights the importance of preprocessing for data quality.

Feature extraction encompasses leveraging the contextual embeddings generated by the Fine-Tuned BERT model. These embeddings capture rich semantic information, enabling the model to understand relationships between words and phrases.

## Instructions for Running the Script

To run the script and categorize resumes using the Fine-Tuned BERT model:

1) Make sure you have the necessary packages installed. Using the given “requirements.txt” file, you can install them.
2) Download the resume_parser.h5 file from the given link (https://drive.google.com/file/d/1z5ElluzXXeCx9-HK60dDVerwFwwaDzTO/view?usp=sharing)
3) 'Category' and 'Resume' columns should be included in a CSV file that contains the resumes to be categorized as input data. You can get those Input CSV files from the git public folder.
4) Execute the script.py program. Indicate the output directory where the categorized resumes will be saved as well as the path to the input CSV file.

The script will sort each resume into its appropriate category folder after processing each one using the Fine-Tuned BERT model. The resumes will be text files that have been classified. Additionally, a categorized_resumes.csv file with filenames and anticipated categories will be created.


## Expected Outputs

The script processes each resume using the Fine-Tuned BERT model and categorizes them into respective folders. Additionally, a `categorized_resumes.csv` file is generated, containing filenames and predicted categories.

## Conclusion

The project showcases the effectiveness of the Fine-Tuned BERT model in automating resume categorization. The documentation provides comprehensive insights into the chosen model, preprocessing, feature extraction, script execution, and outcomes.

---
