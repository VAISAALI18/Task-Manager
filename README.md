# Task Management App

## Overview

The **Task Management App** is designed to help users manage their tasks by extracting key information such as date, time, location, and core tasks from natural language descriptions. The app utilizes various NLP techniques, including Part of Speech (POS) tagging, Named Entity Recognition (NER), and dependency parsing to process tasks. It also generates step-by-step instructions using a pre-trained GPT-2 model.

## Features

- **Task Input:** Allows users to input a task description, and the app will classify the task, extract date, time, location, and core task information, and save it to an Excel file.
- **Task Search:** Enables users to search for previously saved tasks by category and view extracted details.
- **Task Classification:** Uses a Logistic Regression model to classify tasks into categories such as work, personal, etc.
- **Instructions Generation:** Leverages a pre-trained GPT-2 model to generate instructions for each task.

## Technologies Used

- **Python** for data processing, modeling, and app development
- **spaCy** for NLP tasks like POS tagging and NER
- **scikit-learn** for classification using Logistic Regression
- **Streamlit** for building the web application interface
- **Transformers (Hugging Face)** for the GPT-2 model to generate instructions
- **pandas** for data handling and Excel export

