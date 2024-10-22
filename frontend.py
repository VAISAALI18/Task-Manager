import os
import pandas as pd
import re
import spacy
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load spaCy model for POS and NER
nlp = spacy.load('en_core_web_lg')

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)

# Load datasets
df = pd.read_csv('tasks_with_date_time.csv')
location_df = pd.read_excel('final_KB.xlsx')
location_list = location_df['Locations'].tolist()

# TF-IDF feature extraction
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['Task']).toarray()
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 1. Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 2. POS tagging and NER function
def get_pos_and_entities(text):
    doc = nlp(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return pos_tags, entities

# 3. Core task extraction function using dependency parsing
def extract_core_task(text):
    doc = nlp(text)
    root_action = None
    important_nouns = []
    
    for token in doc:
        if token.dep_ == 'ROOT':
            root_action = token.lemma_
        elif token.dep_ in ['dobj', 'nsubj', 'iobj', 'compound'] and token.pos_ == 'NOUN':
            important_nouns.append(token.text)
    
    task_summary = root_action if root_action else ""
    if important_nouns:
        task_summary += " " + " ".join(important_nouns[:2])
    
    task_summary = re.sub(r'\b(?:tommorrow|today|yesterday|am|pm|floor|th|[0-9])\b', '', task_summary).strip()
    return task_summary.strip()

# 4. Extract date, time, location, and core task
def extract_chunks(text):
    doc = nlp(text)
    date, time, location = None, None, None
    
    for ent in doc.ents:
        if ent.label_ == "DATE":
            date = ent.text
        elif ent.label_ == "TIME":
            time = ent.text
        elif ent.label_ in ["GPE", "LOC"]:
            location = ent.text
    
    if not date:
        date_pattern = r'(\b\d{1,2}\s*(?:st|nd|rd|th)?\s*(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|[A-Z][a-z]+)\s*\d{0,4})'
        date_match = re.search(date_pattern, text)
        date = date_match.group(0) if date_match else None

    if not time:
        time_pattern = r'(\b\d{1,2}:\d{2}\s*[APap][Mm]|\b\d{1,2}\s*(?:am|pm|in the morning|in the afternoon|in the evening|at night)\b)'
        time_match = re.search(time_pattern, text)
        time = time_match.group(0) if time_match else None

    if not location:
        normalized_text = text.lower()
        for loc in location_list:
            if loc.lower() in normalized_text:
                location = loc
                break

    core_task = extract_core_task(text)
    return {"date": date, "time": time, "location": location, "task": core_task}

# 5. Classify and save the extracted data to Excel
def classify_and_save(user_input):
    processed_input = preprocess_text(user_input)
    input_vector = tfidf.transform([processed_input]).toarray()
    prediction = model.predict(input_vector)[0]
    
    chunks = extract_chunks(processed_input)
    
    # Prepare data for saving
    extracted_data = {
        "Task Description": user_input,  # Changed from "Task" to "Task Description"
        "Category": prediction,
        "Date": chunks['date'],
        "Time": chunks['time'],
        "Location": chunks['location'],
        "Core Task": chunks['task']
    }

    # Append data to Excel file
    output_file = "extracted_tasks.xlsx"
    new_data = pd.DataFrame([extracted_data])
    
    try:
        # Attempt to read the existing Excel file
        existing_data = pd.read_excel(output_file)
        # Append the new data
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    except FileNotFoundError:
        # If the file doesn't exist, just use the new data
        updated_data = new_data

    # Save the updated data back to Excel
    updated_data.to_excel(output_file, index=False)

    # Generate instructions using GPT-2
    instruction_prompt = f"Provide step-by-step instructions for {user_input}:"
    inputs = tokenizer.encode(instruction_prompt, return_tensors='pt')
    outputs = gpt2_model.generate(
        inputs, 
        max_length=500,         
        num_return_sequences=1, 
        no_repeat_ngram_size=2,  
        do_sample=True,          
        top_k=50,                
        top_p=0.95,             
        temperature=0.7          
    )

    # Decode the generated output into text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return prediction, chunks, generated_text

# Main Streamlit app
def main():
    st.title("Task Management App")

    # Unique key argument added to each radio button
    selected_page = st.sidebar.radio("Select Page", ["About", "Task Input", "Task Search"], key="page_selector")

    # About page
    if selected_page == "About":
        st.header("About the App")
        st.write("This application helps you manage tasks by extracting key information such as date, time, location, and core task.")
        st.write("It uses advanced NLP techniques to classify tasks and generate instructions.")
        
    # Task Input page
    elif selected_page == "Task Input":
        st.header("Input Task Description")
        user_input = st.text_area("Enter your task description:")
        
        if st.button("Submit"):
            if user_input:
                prediction, chunks, generated_text = classify_and_save(user_input)
                st.success(f"Task classified as: **{prediction}**")
                st.write(f"**Extracted Data:**")
                st.write(chunks)
                st.write(f"**Generated Instructions:**\n{generated_text}")
            else:
                st.error("Please enter a task description.")

    # Task Search page
    else:
        st.header("Search Tasks by Category")
        
        # Load the extracted tasks DataFrame
        try:
            extracted_df = pd.read_excel("extracted_tasks.xlsx")
        except FileNotFoundError:
            st.error("No extracted tasks file found. Please add a task first.")
            extracted_df = pd.DataFrame(columns=['Task Description', 'Category', 'Date', 'Time', 'Location', 'Core Task'])

        if not extracted_df.empty:
            category = st.selectbox("Select Category", extracted_df['Category'].unique(), key="category_selector")
        
            if st.button("Search"):
                tasks = extracted_df[extracted_df['Category'] == category]
                
                
                
                if not tasks.empty:
                    st.write(f"**Tasks in Category: {category}**")
                    try:
                        st.write(tasks[['Task Description', 'Date', 'Time', 'Location', 'Core Task']])  # Updated column name to "Task Description"
                    except KeyError as e:
                        st.error(f"KeyError: {e} - Ensure the DataFrame has the expected columns.")
                else:
                    st.write(f"No tasks found for category: {category}")
        else:
            st.write("No tasks available. Please add a task first.")

if __name__ == "__main__":
    main()
