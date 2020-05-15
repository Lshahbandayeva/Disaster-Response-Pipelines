# Disaster Response Pipeline Project
A web app where an emergency worker can input a new message and get classification results in several categories.

# File Descriptions
data/process_data.py: ETL Pipeline takes the file paths of the two datasets  and database, datatset is provided by [Figure Eight](https://www.figure-eight.com/ "Figure Eight"), cleans the datasets, and stores the clean data into a SQLite database in the specified database file path.

model/train_classifier.py: ML Pipeline takes the database file path and model file path, creates and build GridSearchCV model, and stores the classifier into a pickle file to the specified model file path. 

app/run.py: A web app load the model and shows graphs which demonstrate the results.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# Results
The main page of web app includes two visualizations using data from the SQLite database.

# Acknowledgement
Dataset: [Figure Eight](https://www.figure-eight.com/ "Figure Eight").

Starter Code: [Udacity](https://www.udacity.com/ "Udacity"), Data Scientist Nanodegree Program
