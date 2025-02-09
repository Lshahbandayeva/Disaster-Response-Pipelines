import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """This function gets filepaths of messages and categories, merges them and storem 
    it in a single DataFrame object.

    Args:
       messages_filepath: filepath of disaster_messages.csv
       categories_filepath: filepath of disaster_categories.csv

    Returns:
       df: a single DataFrame object that stores both messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, left_on = 'id', right_on = 'id', how = 'left')
    return df


def clean_data(df):
    """This function splits dataset into categories, drops duplicates.

    Args:
       df: raw DataFrame object

    Returns:
       df: cleaned DataFrame object
    """
    categories = df['categories'].str.split(';', expand = True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)
    
    df.drop(columns = ['categories'], axis=1, inplace = True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(keep=False, inplace=True)
    return df


def save_data(df, database_filename):
    """This function gets filename of database, and export the data into a table.

    Args:
       df: a DataFrame object in sqllite
       database_filename: name of database file

    Returns:
       none
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterTable', engine, index=False)

    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()