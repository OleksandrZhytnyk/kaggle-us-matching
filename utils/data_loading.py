import pandas as pd


def create_dataframe(train_path, titles_path):
    train_df = pd.read_csv(train_path)
    titles = pd.read_csv(titles_path)
    train_df = train_df.merge(titles, left_on='context', right_on='code')
    train_df['input'] = train_df['anchor'] + ' ' + train_df['title']
    return train_df
