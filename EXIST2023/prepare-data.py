import dataprocessor

# preprocess training data
df_train = dataprocessor.preprocess_tweets_file('EXIST2023_training_task1_gold_hard.json', 'EXIST2023_training.json')
df_train.to_csv('preprocessed_train.csv', index=False)

# preprocess development data
df_dev = dataprocessor.preprocess_tweets_file('EXIST2023_dev_task1_gold_hard.json', 'EXIST2023_dev.json')
df_dev.to_csv('preprocessed_dev.csv', index=False)