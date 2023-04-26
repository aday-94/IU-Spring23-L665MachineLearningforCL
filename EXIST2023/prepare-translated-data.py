import pandas as pd
import datatranslator


# read preprocessed tweets from CSV files
df_train = pd.read_csv('preprocessed_train.csv')
df_dev = pd.read_csv('preprocessed_dev.csv')


# translate tweets to the target language
target_lang = 'es' # or 'en' for English
df_train['translated_text'] = df_train['text'].apply(lambda x: datatranslator.translate_text(x, target_lang))
df_dev['translated_text'] = df_dev['text'].apply(lambda x: datatranslator.translate_text(x, target_lang))

# Save translated tweets to new CSV files
df_train.to_csv('translated_train_' + target_lang + '.csv', index=False)
df_dev.to_csv('translated_dev_' + target_lang + '.csv', index=False)
