import pandas as pd

df = pd.read_csv("data/ielts_writing_dataset.csv")

df = df.dropna(subset=['Essay','Overall'])
df['Essay'] = df['Essay'].str.replace('\n',' ').str.strip()

df['word_count'] = df['Essay'].apply(lambda x: len(x.split()))
df['char_count'] = df['Essay'].apply(len)

print(df[['Essay','word_count','char_count']].head())

df.to_csv("data/ielts_clean.csv", index=False)
print("Cleaned dataset saved as ielts_clean.csv")