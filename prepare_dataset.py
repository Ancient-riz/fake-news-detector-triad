import pandas as pd

# Load files
df_fake = pd.read_csv('Fake.csv')
df_real = pd.read_csv('True.csv')

# Add labels
df_fake['label'] = 1
df_real['label'] = 0

# Combine
df = pd.concat([df_fake[['text', 'label']], df_real[['text', 'label']]])

# Shuffle and save
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('fake_news_dataset.csv', index=False)

print("✅ Dataset prepared and saved as fake_news_dataset.csv")
