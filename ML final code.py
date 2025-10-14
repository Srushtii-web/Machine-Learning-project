import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

data = pd.read_csv('athlete_events.csv.zip')
print("Dataset loaded successfully!")
print("Shape:", data.shape)
print(data[['Event', 'Sport']].head())

data = data.dropna(subset=['Event', 'Sport'])

X = data['Event']
y = data['Sport']

print("\nClass Distribution:")
print(y.value_counts())

counts = y.value_counts()
valid_classes = counts[counts >= 2].index
mask = y.isin(valid_classes)
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("\n📈 Model Performance:")  # This may still fail if encoding isn't fixed
print(f"Accuracy: {accuracy:.2f}")
if accuracy == 1.0:
    print("\n⚠️ Warning: Perfect accuracy is unusual. Please check that your train/test split is correct and there is no data leakage.")
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

plt.figure(figsize=(16, 6))
y.value_counts().plot(kind='bar')
plt.title('Distribution of Sports in Dataset')
plt.xlabel('Sport')
plt.ylabel('Count')
plt.xticks(rotation=90)

plt.text(
    0.95, 0.95, f'Accuracy: {accuracy:.2f}',
    transform=plt.gca().transAxes,
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
)

plt.tight_layout()
plt.show()

