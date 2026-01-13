# ✅ Import libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.neighbors import NearestNeighbors 
 
# ✅ Step 1: Enter your CSV file path 
# Example 1: If books.csv is in the same folder → "books.csv" 
csv_path = input("Enter CSV file path: ") 
 
# ✅ Step 2: Load data from CSV 
df = pd.read_csv(csv_path)   
print(" CSV Loaded Successfully!\n") 
print(df.head()) 
 
# ✅ Step 3: Create user-item matrix (pivot table) 
book_matrix = df.pivot_table(index='Book', columns='UserID', values='Rating').fillna(0) 
 
# ✅ Step 4: Train the KNN model 
model = NearestNeighbors(metric='cosine', algorithm='brute') 
model.fit(book_matrix) 
 
# ✅ Step 5: Define recommendation function 
def recommend_books(book_name, k=5): 
    book_name = book_name.strip() 
 
    if book_name not in book_matrix.index: 
        print(f"❌ Book '{book_name}' not found in the dataset.") 
        return 
 
    distances, indices = model.kneighbors([book_matrix.loc[book_name]], n_neighbors=k+1) 
 
    recommended_books = [] 
    similarity_scores = [] 
 
    print(f"\n✅ Books similar to '{book_name}':\n") 
    for i in range(1, len(distances[0])): 
        similar_book = book_matrix.index[indices[0][i]] 
        similarity = 1 - distances[0][i] 
        recommended_books.append(similar_book) 
        similarity_scores.append(similarity) 
        print(f"{i}. {similar_book} (Similarity Score: {similarity:.2f})") 
 
    # ✅Plot similarity chart 
    if recommended_books: 
        plt.figure(figsize=(8, 5)) 
        bars = plt.bar(recommended_books, similarity_scores) 
        plt.title(f"📈Similarity Scores for '{book_name}' Recommendations") 
        plt.xlabel("Recommended Books") 
        plt.ylabel("Similarity Score (0-1)") 
        plt.ylim(0, 1) 
        plt.grid(axis='y', linestyle='--', alpha=0.7) 
 
        for bar, score in zip(bars, similarity_scores): 
            plt.text(bar.get_x() + bar.get_width()/2, 
                     bar.get_height() + 0.02, 
                     f"{score:.2f}", 
                     ha='center', va='bottom', fontsize=10) 
 
        plt.show() 
 
# ✅ Step 6: Ask user for book name 
book_name = input("Enter a book name to get recommendations: ") 
recommend_books(book_name, k=3)