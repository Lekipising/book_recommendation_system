import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib.backends.backend_pdf import PdfPages

df = pd.read_csv("books.csv", on_bad_lines="skip")
# print(df.head())


# find all null values
# print(df.isnull().sum())


# describe - scores are all between 0 and 5
# print(df.describe())

# top 10 rated books
top_ten = df[df["ratings_count"] > 1000000]
top_ten.sort_values(by="average_rating", ascending=False)
plt.style.use("seaborn-whitegrid")
plt.figure(figsize=(10, 10), dpi=80)
data = top_ten.sort_values(by="average_rating", ascending=False).head(10)
sns.barplot(x="average_rating", y="title", data=data, palette="brg")
# increase size
# show graph

# top authors
most_books = (
    df.groupby("authors")["title"]
    .count()
    .reset_index()
    .sort_values("title", ascending=False)
    .head(10)
    .set_index("authors")
)
plt.figure(figsize=(15, 10))
ax = sns.barplot(most_books["title"], most_books.index, palette="brg")
ax.set_title("Top 10 authors with most books")
ax.set_xlabel("Total number of books")
totals = []
for i in ax.patches:
    totals.append(i.get_width())
total = sum(totals)
for i in ax.patches:
    ax.text(
        i.get_width() + 0.2,
        i.get_y() + 0.2,
        str(round(i.get_width())),
        fontsize=15,
        color="black",
    )

# show graph


# books have been reviewed the most.
most_rated = (
    df.sort_values("ratings_count", ascending=False).head(
        10).set_index("title")
)
plt.figure(figsize=(15, 10))
ax = sns.barplot(most_rated["ratings_count"],
                 most_rated.index, palette="brg")
totals = []
for i in ax.patches:
    totals.append(i.get_width())
total = sum(totals)
for i in ax.patches:
    ax.text(
        i.get_width() + 0.2,
        i.get_y() + 0.2,
        str(round(i.get_width())),
        fontsize=15,
        color="black",
    )

# show graph


# find a relation between our average score and the number of scores.
df.average_rating = df.average_rating.astype(float)
fig, ax = plt.subplots(figsize=[15, 10])
sns.distplot(df["average_rating"], ax=ax)
ax.set_title("Average rating distribution for all books", fontsize=20)
ax.set_xlabel("Average rating", fontsize=13)

ax = sns.relplot(
    data=df,
    x="average_rating",
    y="ratings_count",
    color="red",
    sizes=(100, 200),
    height=7,
    marker="o",
)
plt.title("Relation between Rating counts and Average Ratings", fontsize=15)
ax.set_axis_labels("Average Rating", "Ratings Count")

plt.figure(figsize=(15, 10))
ax = sns.relplot(
    x="average_rating",
    y="  num_pages",
    data=df,
    color="red",
    sizes=(100, 200),
    height=7,
    marker="o",
)
ax.set_axis_labels("Average Rating", "Number of Pages")


plt.show()

# make copy
df2 = df.copy()

# create a new column called ‘rating_between’.df2.loc[ (df2['average_rating'] >= 0) & (df2['average_rating'] <= 1), 'rating_between'] = "between 0 and 1"
df2.loc[
    (df2["average_rating"] > 1) & (
        df2["average_rating"] <= 2), "rating_between"
] = "between 1 and 2"
df2.loc[
    (df2["average_rating"] > 2) & (
        df2["average_rating"] <= 3), "rating_between"
] = "between 2 and 3"
df2.loc[
    (df2["average_rating"] > 3) & (
        df2["average_rating"] <= 4), "rating_between"
] = "between 3 and 4"
df2.loc[
    (df2["average_rating"] > 4) & (
        df2["average_rating"] <= 5), "rating_between"
] = "between 4 and 5"

# split the language code column to retrieve these languages
# individually and give them the value of 1 and 0 also where 1 will be assigned
# if the book is written in a particular language eg English
# and 0 if it is not written in English:
rating_df = pd.get_dummies(df2["rating_between"])
language_df = pd.get_dummies(df2["language_code"])


# concatenate these two data frames into one and name it as features.
# It will contain the values of rating_df and language_df
# and will also have the values of average grade and number of grades:

features = pd.concat(
    [rating_df, language_df, df2["average_rating"], df2["ratings_count"]], axis=1
)

# use the Min-Max scaler to reduce these values.
# This will help reduce the bias for some of the books that have too many features.
# The algorithm will find the median for all and equalize it:
min_max_scaler = MinMaxScaler()
features = min_max_scaler.fit_transform(features)


# use the KNN algorithm to build our Book Recommendation system with Machine Learning
model = neighbors.NearestNeighbors(n_neighbors=6, algorithm="ball_tree")
model.fit(features)
dist, idlist = model.kneighbors(features)


# create a function
# When this function is called, we will have to pass the name of the book to it.
# The model will try to find books based on the features.
# We’ll store those book names that the system
# recommends in a list and return them at the end:


def BookRecommender(book_name):
    book_list_name = []
    book_id = df2[df2["title"] == book_name].index
    book_id = book_id[0]
    for newid in idlist[book_id]:
        book_list_name.append(df2.loc[newid].title)
    return book_list_name


BookNames = BookRecommender(
    "Harry Potter and the Half-Blood Prince (Harry Potter  #6)")

for book in BookNames:
    print(book)
