from pyspark import SparkContext

sc = SparkContext("local", "WordCount")

file_rdd = sc.textFile("/home/adduser/book1.txt")

stopwords_rdd = sc.textFile("/home/adduser/stopwords.txt")

stopwords_broadcast = sc.broadcast(stopwords_rdd.collect())

def filter_stopwords(word):
    return word.lower() not in stopwords_broadcast.value

words_rdd = file_rdd.flatMap(lambda line: line.lower().split()).filter(filter_stopwords).map(lambda word: (word, 1))

word_counts = words_rdd.reduceByKey(lambda x, y: x + y)

# Sort the word counts by the count value in descending order
sorted_word_counts = word_counts.sortBy(lambda x: x[1], ascending=False)

# Collect the top 25 words
top_25_words = sorted_word_counts.take(25)

# Print the top 25 words
for word, count in top_25_words:
    print(f"{word}: {count}")

sc.stop()
