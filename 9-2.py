import pickle
import re
import os
from movieclassifier.vectorizer import vect
import sqlite3
conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()
# c.execute('create table review_db (review Text,sentiment integer,date text)')
# example1 = 'I love this movie'
# c.execute('insert into review_db (review,sentiment,date) values(?,?,DATETIME("now"))',(example1,1))
# example2 = 'I disliked this movie'
# c.execute('insert into review_db (review,sentiment,date) values(?,?,DATETIME("now"))',(example2,0))
# conn.commit()

c.execute("select * from review_db where date between '2015-01-01 00:00:00' and DATETIME('now')")
results = c.fetchall()
print(results)
conn.close()


