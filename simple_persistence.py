#!/usr/bin/python
import MySQLdb

db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                     user="root",         # your username
                     passwd="Susu140abf8291",  # your password
                     db="imagerec")        # name of the data base

# you must create a Cursor object. It will let
#  you execute all the queries you need
cur = db.cursor()

# Use all the SQL you like
cur.execute("SHOW tables")

# print all the first cell of all the rows
print(cur.fetchall())

db.close()
