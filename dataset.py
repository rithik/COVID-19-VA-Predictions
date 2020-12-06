import sqlite3
import csv
import pandas as pd
import demographics
import unemployment
import covid

def export(binary=False): 
    conn = sqlite3.connect('data.sqlite')
    if binary: 
       df = pd.read_sql_query(""" SELECT fips, `report-date`, locality, `num-days`, c1,h1,d1, c2,h2,d2,
                                  c3,h3,d3, c4,h4,d4, c5,h5,d5, c6,h6,d6, c7,h7,d7, c8,h8,d8, c9,h9,d9, c10,h10,d10 
                                  `unemployment-percent`, `median-household-income`,`age0-14`, `age15-24`, `age25-34`, `age35-44`, `age45-54`, 
                                  `age55-64`, `age65-85`, white, black, `american-indian`, asian, total, c1 > c2 as greaterC, h1 > h2 as greaterH, d1 > d2 as greaterD
                                  FROM cases10Day NATURAL JOIN demographics NATURAL JOIN unemployment WHERE `report-date` != '2020-12-01'; """, conn) 
    else:
        df = pd.read_sql_query("SELECT * FROM cases10Day NATURAL JOIN demographics NATURAL JOIN unemployment WHERE `report-date` != '2020-12-01'; ", conn)
    df.to_excel('dataset{}.xlsx'.format("binary" if binary else ""))
    df.to_csv('dataset{}.csv'.format("binary" if binary else ""))

    conn.close()


if __name__ == "__main__":
    demographics.run()
    unemployment.run()
    covid.run()
    export(binary=True)
