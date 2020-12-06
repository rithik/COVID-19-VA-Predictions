import sqlite3
import csv

def run():
    conn = sqlite3.connect('data.sqlite')

    c = conn.cursor()

    c.execute("DROP TABLE unemployment")

    c.execute("""
        CREATE TABLE IF NOT EXISTS unemployment (
            fips varchar(5) PRIMARY KEY,
            `unemployment-percent` decimal(8,4), 
            `median-household-income` integer
        )
    """)

    with open('Unemployment.csv', 'r') as f:
        data = csv.reader(f)
        for row in data:
            if not row[0].startswith('51'):
                continue
            sql = ''' INSERT INTO unemployment(fips, `unemployment-percent`, `median-household-income`)
                    VALUES(?,?,?) '''
            cur = conn.cursor()
            cur.execute(sql, (row[0], float(row[10]), int(row[11][1:].replace(",", ""))))

    conn.commit()
    conn.close()
