import sqlite3
import csv
import datetime 

def run(binary=False):
    conn = sqlite3.connect('data.sqlite')

    c = conn.cursor()

    c.execute("DROP TABLE cases")
    c.execute("DROP TABLE cases10Day")
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            fips varchar(5) NOT NULL,
            `report-date` date NOT NULL,
            locality varchar(100), 
            cases integer, 
            hospitalizations integer,
            deaths integer,
            PRIMARY KEY(fips, `report-date`)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS cases10Day (
            fips varchar(5) NOT NULL,
            `report-date` date NOT NULL,
            locality varchar(100), 
            `num-days` bigint,
            c1 integer, 
            h1 integer,
            d1 integer,
            c2 integer, 
            h2 integer,
            d2 integer,
            c3 integer, 
            h3 integer,
            d3 integer,
            c4 integer, 
            h4 integer,
            d4 integer,
            c5 integer, 
            h5 integer,
            d5 integer,
            c6 integer, 
            h6 integer,
            d6 integer,
            c7 integer, 
            h7 integer,
            d7 integer,
            c8 integer, 
            h8 integer,
            d8 integer,
            c9 integer, 
            h9 integer,
            d9 integer,
            c10 integer, 
            h10 integer,
            d10 integer,
            PRIMARY KEY(fips, `report-date`)
        )
    """)


    fips_last_cases = {}

    fips_day = {}

    with open('VDHCovidCases.csv', 'r') as f:
        data = csv.reader(f)
        for row in data:
            fips = row[1]
            if not fips.startswith('51'):
                continue
            if fips in fips_last_cases: 
                curr_c = int(row[4]) - fips_last_cases[fips]['c']
                curr_h = int(row[5]) - fips_last_cases[fips]['h']
                curr_d = int(row[6]) - fips_last_cases[fips]['d']

                fips_last_cases[fips]['c'] += curr_c
                fips_last_cases[fips]['h'] += curr_h
                fips_last_cases[fips]['d'] += curr_d
            else:
                curr_c = int(row[4])
                curr_h = int(row[5])
                curr_d = int(row[4])
                fips_last_cases[fips] = {}
                fips_last_cases[fips]['c'] = curr_c
                fips_last_cases[fips]['h'] = curr_h
                fips_last_cases[fips]['d'] = curr_d

            sql = ''' INSERT INTO cases(fips, `report-date`, locality, cases, hospitalizations, deaths)
                    VALUES(?,?,?,?,?,?) '''
            cur = conn.cursor()
            curr_date = datetime.datetime.strptime(row[0], '%m/%d/%Y').strftime('%Y-%m-%d')
            if not fips in fips_day:
                fips_day[fips] = {}
            fips_day[fips][curr_date] = {
                'c': curr_c,
                'h': curr_h,
                'd': curr_d,
                'l': row[2]
            }
            cur.execute(sql, (row[1], curr_date, row[2], curr_c, curr_h, curr_d))


    d1 = datetime.date(2020, 3, 27)
    d2 = datetime.date(2020, 12, 1)

    dd = [d1 + datetime.timedelta(days=x) for x in range((d2-d1).days + 1)]

    for fips in fips_day:
        for date in dd:
            sql = ''' INSERT INTO cases10Day(fips, `report-date`, locality, `num-days`, c1,h1,d1, c2,h2,d2,
                                                                    c3,h3,d3, c4,h4,d4,
                                                                    c5,h5,d5, c6,h6,d6,
                                                                    c7,h7,d7, c8,h8,d8,
                                                                    c9,h9,d9, c10,h10,d10)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''
            date_fmt = date.strftime('%Y-%m-%d')
            epoch_time = (date - datetime.date(2020, 3, 17)).days
            locality = fips_day[fips][date_fmt]['l']
            data = [fips, date_fmt, locality, epoch_time]
            date_range = [date - datetime.timedelta(days=x) for x in range(0, 10)]
            for k in date_range:
                data_day = k.strftime('%Y-%m-%d')
                data.append(fips_day[fips][data_day]['c'])
                data.append(fips_day[fips][data_day]['h'])
                data.append(fips_day[fips][data_day]['d'])
            cur = conn.cursor()
            cur.execute(sql, tuple(data))

    conn.commit()
    conn.close()
