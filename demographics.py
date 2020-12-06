import sqlite3

def run():
    with open('pcen_v2019_y1019.txt', 'r') as f:
        demographics_data = [line.rstrip() for line in f if line.startswith('201951')]

    conn = sqlite3.connect('data.sqlite')

    c = conn.cursor()

    c.execute("DROP TABLE demographics")
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS demographics (
            fips varchar(5) PRIMARY KEY,
            `age0-14` decimal(8,4), 
            `age15-24` decimal(8,4), 
            `age25-34` decimal(8,4), 
            `age35-44` decimal(8,4), 
            `age45-54` decimal(8,4), 
            `age55-64` decimal(8,4), 
            `age65-85` decimal(8,4), 
            white decimal(8,4),
            black decimal(8,4),
            `american-indian` decimal(8,4),
            asian decimal(8,4),
            total integer
        )
    """)

    fips_data = {}

    curr_fips = None
    fips_age_distribution = [0, 0, 0, 0, 0, 0, 0]
    fips_race_distribution = [0, 0, 0, 0]

    for row in demographics_data:
        fips = row[4:9]
        if curr_fips == None:
            curr_fips = fips
        if curr_fips != fips:
            fips_data[curr_fips] = {}
            fips_data[curr_fips]['age'] = fips_age_distribution
            fips_data[curr_fips]['race'] = fips_race_distribution
            curr_fips = fips
            fips_age_distribution = [0, 0, 0, 0, 0, 0, 0]
            fips_race_distribution = [0, 0, 0, 0]

        age = int(row[9:11])
        race = int(row[11])
        hispanic = row[12]
        num_people = int(row[93:101])

        if age < 15: 
            age_index = 0
        elif age < 65: 
            age_index = (age-5)//10
        else:
            age_index = 6

        if race in [1, 2]: # white
            race_index = 0
        elif race in [3, 4]: # black
            race_index = 1
        elif race in [5, 6]: # american indian
            race_index = 2
        else: # asian
            race_index = 3
        
        fips_age_distribution[age_index] += num_people
        fips_race_distribution[race_index] += num_people

    for district in fips_data:
        raw_age = fips_data[district]['age']
        raw_race = fips_data[district]['race']
        total_people = sum(raw_age)
        total_data = [age_group/total_people for age_group in raw_age]
        race_precents = [race_group/total_people for race_group in raw_race]

        total_data.extend(race_precents)
        total_data.insert(0, district)
        total_data.append(total_people)

        total_data_tuple = tuple(total_data)
        sql = ''' INSERT INTO demographics(fips, `age0-14`, `age15-24`, `age25-34`, `age35-44`, `age45-54`, 
                                                `age55-64`, `age65-85`, white, black, `american-indian`, asian, total)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) '''
        cur = conn.cursor()
        cur.execute(sql, total_data_tuple)

    conn.commit()
    conn.close()
