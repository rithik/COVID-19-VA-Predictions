import pandas as pd


import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv').sort_values(by=['report-date'])
df = df[df['report-date'] == '2020-11-30']

df = df[df['fips'] <= 51060]
x = df[['fips', 'age0-14', 'age15-24', 'age25-34', 'age35-44', 'age45-54', 'age55-64', 'age65-85']].to_numpy().astype('float32')

labels = df['fips'].to_numpy()

a0 = df['age0-14'].to_numpy()
a1 = df['age15-24'].to_numpy()
a2 = df['age25-34'].to_numpy()
a3 = df['age35-44'].to_numpy()
a4 = df['age45-54'].to_numpy()
a5 = df['age55-64'].to_numpy()
a6 = df['age65-85'].to_numpy()

width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(labels, a0, width, label='Age 0-14')
ax.bar(labels, a1, width, bottom=a0,
       label='Age 15-24')
ax.bar(labels, a2, width, bottom=a1,
       label='Age 25-34')
ax.bar(labels, a3, width, bottom=a2,
       label='Age 35-44')
ax.bar(labels, a4, width, bottom=a3,
       label='Age 45-54')
ax.bar(labels, a5, width, bottom=a4,
       label='Age 55-64')
ax.bar(labels, a6, width, bottom=a5,
       label='Age 65-85')


ax.set_ylabel('Age Precent')
ax.set_title('Age Breakdown by FIPS district')
ax.legend()

plt.show()


