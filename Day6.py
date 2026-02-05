import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Marks_Data.csv")

# Mean percentage of Roma
m = df.loc[df['Student_Name'] == 'Roma',
        ['Maths', 'Science', 'English']].mean(axis=1).iloc[0]

print("Percentage:", m, "%")

if m < 45:
    print("Fail")
else:
    print("Pass")

# Total & Rank
df['Total'] = df[['Maths','Science','English']].sum(axis=1)
df['Rank'] = df['Total'].rank(ascending=False)

print(df)

# Plot
plt.figure()
df.plot(x='Student_Name', y=['Maths','Science','English'], kind='bar')
plt.show()
