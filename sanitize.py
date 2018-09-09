import pandas as pd
import sys

# Used to "sanitize"/convert the datasets into a format our regular program could use.
# Replaces categorical data with dummy variables, missing data with placeholders,
# and swaps rows around if necessary.

# Warning: This is kind of a hacky file that we changed to fit whatever data we were
# working with.

if (len(sys.argv) != 2):
	raise Exception('No file/output provided')

df = pd.read_csv(sys.argv[1], sep=',')

print (df)

df = df.drop('ID', axis=1)

df = df.fillna(df.mean())
# dropped = df.loc[:,df.columns[-1]]
# df = df.drop(df.columns[len(df.columns)-1], axis = 1)
# df.insert(0, 0, dropped)
print (df)

df = pd.get_dummies(df)
print (df)

df.to_csv('sanitized_output.csv', sep='\t', header=False, index=False)
