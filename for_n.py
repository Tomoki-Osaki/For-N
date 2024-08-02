import wrangle_qualtrics as wq
import pandas as pd
import numpy as np
from scipy import stats
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.simplefilter('ignore')
sns.set_theme('talk', rc={'figure.figsize':(12, 8)})

df = pd.read_csv("added_bots.csv")
df.head()

df = wq.clean_qualtrics_data(df)
df = wq.grouping(df)
df = wq.average_scale_scores(df)

df.head()

panas = [f'Q11_{i}' for i in range(1, 17)]
df_panas = df[panas]

first_df = df.query('exp_count == 1')
second_df = df.query('exp_count == 2')

df_panas = first_df[panas]
df_panas2 = second_df[panas]

df_comp = first_df[["competence", "ABEF_CDGH"]]
df_warm = first_df[["warmness", "ABEF_CDGH"]]
df_use = first_df[["usability", "ABEF_CDGH"]]

df_comp2 = second_df[["competence", "ABEF_CDGH"]]
df_warm2 = second_df[["warmness", "ABEF_CDGH"]]
df_use2 = second_df[["usability", "ABEF_CDGH"]]

"""
The effects to see by comparing: 
ABCD and EFGH:
    The effects of episodic chatting (R+).
ABEF and CDGH:
    The effects of order of Control and Rug(+).
AB and CD:
    The effects of order of Control bot or Rug bot in R.
EF and GH:
    The effects of order of Control bot or Rug+ bot in R+.
AB and EF:
    The effects of episodic chatting (R+) in order of C to R.
CD and GH:
    The effects of episodic chatting (R+) in order of R to C.
"""

sns.boxplot(data=df, y='panas_pos', hue='group'); plt.show()
sns.boxplot(data=df, y='panas_pos', hue='ABCD_EFGH'); plt.show()
sns.boxplot(data=df, y='panas_pos', hue='AB_CD_EF_GH'); plt.show()

sns.boxplot(data=df, y='panas_neg', hue='group'); plt.show()
sns.boxplot(data=df, y='panas_neg', hue='ABCD_EFGH'); plt.show()
sns.boxplot(data=df, y='panas_neg', hue='AB_CD_EF_GH'); plt.show()

# make a dataframe consist of differences of panas scores between before and 
# after chatting with the different types of bots
df_diff = wq.make_df_of_diff(df, ['panas_pos', 'panas_neg'])
df_diff.head()

sns.boxplot(data=df_diff, y='panas_pos_diff', hue='group'); plt.show()
sns.boxplot(data=df_diff, y='panas_pos_diff', hue='AB_CD_EF_GH'); plt.show()
sns.boxplot(data=df_diff, y='panas_pos_diff', hue='ABCD_EFGH'); plt.show()
sns.boxplot(data=df_diff, y='panas_pos_diff', hue='ABEF_CDGH'); plt.show()

sns.boxplot(data=df_diff, y='panas_neg_diff', hue='group'); plt.show()
sns.boxplot(data=df_diff, y='panas_neg_diff', hue='AB_CD_EF_GH'); plt.show()
sns.boxplot(data=df_diff, y='panas_neg_diff', hue='ABCD_EFGH'); plt.show()
sns.boxplot(data=df_diff, y='panas_neg_diff', hue='ABEF_CDGH'); plt.show()

stats.ttest_1samp(df_diff['panas_pos_diff'], 0)
stats.ttest_1samp(df_diff['panas_neg_diff'], 0)

groupAB = df_diff.query('AB_CD_EF_GH == "groupAB"')
groupCD = df_diff.query('AB_CD_EF_GH == "groupCD"')
groupEF = df_diff.query('AB_CD_EF_GH == "groupEF"')
groupGH = df_diff.query('AB_CD_EF_GH == "groupGH"')
groupABCD = df_diff.query('ABCD_EFGH == "groupABCD"')
groupEFGH = df_diff.query('ABCD_EFGH == "groupEFGH"')
groupABEF = df_diff.query('ABEF_CDGH == "groupABEF"')
groupCDGH = df_diff.query('ABEF_CDGH == "groupCDGH"')

stats.ttest_ind(groupAB['panas_pos_diff'], groupEF['panas_pos_diff'])
stats.ttest_ind(groupABCD['panas_pos_diff'], groupEFGH['panas_pos_diff'])
stats.ttest_ind(groupAB['panas_neg_diff'], groupEF['panas_neg_diff'])
stats.ttest_ind(groupABCD['panas_neg_diff'], groupEFGH['panas_neg_diff'])
