import pandas as pd
import numpy as np
from scipy import stats
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.simplefilter('ignore')
sns.set_theme('talk', rc={'figure.figsize':(12, 10)})
import wrangle_qualtrics as wq

df_original = pd.read_csv("added_bots.csv")
df_original.head()

df = wq.clean_qualtrics_data(df_original)
df = wq.grouping(df)
df = wq.combine_R_and_Rep(df)
df = wq.average_scale_scores(df)

df.head()

# make a dataframe consist of differences of panas scores between before and 
# after chatting with the different types of bots
df_diff = wq.make_df_of_diff(
    df, ['panas_pos', 'panas_neg', 'competence', 'warmness', 'usability', 
         "willingness", "understanding", "Q53", "Q55", "Q57"]
)
df_diff.head()
df_diff_cols = pd.Series(df_diff.columns)
diff_columns = [diff for diff in df_diff.columns if "diff" in diff]

def ttest(df, col, group, group1, group2, paired=False):
    a = df[df[group] == group1][col]
    b = df[df[group] == group2][col]
    if paired == True:
        res = stats.ttest_rel(a, b)
    else:
        res = stats.ttest_ind(a, b, equal_var=False)
    tval = res.statistic
    pval = res.pvalue
    if pval < .05:
        msg = "Significance"
    else:
        msg = "ns"
    
    print(f"t-value: {tval}, p-value: {pval}, {msg}")
    
def utest(df, col, group, group1, group2):
    a = df[df[group] == group1][col]
    b = df[df[group] == group2][col]
    res = stats.mannwhitneyu(a, b)
    statistics = res.statistic
    pval = res.pvalue
    if pval < .05:
        msg = "Significance"
    else:
        msg = "ns"
    
    print(f"statistics: {statistics}, p-value: {pval}, {msg}")

ttest(df, "competence", "ABEF_CDGH", "groupABEF", "groupCDGH")
ttest(df_diff, "competence_diff", "ABEF_CDGH", "groupABEF", "groupCDGH")
ttest(df_diff, "warmness_diff", "ABEF_CDGH", "groupABEF", "groupCDGH")
ttest(df_diff, "usability_diff", "ABEF_CDGH", "groupABEF", "groupCDGH")
ttest(df_diff, "competence_diff", "ABEF_CDGH", "groupABEF", "groupCDGH")

def plot_boxplots_diff(hue):
    for i in diff_columns:
        sns.boxplot(data=df_diff, y=i, hue=hue,).set(title=i)
        plt.show()

tmp1 = df_diff.query("ABEF_CDGH == 'groupABEF'")[diff_columns]
tmp2 = df_diff.query("ABEF_CDGH == 'groupCDGH'")[diff_columns]

### ------------------------------------------------------------
sns.heatmap(tmp1.corr(), annot=True).set(title="groupABEF diff")
sns.heatmap(tmp2.corr(), annot=True).set(title="groupCDGF diff")


mask1 = np.triu(np.ones_like(tmp1.corr(), dtype=bool))
sns.heatmap(tmp1.corr(), mask=mask1, square=True, annot=True).set(title="groupABEF diff")

mask2 = np.triu(np.ones_like(tmp2.corr(), dtype=bool))
sns.heatmap(tmp2.corr(), mask=mask2, square=True, annot=True).set(title="groupCDGH diff")

stats.pearsonr(tmp1["warmness_diff"], tmp1["panas_pos_diff"])
stats.pearsonr(tmp2["warmness_diff"], tmp2["panas_pos_diff"])
### ------------------------------------------------------------

plot_boxplots_diff(hue="group")
plot_boxplots_diff(hue="ABCD_EFGH")
plot_boxplots_diff(hue="ABEF_CDGH")
plot_boxplots_diff(hue="AB_CD_EF_GH")

utest(df_diff, "warmness_diff", "AB_CD_EF_GH", "groupEF", "groupGH")
utest(df_diff, "competence_diff", "AB_CD_EF_GH", "groupEF", "groupGH")
utest(df_diff, "usability_diff", "AB_CD_EF_GH", "groupEF", "groupGH")
utest(df_diff, "warmness_diff", "ABCD_EFGH", "groupABCD", "groupEFGH")
utest(df_diff, "competence_diff", "ABEF_CDGH", "groupABEF", "groupCDGH")

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

sns.histplot(data=df_diff, x="competence_diff", hue="ABEF_CDGH", multiple="dodge")
sns.histplot(data=df_diff, x="warmness_diff", hue="ABEF_CDGH", multiple="dodge")
sns.histplot(data=df_diff, x="usability_diff", hue="ABEF_CDGH", multiple="dodge")
group1 = df_diff.query('ABEF_CDGH == "groupABEF"')
group2 = df_diff.query("ABEF_CDGH == 'groupCDGH'")
stats.ttest_ind(group1["competence_diff"], group2["competence_diff"])
stats.ttest_ind(group1["warmness_diff"], group2["warmness_diff"])
stats.ttest_ind(group1["usability_diff"], group2["usability_diff"])


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
