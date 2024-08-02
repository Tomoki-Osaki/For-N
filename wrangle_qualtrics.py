import pandas as pd
from factor_analyzer import FactorAnalyzer
from collections import Counter

def clean_qualtrics_data(df_to_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df_to_clean : pd.DataFrame
        dataframe to clean that has columns named 'Q_11_{x}', 'user_id', 'exp_count'

    Returns
    -------
    df : pd.DataFrame
        dataframe after dropping unnecesarry rows and columns, and converting to numeric.
    """
    
    df = df_to_clean.copy()
    
    # drop unnecessary rows and columns
    df = df.loc[2:, 'Q11_1':]
    df = df.dropna()
    # drop user's data who chatted with the bot only once
    id_count = Counter(df['user_id'])
    drop_list = [user_id for user_id, num_ans in id_count.items() if num_ans == 1]
    df = df.query('user_id not in @drop_list')
    df = df.sort_values(by=['user_id'], ignore_index=True)
    
    # make data numerical in order to compute them
    df['Q4'] = pd.to_numeric(df['Q4'])
    df['Q4'].replace(6, 5, inplace=True)
    
    for i in range(1, 17):
        df[f'Q11_{i}'] = pd.to_numeric(df[f'Q11_{i}'])
    for j in range(43, 62, 2):
        df[f'Q{j}'] = pd.to_numeric(df[f'Q{j}'])
        df[f'Q{j}'].replace(6, 5, inplace=True)
        
    # competence = [f'Q{i}' for i in [4, 43, 45]]
    # warmness = [f'Q{i}' for i in [47, 49, 51]]
    # usability = [f'Q{i}' for i in [53, 55, 57]]

    # df['competence'] = df.loc[:, competence].mean(axis=1)
    # df['warmness'] = df.loc[:, warmness].mean(axis=1)
    # df['usability'] = df.loc[:, usability].mean(axis=1)

    return df


def grouping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        must have a column named 'group'.

    Returns
    -------
    df : pd.DataFrame
        dataframe would have new two columns representing the depth of grouping.
    """
    
    df['ABCD_EFGH'], df['ABEF_CDGH'], df['AB_CD_EF_GH'] = '', '', ''
    
    for i, group in enumerate(df['group']):
        
        if group in ['groupa', 'groupb', 'groupc', 'groupd']:
            df.loc[i, 'ABCD_EFGH'] = 'groupABCD'
        else:
            df.loc[i, 'ABCD_EFGH'] = 'groupEFGH'
        
        if group in ['groupa', 'groupb', 'groupe', 'groupf']:
            df.loc[i, 'ABEF_CDGH'] = 'groupABEF'
        else:
            df.loc[i, 'ABEF_CDGH'] = 'groupCDGH'
        
        if group in ['groupa', 'groupb']:
            df.loc[i, 'AB_CD_EF_GH'] = 'groupAB'
        elif group in ['groupc', 'groupd']:
            df.loc[i, 'AB_CD_EF_GH'] = 'groupCD'   
        elif group in ['groupe', 'groupf']:
            df.loc[i, 'AB_CD_EF_GH'] = 'groupEF'
        else:
            df.loc[i, 'AB_CD_EF_GH'] = 'groupGH'
            
    return df


def average_scale_scores(df_target: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df_target : pd.DataFrame
        dataframe that has columns consists of scales.

    Returns
    -------
    df : pd.DataFrame
        dataframe has new columns containing averaged values scales.
    """
    df = df_target.copy()
        
    # average panas's positive and negative scales, respectively
    ps = [5, 6, 7, 9, 11, 12, 13, 14]
    ns = [1, 2, 3, 4, 8, 10, 15, 16]    
    panas_pos = [f'Q11_{p}' for p in ps]
    panas_neg = [f'Q11_{n}' for n in ns]
    
    competence = [f'Q{i}' for i in [4, 43, 45]]
    warmness = [f'Q{i}' for i in [47, 49, 51]]
    usability = [f'Q{i}' for i in [53, 55, 57]]
    
    df['panas_pos'] = df[panas_pos].mean(axis=1)
    df['panas_neg'] = df[panas_neg].mean(axis=1)
    df['competence'] = df[competence].mean(axis=1)
    df['warmness'] = df[warmness].mean(axis=1)
    df['usability'] = df[usability].mean(axis=1)
    
    df = df.sort_values(by='group').reset_index().drop('index', axis=1)    
    
    return df


def make_df_of_diff(df: pd.DataFrame, params: list[str]) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        dataframe that has a column named 'group'.
    params : list[str]
        list of parameters to compute the difference.

    Returns
    -------
    df_diff : pd.DataFrame
        dataframe that has columns of group and differences of parameters.
    """
    df_sorted = df.sort_values('exp_count')
    
    first_df = df_sorted.query('exp_count == 1')
    first_df.sort_values('user_id', inplace=True, ignore_index=True)
    second_df = df_sorted.query('exp_count == 2')
    second_df.sort_values('user_id', inplace=True, ignore_index=True)
    
    groups = []
    for i, group in zip(range(0, len(df), 2), first_df['group']):
        groups.append(group)
    df_diff = pd.DataFrame({'group': groups})
    
    for param in params:
        diffs = second_df[param].sub(first_df[param])
        df_diff[f'{param}_diff'] = diffs
    
    df_diff = grouping(df_diff)
    df_diff.sort_values('group', inplace=True, ignore_index=True)
    
    return df_diff


def factor_analyze(n, df):
    fa = FactorAnalyzer(n_factors=n, rotation=None)
    fa.fit(df)
    loadings_df = pd.DataFrame(fa.loadings_)
    
    return loadings_df
