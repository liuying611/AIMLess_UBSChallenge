import pandas as pd
import numpy as np

def clean_data(file_path = '../data/skylab_instagram_datathon_dataset.csv'):
    df = pd.read_csv(file_path, header=0, sep=";")
    df['period_end_date'] = pd.to_datetime(df['period_end_date']) # turn to datetime
    df.drop(columns=['period', 'calculation_type'], inplace=True) # redundant
    df_allbrands = df[df['business_entity_doing_business_as_name'] == 'All Brands'].copy() # put the summary data in a different df
    df_allbrands.drop(columns=['business_entity_doing_business_as_name', 'legal_entity_name', 'domicile_country_name',
                               'ultimate_parent_legal_entity_name', 'primary_exchange_name'], inplace=True) # redundant
    df.drop(df[df['business_entity_doing_business_as_name'] == 'All Brands'].index, inplace=True) # remove summary data from main df

    brands = df['business_entity_doing_business_as_name'].unique() # list of the brands
    compset_groups = df['compset_group'].unique() # list of the compset_groups
    compsets = df['compset'].unique() # list of the compsets

    groups_bycompset = {} # dictionary that shows which compset is in which compset group(s)
    for compset in compsets:
        df_compset = df[df['compset'] == compset]
        groups_bycompset[compset] = df_compset['compset_group'].unique()

    # df_brands excludes the brand information in a different df
    df_brands = df[['business_entity_doing_business_as_name', 'compset_group', 'compset', 'legal_entity_name',
                    'domicile_country_name', 'ultimate_parent_legal_entity_name', 'primary_exchange_name']].copy()
    df_brands = df_brands.groupby('business_entity_doing_business_as_name').agg({
        'compset_group': 'first',
        'compset': lambda x: sorted(list(set(x))), # collapse all compsets in one row with a list of compsets
        'legal_entity_name': 'first',
        'domicile_country_name': 'first',
        'ultimate_parent_legal_entity_name': 'first',
        'primary_exchange_name': 'first'
    }).reset_index()
    df.drop(columns=['compset_group', 'compset', 'legal_entity_name', 'domicile_country_name',
                     'ultimate_parent_legal_entity_name', 'primary_exchange_name'], inplace=True) # remove the brand info from the main df
    df.fillna(-1, inplace=True) # replace NAs with -1 (to enable drop_duplicates)
    df.drop_duplicates(inplace=True) # Duplicate lines because of compset
    df.replace(-1, float('nan'), inplace=True) # turn back to NaN
    df.sort_values(by=['business_entity_doing_business_as_name', 'period_end_date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df, df_brands, df_allbrands, brands, compsets, compset_groups, groups_bycompset


def missing(df):
    df_missing = df.drop(columns=['period_end_date'])
    df_missing = df_missing.groupby('business_entity_doing_business_as_name').agg(lambda x: np.isnan(x).sum())
    df_missing.reset_index(inplace=True)
    columns = df_missing.columns.tolist()
    columns[1:] = [column + '_NaNs' for column in columns[1:]]  # Add '_NaNs' to each column name except the first
    df_missing.columns = columns

    return df_missing

def derivatives_data(df):

    """Data Frame df_rate_of_change containing the 1st time derivative in parameters pictures, videos, comments, likes and followers"""
    #create copy for dataframe consisting of 1st derivative
    df_rate_of_change = df.copy()
    parameters = ['followers', 'pictures', 'videos', 'comments', 'likes']
    #I AM NOT YET GOING TO FILL NA'S WITH ZEROS BECAUSE THERE COULD BE OTHERS WHICH WE DONT WANT TO FILL
    #add the first time derivative to each column
    df_rate_of_change['change in followers'] = df_rate_of_change['followers'].diff()
    df_rate_of_change.rename(columns={'pictures': 'change in pictures'}, inplace=True)
    df_rate_of_change.rename(columns={'videos': 'change in videos'}, inplace=True)
    df_rate_of_change.rename(columns={'comments': 'change in comments'}, inplace=True)
    df_rate_of_change.rename(columns={'likes': 'change in likes'}, inplace=True)
    df_rate_of_change.drop(columns=['followers'], inplace=True)

    """Data Frame df_curvature containing the 2nd time derivative in parameters pictures, videos, comments, likes and followers"""
    new_parameters = ['change in followers', 'change in pictures', 'change in videos', 'change in comments', 'change in likes']
    #create copy for dataframe consisting of 2nd derivative
    df_curvature = df_rate_of_change.copy()

    #I AM NOT YET GOING TO FILL NA'S WITH ZEROS BECAUSE THERE COULD BE OTHERS WHICH WE DONT WANT TO FILL
    #add the 2nd time derivative in the parameters to each column
    for i, parameter in enumerate(new_parameters):
        df_curvature[parameter] = df_rate_of_change[parameter].diff()
        df_curvature.rename(columns={ parameter : f'curvature in {parameters[i]}'}, inplace=True)
    
    return df_rate_of_change, df_curvature