import pandas as pd

def clean_data(file_path = '../data/skylab_instagram_datathon_dataset.csv'):
    df = pd.read_csv(file_path, header=0, sep=";")
    df.drop(columns=['period', 'calculation_type'], inplace=True)
    df_allbrands = df[df['business_entity_doing_business_as_name'] == 'All Brands'].copy()
    df_allbrands.drop(columns=['business_entity_doing_business_as_name', 'legal_entity_name', 'domicile_country_name',
                               'ultimate_parent_legal_entity_name', 'primary_exchange_name'], inplace=True)
    df.drop(df[df['business_entity_doing_business_as_name'] == 'All Brands'].index, inplace=True)

    brands = df['business_entity_doing_business_as_name'].unique()
    compset_groups = df['compset_group'].unique()
    compsets = df['compset'].unique()

    groups_bycompset = {}
    for compset in compsets:
        df_compset = df[df['compset'] == compset]
        groups_bycompset[compset] = df_compset['compset_group'].unique()

    df_brands = df[['business_entity_doing_business_as_name', 'compset_group', 'compset', 'legal_entity_name',
                    'domicile_country_name', 'ultimate_parent_legal_entity_name', 'primary_exchange_name']].copy()
    df_brands = df_brands.groupby('business_entity_doing_business_as_name').agg({
        'compset_group': 'first',
        'compset': lambda x: sorted(list(set(x))),
        'legal_entity_name': 'first',
        'domicile_country_name': 'first',
        'ultimate_parent_legal_entity_name': 'first',
        'primary_exchange_name': 'first'
    }).reset_index()
    df.drop(columns=['compset_group', 'compset', 'legal_entity_name', 'domicile_country_name',
                     'ultimate_parent_legal_entity_name', 'primary_exchange_name'], inplace=True)
    df.fillna(-1, inplace=True)
    df.drop_duplicates(inplace=True)
    df.sort_values(by=['business_entity_doing_business_as_name', 'period_end_date'], inplace=True)
    
    return df, df_brands, df_allbrands, brands, compsets, compset_groups, groups_bycompset