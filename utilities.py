import pandas as pd
import numpy as np


##############################################################################################


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


##############################################################################################


def missing_df(df):
    entries_per_business = df['business_entity_doing_business_as_name'].value_counts()

    df_missing = df.drop(columns=['period_end_date'])
    df_missing.insert(1, 'Total Entries:', entries_per_business)
    df_missing = df_missing.groupby('business_entity_doing_business_as_name').agg(lambda x: np.isnan(x).sum())
    df_missing.reset_index(inplace=True)
    columns = df_missing.columns.tolist()
    columns[2:] = ['Missing ' + column + ":" for column in columns[2:]]  # Add '_NaNs' to each column name except the first
    df_missing.columns = columns

    df_missing.rename(columns={'business_entity_doing_business_as_name': 'Business'}, inplace=True)

    return df_missing


##############################################################################################


def missing_values(df):
    '''''
    comment
    '''''
    # Count the number of rows with at least one NaN value across any column and print
    rows_with_nan = df.isna().any(axis=1).sum()
    print("Number of rows with at least one NaN before cleaning:", rows_with_nan)

    # Function to remove leading NaN-series in 'followers' for each group(business)
    def remove_leading_nans(group):
        # Get the first index where 'followers' is not NaN
        first_valid_index = group['followers'].first_valid_index()
        # If all are NaNs, return an empty DataFrame from this group
        if first_valid_index is None:
            return pd.DataFrame()
        # Return the DataFrame starting from the first non-NaN 'followers' row
        return group.loc[first_valid_index:]

    # Apply the function to each group and concatenate the results
    cleaned_df = df.groupby('business_entity_doing_business_as_name', group_keys=False).apply(remove_leading_nans)

    # Count the number of rows with at least one NaN value across any column
    rows_with_nan = cleaned_df.isna().any(axis=1).sum()
    print("Number of rows with at least one NaN after dropping series of Nan's at beginning of businesses:", rows_with_nan)

    cleaned_df = cleaned_df.reset_index(drop=True)

    #How long are the remaining series of Nan's in each column?
    def calculate_nan_series_lengths_indices_and_total(_df):
        # Initialize a dictionary to store the lengths of NaN series and their indices for each column
        nan_series_details = {col: [] for col in _df.columns if _df[col].isna().any()}
        total_nan_count = 0
        # Iterate through each column that contains NaN
        for col in nan_series_details.keys():
            current_series_length = 0
            series_start_index = None
            previous_business = None
            # Iterate through each row
            for idx, row in _df.iterrows():
                if pd.isna(row[col]):
                    # Check if we've moved to a new business
                    if row['business_entity_doing_business_as_name'] != previous_business:
                        if current_series_length > 0:
                            # Save the length and indices of the previous series before starting a new one
                            nan_series_details[col].append((current_series_length, series_start_index, idx - 1))
                            total_nan_count += current_series_length
                        # Reset the series length for the new business and record start index
                        current_series_length = 1
                        series_start_index = idx
                    else:
                        # Increment the series length
                        current_series_length += 1
                else:
                    if current_series_length > 0:
                        # End of a series, append its length and indices to the list and add to total count
                        nan_series_details[col].append((current_series_length, series_start_index, idx - 1))
                        total_nan_count += current_series_length
                        current_series_length = 0
                        series_start_index = None
                
                # Update the previous business name
                previous_business = row['business_entity_doing_business_as_name']
            
            # Check if the last row in the dataframe was a NaN and needs to be added
            if current_series_length > 0:
                nan_series_details[col].append((current_series_length, series_start_index, _df.index[-1]))
                total_nan_count += current_series_length

        return nan_series_details, total_nan_count

    nan_series_info, total_nan = calculate_nan_series_lengths_indices_and_total(cleaned_df)
    #print(nan_series_info)
    #print("Total NaN entries:", total_nan)

    #manually drop 4 remaining series at end of groups
    indices_to_drop = list(range(20769, 20842)) + list(range(30040, 30082)) + list(range(34058, 34146)) + list(range(71279, 71309))
    cleaned_df = cleaned_df.drop(indices_to_drop)

    #How many remaining number of rows with Nan's are there?
    # Calculate the number of NaNs per row
    nan_counts_per_row = cleaned_df.isna().sum(axis=1)

    # Count how many rows have exactly 4, 3, and 2 NaNs
    nan_4 = (nan_counts_per_row == 4).sum()
    nan_3 = (nan_counts_per_row == 3).sum()
    nan_2 = (nan_counts_per_row == 2).sum()
    nan_1 = (nan_counts_per_row == 1).sum()

    print("\n Remaining number of rows with Nan that are not at beginning or end:")
    print("Number of rows with 4 NaNs:", nan_4)
    print("Number of rows with 3 NaNs:", nan_3)
    print("Number of rows with 2 NaNs:", nan_2)
    print("Number of rows with 1 NaNs:", nan_1)
    print("Total remaining rows with at least one Nan: ", nan_1+nan_2+nan_3+nan_4)
    #print("Total entries being an Nan: ",nan_1+2*nan_2+3*nan_3+4*nan_4)

    print("\n Total number of rows after cleaning:", len(cleaned_df))

    return cleaned_df


##############################################################################################


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

def normalized_data_frame(df, df_allbrands = clean_data()[2]):
    #create a new dataframe to make space for the normalized data
    df_normalized = df.copy()
    # Convert the 'period_end_date' column to datetime objects
    df_normalized['period_end_date'] = pd.to_datetime(df_normalized['period_end_date'])

    #Normalization
    #e.g. divide the number of followers for a company by the sum of all followers of all companies
    for date in df_normalized['period_end_date'].unique():
        # Get the indices where the 'period_end_date' matches the current date
        indices = (df_normalized['period_end_date'] == date)
        
        # Normalize each column in df_normalized by the corresponding value in df_allbrands_total
        df_normalized.loc[indices, 'followers'] /= df_allbrands_total.loc[df_allbrands_total['period_end_date'] == date, 'followers'].values[0]
        df_normalized.loc[indices, 'pictures']  /= df_allbrands_total.loc[df_allbrands_total['period_end_date'] == date, 'pictures'].values[0]
        df_normalized.loc[indices, 'videos']    /= df_allbrands_total.loc[df_allbrands_total['period_end_date'] == date, 'videos'].values[0]
        df_normalized.loc[indices, 'comments']  /= df_allbrands_total.loc[df_allbrands_total['period_end_date'] == date, 'comments'].values[0]
        df_normalized.loc[indices, 'likes']     /= df_allbrands_total.loc[df_allbrands_total['period_end_date'] == date, 'likes'].values[0]
        
    return df_normalized


##############################################################################################
