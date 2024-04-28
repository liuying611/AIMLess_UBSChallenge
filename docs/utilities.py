import pandas as pd
import numpy as np
import math


##############################################################################################


def clean_data(file_path='../../data/skylab_instagram_datathon_dataset.csv'):
    """
    This function clean_data performs several operations to clean and preprocess a DataFrame read from a CSV file. Here's a breakdown of each step:

    Read Data: Reads the CSV file into a DataFrame using pandas read_csv() function.
    Convert to Datetime: Converts the 'period_end_date' column to datetime format using pd.to_datetime().
    Drop Redundant Columns: Drops the columns 'period' and 'calculation_type' from the DataFrame as they are considered redundant.
    Separate Summary Data: Copies the rows where the 'business_entity_doing_business_as_name' column is 'All Brands' into a separate DataFrame called df_allbrands. Then removes these rows from the main DataFrame.
    Extract Unique Values: Extracts unique values of 'business_entity_doing_business_as_name', 'compset_group', and 'compset' columns.
    Grouping by Compset: Creates a dictionary groups_bycompset where each key is a unique 'compset' value and the corresponding value is an array of unique 'compset_group' values.
    Aggregate Brand Information: Creates a DataFrame df_brands containing aggregated brand information. It groups the DataFrame by 'business_entity_doing_business_as_name' and aggregates other columns by selecting the first value and creating a sorted list of unique 'compset' values.
    Remove Redundant Columns (Again): Drops brand-related columns from the main DataFrame.
    Sort and Reset Index: Sorts the DataFrame by 'business_entity_doing_business_as_name' and 'period_end_date', then resets the index.
    Finally, the function returns several objects: the cleaned DataFrame (df), DataFrame containing brand information (df_brands), DataFrame containing summary data for all brands (df_allbrands), lists of unique brands, compsets, and compset groups, and a dictionary mapping compsets to compset groups (groups_bycompset).

    :returns:
    df: main dataframe containing only the period_end_date, business_entity_doing_business_as_name and the five numerical quantities without any redundancies. The data is sorted by date and Brand. There are no NANs in period_end_date and business_entity_doing_business_as_name.
    df_brands: Contains the meta-information for each brand (business_entity_doing_business_as_name), namely the compset_group, a list of compsets, legal_entity_name, domicile_country_name, ultimate_parent_legal_entity_name and primary_exchange_name.
    df_allbrands: The rows of df with compset='All Brands'
    brands: List of all brands (business_entity_doing_business_as_name)
    compsets: List of all compsets
    compset_groups: List of all compset_groups
    groups_bycompset: Dictionary, defining in which compset_group(s) each compset occurs.
    """

    df = pd.read_csv(file_path, header=0, sep=";")
    df['period_end_date'] = pd.to_datetime(df['period_end_date'])  # turn to datetime
    df.drop(columns=['period', 'calculation_type'], inplace=True)  # redundant
    df_allbrands = df[
        df['business_entity_doing_business_as_name'] == 'All Brands'].copy()  # put the summary data in a different df
    df_allbrands.drop(columns=['business_entity_doing_business_as_name', 'legal_entity_name', 'domicile_country_name',
                               'ultimate_parent_legal_entity_name', 'primary_exchange_name'], inplace=True)  # redundant
    df.drop(df[df['business_entity_doing_business_as_name'] == 'All Brands'].index,
            inplace=True)  # remove summary data from main df

    brands = df['business_entity_doing_business_as_name'].unique()  # list of the brands
    compset_groups = df['compset_group'].unique()  # list of the compset_groups
    compsets = df['compset'].unique()  # list of the compsets

    groups_bycompset = {}  # dictionary that shows which compset is in which compset group(s)
    for compset in compsets:
        df_compset = df[df['compset'] == compset]
        groups_bycompset[compset] = df_compset['compset_group'].unique()

    # df_brands excludes the brand information in a different df
    df_brands = df[['business_entity_doing_business_as_name', 'compset_group', 'compset', 'legal_entity_name',
                    'domicile_country_name', 'ultimate_parent_legal_entity_name', 'primary_exchange_name']].copy()
    df_brands = df_brands.groupby('business_entity_doing_business_as_name').agg({
        'compset_group': 'first',
        'compset': lambda x: sorted(list(set(x))),  # collapse all compsets in one row with a list of compsets
        'legal_entity_name': 'first',
        'domicile_country_name': 'first',
        'ultimate_parent_legal_entity_name': 'first',
        'primary_exchange_name': 'first'
    }).reset_index()
    df.drop(columns=['compset_group', 'compset', 'legal_entity_name', 'domicile_country_name',
                     'ultimate_parent_legal_entity_name', 'primary_exchange_name'],
            inplace=True)  # remove the brand info from the main df
    df.fillna(-1, inplace=True)  # replace NAs with -1 (to enable drop_duplicates)
    df.drop_duplicates(inplace=True)  # Duplicate lines because of compset
    df.replace(-1, float('nan'), inplace=True)  # turn back to NaN
    df.sort_values(by=['business_entity_doing_business_as_name', 'period_end_date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df, df_brands, df_allbrands, brands, compsets, compset_groups, groups_bycompset


##############################################################################################


def missing_df(df):
    """
    Processes the input DataFrame to count missing values for each business entity, include total data entries per business,
    and format the output for clarity.
    
    The function drops the 'period_end_date' column, inserts a count of total entries for each business, computes the
    sum of NaNs for each remaining column grouped by business, and reformats the column names to indicate missing data.
    It also renames the 'business_entity_doing_business_as_name' to 'Business' for readability.
    
    Parameters:
    df (DataFrame): A pandas DataFrame containing business data including a 'period_end_date' column and potentially missing values.
    
    Returns:
    DataFrame: A transformed DataFrame with each business entity as rows, columns showing the count of NaNs for each remaining
    column, the total entries per business, and reformatted column names.
    """
    # Count total entries for each business
    entries_per_business = df['business_entity_doing_business_as_name'].value_counts()
    # Drop the 'period_end_date' column from the DataFrame
    df_missing = df.drop(columns=['period_end_date'])
    # Insert a new column with total entries per business at position 1
    df_missing.insert(1, 'Total Entries:', entries_per_business)
    # Group data by 'business_entity_doing_business_as_name' and calculate the sum of NaNs for each group
    df_missing = df_missing.groupby('business_entity_doing_business_as_name').agg(lambda x: np.isnan(x).sum())
    # Reset the index to turn group labels into standard columns
    df_missing.reset_index(inplace=True)

    # Modify the column names from the third column onwards to indicate they represent missing data counts
    columns = df_missing.columns.tolist()
    columns[2:] = ['Missing ' + column + ":" for column in columns[2:]]  # Enhance readability by adding 'Missing'
    df_missing.columns = columns

    # Rename the 'business_entity_doing_business_as_name' column to 'Business' for better clarity
    df_missing.rename(columns={'business_entity_doing_business_as_name': 'Business'}, inplace=True)

    # Return the modified DataFrame
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
    """
    Data Frame df_rate_of_change containing the 1st time derivative 
    in parameters pictures, videos, comments, likes, and followers
    """
    # Create a copy of the dataframe for the first derivative
    df_rate_of_change = df.copy()
    parameters = ['followers', 'pictures', 'videos', 'comments', 'likes']
    
    # Add the first time derivative to each column
    df_rate_of_change['change in followers'] = df_rate_of_change['followers'].diff()
    df_rate_of_change.rename(columns={'pictures': 'change in pictures'}, inplace=True)
    df_rate_of_change.rename(columns={'videos': 'change in videos'}, inplace=True)
    df_rate_of_change.rename(columns={'comments': 'change in comments'}, inplace=True)
    df_rate_of_change.rename(columns={'likes': 'change in likes'}, inplace=True)
    df_rate_of_change.drop(columns=['followers'], inplace=True)

    """
    Data Frame df_curvature containing the 2nd time derivative 
    in parameters pictures, videos, comments, likes, and followers
    """
    new_parameters = ['change in followers', 'change in pictures', 'change in videos', 'change in comments', 'change in likes']
    # Create a copy for the dataframe consisting of the 2nd derivative
    df_curvature = df_rate_of_change.copy()
    
    # Add the 2nd time derivative in the parameters to each column
    for i, parameter in enumerate(new_parameters):
        df_curvature[parameter] = df_rate_of_change[parameter].diff()
        df_curvature.rename(columns={ parameter : f'curvature in {parameters[i]}'}, inplace=True)
    
    return df_rate_of_change, df_curvature

def normalization(df, df_allbrands=clean_data()[2]):
    """
    Normalize the data frame by dividing each column by the corresponding 
    sum of all columns using df_allbrands for each date.
    """
    df_allbrands_total = df_allbrands.copy()
    df_allbrands_total = df_allbrands_total[df_allbrands_total['compset'] == 'Study (All Brands)']
    # Create a new dataframe to make space for the normalized data
    df_normalized = df.copy()
    # Convert the 'period_end_date' column to datetime objects
    df_normalized['period_end_date'] = pd.to_datetime(df_normalized['period_end_date'])

    # Normalization
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


def ranking_followers(df, df_brands, compset_groups):
    """
    This function ranking_followers performs the following tasks:

    Create Dictionary of Brands by Compset Group: It creates a dictionary brands_by_cgroup where each key is a compset group, and the corresponding value is a list of brands contained in that compset group. It filters the brand DataFrame df_brands based on each compset group and extracts the list of brands.
    Generate Rankings for Each Compset Group: It creates a dictionary cgroup_ranking where each key is a compset group. For each compset group, it filters the main DataFrame df to include only the rows corresponding to brands in that compset group, drops rows with missing 'followers' values, and then generates rankings ('FRanking') for each brand based on the number of followers. Rankings are generated for each date separately using the rank method, and NaN values are handled.
    Calculate Difference in Rankings: It calculates the difference in rankings ('diff_FRanking') for each brand within each compset group. This is done by taking the difference of consecutive rankings using the diff method.
    Blur Difference in Rankings: It calculates a blurred version of the difference in rankings ('diff_FRanking_blur') by taking the rolling mean with a window size of 5.
    Normalize Blurred Difference in Rankings: It normalizes the blurred difference in rankings ('diff_FRanking_blur_norm') by dividing each value by the number of brands in the corresponding compset group.

    :return:
    cgroup_ranking: Dictionary, containing for each compset_group a dataframe with just the part of df with brands in that compset_group. Additional columns are added, specifying the ranking by followers of each brand in a certain compset_group for every week, the change in this ranking from week to week and blurred and normalized versions of the previous column.
    brands_by_cgroup: Dictionary, containing the brands of each compset_group.

    """

    brands_by_cgroup = {}  # dict: list of brands contained in each compset_group
    for group in compset_groups:
        brands_by_cgroup[group] = df_brands[df_brands['compset_group'] == group][
            'business_entity_doing_business_as_name'].tolist()

    # Dictionary of data-frames. For each compset_group there is a dataframe containing just the part of df with brands in that compset_group
    cgroup_ranking = {cgroup: df[df['business_entity_doing_business_as_name'].isin(brands_by_cgroup[cgroup])].dropna(
        subset=['followers']).copy() for cgroup in compset_groups}

    #Add Column of Follower Ranking amoung all the brands in one sector (compset_group)
    for cgroup in compset_groups:
        cgroup_ranking[cgroup]['FRanking'] = np.nan
        for date, data in cgroup_ranking[cgroup].groupby('period_end_date'):
            cgroup_ranking[cgroup].loc[data.index, 'FRanking'] = data['followers'].rank(ascending=False,method='dense').astype(int)

    #Add Column containing the difference in Follower Ranking from one week to the next
    for cgroup in compset_groups:
        grouped = cgroup_ranking[cgroup].groupby('business_entity_doing_business_as_name')
        cgroup_ranking[cgroup]['diff_FRanking'] = grouped['FRanking'].diff()

    # Add Column containing the rolling mean of the previous column (difference in Follower Ranking). This is done to decrease the importance of single events and increase bigger trends.
    for cgroup in compset_groups:
        cgroup_ranking[cgroup]['diff_FRanking_blur'] = cgroup_ranking[cgroup]['diff_FRanking'].rolling(window=5, center=True).mean()

    # Add Column containing the previous column normalized by the number of brands in the respective compset_group. Since it is more likely to move by many ranks in a sector (compset_group) with more brands, this is meant to make different compset_groups more comparable. This column is not used in the current Data Analysis.
    for cgroup in compset_groups:
        cgroup_ranking[cgroup]['diff_FRanking_blur_norm'] = cgroup_ranking[cgroup].apply(
            lambda row: row['diff_FRanking_blur'] / len(brands_by_cgroup[cgroup]), axis=1)

    return cgroup_ranking, brands_by_cgroup