import eia
import pandas as pd
import matplotlib.pyplot as plt
import calendar
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

def retrieve_time_series(api, series_ID):
    """
    Return the time series dataframe, based on API and unique Series ID
    api: API that we're connected to
    series_ID: string. Name of the series that we want to pull from the EIA API
    """
    #Retrieve Data By Series ID 
    series_search = api.data_by_series(series=series_ID)
    ##Create a pandas dataframe from the retrieved time series
    df = pd.DataFrame(series_search)
    return df
    
def plot_data(df, x_variable, y_variable, title):
    """
    Plot the x- and y- variables against each other, where the variables are columns in
    a pandas dataframe
    df: Pandas dataframe. 
    x_variable: String. Name of x-variable column
    y_variable: String. Name of y-variable column
    title: String. Desired title name
    """
    fig, ax = plt.subplots()
    ax.plot_date(df[x_variable], 
                 df[y_variable], marker='', linestyle='-', label=y_variable)
    fig.autofmt_xdate()
    plt.title(title)
    plt.show()
    
def generate_histogram_of_aggregated_counts(df, 
                                            peak_demand_hour_column, 
                                            group_by_column):
    """
    Generate a histogram of peak demand hour counts, grouped by a column
    Arguments:
        df: Pandas dataframe
        peak_demand_hour_column: String. Name of the column for peak demand hour
        group_by_column: String. Name of column to group by 
    """
    #Create a histogram of counts by hour, grouped by month
    fig = plt.figure(figsize = (20,15))
    ax = fig.gca()
    axarr = df[peak_demand_hour_column].hist(by=df[group_by_column], bins=24, ax=ax)
    for ax in axarr.flatten():
        ax.set_xlabel("Peak Demand Hour (0-23)")
        ax.set_ylabel("Number of Occurrences")
    #Count number of peak hour occurrences, grouped by month
    peak_hour_counts=pd.DataFrame(df.groupby([peak_demand_hour_column, 
                                              group_by_column]).size()).reset_index().rename(columns={0:'Counts'})
    #Pull peak hour for each month and write back as a column
    peak_hour_counts['Number_Occurrences']=peak_hour_counts.groupby([group_by_column])['Counts'].transform('max')
    #Subset the dataframe to only include max counts for peak demand hours for each month
    peak_hour_counts=peak_hour_counts[peak_hour_counts['Counts']==peak_hour_counts['Number_Occurrences']]
    #Order the dataframe by group_by_column
    peak_hour_counts=peak_hour_counts.sort_values(by=[group_by_column])
    #Print the subsetted dataframe
    print(peak_hour_counts[[group_by_column, peak_demand_hour_column, 'Number_Occurrences']])
    
def grid_search_rf(parameter_grid, train_features, train_labels):
    """
    Perform Grid Search on the random forest classifier model, in order to optimize model parameters
    parameter_grid: grid parameters to test against to determine optimal parameters
    train_features: Numpy array, containing training set features
    train_labels: Numpy array, containing training set labels
    """
    # Create a random forest classifier model
    rf = RandomForestClassifier()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, 
                               param_grid = parameter_grid, 
                               cv = 3, 
                               n_jobs = -1, 
                               verbose = 2)
    grid_search.fit(train_features, train_labels)
    print(grid_search.best_params_)

def main():
    """
    Run main script
    """
    #Create EIA API using your specific API key
    api_key = 'YOUR API HERE'
    api = eia.API(api_key)
    
    #Pull the electricity price data
    series_ID='EBA.TEX-ALL.D.H'
    electricity_demand_df=retrieve_time_series(api, series_ID)
    electricity_demand_df.reset_index(level=0, inplace=True)
    #Rename the columns for easer analysis
    electricity_demand_df.rename(columns={'index':'Date_Time',
            electricity_demand_df.columns[1]:'Electricity_Demand_MWh'}, 
            inplace=True)
    #Format the 'Date' column 
    electricity_demand_df['Date_Time']=electricity_demand_df['Date_Time'].astype(str).str[:-4]
    #Remove the 'T' from the Date column
    electricity_demand_df['Date_Time'] = electricity_demand_df['Date_Time'].str.replace('T' , ' ')
    #Convert the Date column into a date object
    electricity_demand_df['Date_Time']=pd.to_datetime(electricity_demand_df['Date_Time'], format='%Y %m%d %H')
    #Plot the data on a yearly basis, using 2019 as an example year
    plot_data(df=electricity_demand_df[(electricity_demand_df['Date_Time']>=pd.to_datetime('2019-01-01')) &
                                    (electricity_demand_df['Date_Time']<pd.to_datetime('2020-01-01'))], 
                                    x_variable='Date_Time', 
                                    y_variable='Electricity_Demand_MWh', 
                                    title='TX Electricity Demand: 2019')
    #Plot the data on a monthly basis, using December 2017 as an example
    plot_data(df=electricity_demand_df[(electricity_demand_df['Date_Time']>=pd.to_datetime('2017-12-01')) &
                                    (electricity_demand_df['Date_Time']<pd.to_datetime('2018-01-01'))], 
                                    x_variable='Date_Time', 
                                    y_variable='Electricity_Demand_MWh', 
                                    title='TX Electricity Demand: December 2017')
    #Plot the data on a weekly basis, using July 1-7, 2019 as an example
    plot_data(df=electricity_demand_df[(electricity_demand_df['Date_Time']>=pd.to_datetime('2019-07-01')) &
                                    (electricity_demand_df['Date_Time']<pd.to_datetime('2019-07-07'))], 
                                    x_variable='Date_Time', 
                                    y_variable='Electricity_Demand_MWh', 
                                    title='TX Electricity Demand: December 2017')
    #Pull the hour into and individual column
    electricity_demand_df['Hour']=electricity_demand_df['Date_Time'].dt.hour
    #Pull the day of month for each reading
    electricity_demand_df['Day_Of_Month']=electricity_demand_df['Date_Time'].dt.day
    #Pull day of week for each reading
    electricity_demand_df['Day_Of_Week']=electricity_demand_df['Date_Time'].dt.day_name()
    #Pull the numeric value for day of the week
    electricity_demand_df['Day_Of_Week_Numeric']=electricity_demand_df['Date_Time'].dt.dayofweek+1
    #Pull the date in terms of week
    electricity_demand_df['Week']=electricity_demand_df['Date_Time'].dt.week
    #Pull the month of the year
    electricity_demand_df['Month']=electricity_demand_df['Date_Time'].dt.month.apply(lambda x: calendar.month_abbr[x])
    #Pull the numeric value for month
    electricity_demand_df['Month_Numeric']=electricity_demand_df['Date_Time'].dt.month
    #Pull th year
    electricity_demand_df['Year']=electricity_demand_df['Date_Time'].dt.year
    
    #Calculate the hour with max demand for each date in the data set
    electricity_demand_df['Peak_Demand_Hour_MWh_For_Day']=electricity_demand_df.groupby(['Day_Of_Month', 'Month', 'Year'], 
                                                 sort=False)['Electricity_Demand_MWh'].transform('max')
    
    #Create time series with just peak hourly data
    peak_demand_hour_df=electricity_demand_df[electricity_demand_df['Electricity_Demand_MWh']==electricity_demand_df['Peak_Demand_Hour_MWh_For_Day']]
    #Rename the 'Hour' column to 'Peak_Demand_Hour'
    peak_demand_hour_df=peak_demand_hour_df.rename(columns={'Hour': 'Peak_Demand_Hour'})
    #Create a histogram of counts by hour
    ax=peak_demand_hour_df['Peak_Demand_Hour'].value_counts().plot(kind='bar', title='Peak Demand Hour by Number of Occurrences')
    ax.set_xlabel("Demand Hour (0-23 hour)")
    ax.set_ylabel("Number of Occurrences")
    
    #Create a histogram of counts by peak demand hour, grouped by day of the week
    generate_histogram_of_aggregated_counts(peak_demand_hour_df, 
                                            peak_demand_hour_column='Peak_Demand_Hour', 
                                            group_by_column='Day_Of_Week_Numeric')
    #Create a histogram of counts by peak demand hour, grouped by month
    generate_histogram_of_aggregated_counts(peak_demand_hour_df, 
                                            peak_demand_hour_column='Peak_Demand_Hour', 
                                            group_by_column='Month_Numeric')
    #Subset the dataframe to only include the features and labels that we're going to use 
    #in the random forest model
    peak_demand_hour_model=peak_demand_hour_df[['Peak_Demand_Hour',
                                                      'Day_Of_Week', 
                                                      'Week',
                                                      'Month',
                                                      'Year']]
    #Convert the Week, Year, and Peak_Demand_Your variables into categoric string variables (from numeric)
    peak_demand_hour_model.loc[:,'Week']=peak_demand_hour_model['Week'].apply(str)
    peak_demand_hour_model.loc[:,'Year']=peak_demand_hour_model['Year'].apply(str)
    peak_demand_hour_model.loc[:,'Peak_Demand_Hour']='Hour '+peak_demand_hour_model['Peak_Demand_Hour'].apply(str)
    #Pull the counts per peak demand hour category
    counts_by_category=pd.DataFrame(peak_demand_hour_model.groupby('Peak_Demand_Hour')['Peak_Demand_Hour'].count())
    #Isolate peak hour occurrences that occur more than 15 times
    more_than_15_occurrences=counts_by_category[counts_by_category['Peak_Demand_Hour']>15]
    #Filter the data set to only include instances with more than 15 occurrences--this is just to remove
    #any super anomalous cases from the model
    peak_demand_hour_model=peak_demand_hour_model[peak_demand_hour_model['Peak_Demand_Hour'].isin(list(more_than_15_occurrences.index))]
    #Remove the labels from the features
    features= peak_demand_hour_model.drop('Peak_Demand_Hour', axis = 1)
    #One hot encode the categorical features
    features = pd.get_dummies(features)
    #Create labels 
    labels = np.array(peak_demand_hour_model['Peak_Demand_Hour'])
    #Saving feature names for later use
    feature_list = list(features.columns)
    # Convert to numpy array
    features = np.array(features)
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, 
                                                                                labels, 
                                                                                test_size = 0.25, 
                                                                                random_state = 5)  
    #Create the parameter grid, which is plugged into
    #GridSearchCV, where all hyperparamter combos are tested to find the optimal parameters combination
    parameter_grid={'max_depth': [80, 90, 100, 110],
                    'n_estimators': [700, 800, 900, 1000, 1100, 1200]}   
    grid_search_rf(parameter_grid, train_features, train_labels)
    """
    Grid Search Outputs:
        Fitting 3 folds for each of 20 candidates, totalling 60 fits
        [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
        [Parallel(n_jobs=-1)]: Done  33 tasks      | elapsed:  2.1min
        [Parallel(n_jobs=-1)]: Done  60 out of  60 | elapsed:  3.7min finished
        {'max_depth': 80, 'n_estimators': 1000}
    """
    #Plug in optimized model parameters into final RF model 
    rf = RandomForestClassifier(n_estimators=1000, 
                                max_depth=80,
                                random_state = 1000)
    #Fit the model 
    rf.fit(train_features, train_labels)
    # Use the forest's predict method on the test data
    print(confusion_matrix(test_labels, 
                           rf.predict(test_features),
                           labels=['Hour 0', 'Hour 1',
                                   'Hour 2', 'Hour 14',
                                   'Hour 15', 'Hour 21', 'Hour 22', 'Hour 23']))
    accuracy_score(test_labels, rf.predict(test_features), normalize=True, sample_weight=None)
    #Obtain feature importances in the model
    feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = feature_list,
                                   columns=['importance']).sort_values('importance', ascending=False)
    print(feature_importances)

#Run the main script
if __name__== "__main__":
    main()
