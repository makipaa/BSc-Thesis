from re import split
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
import time
import logging
import os
import datetime

DATA_PATH = 'data/processed/data.pkl'

FEATURES = ['daily_peak_occupancy', 'hospital_beds_available', 'weekday', 'month:', 'holiday_t', 
            'holiday_name', 'is_working_day', 'weather', 'website_visits',
            'google_trends', 'public_events_num', 'ekströms']
PARAM_GRID = {
    'changepoint_prior_scale' : [0.001, 0.05, 0.5],
    'seasonality_prior_scale' : [0.01, 1, 10],
    'holidays_prior_scale' : [0.01, 1, 10],
    'seasonality_mode' : ['additive', 'multiplicative']
}


class suppress_stdout_stderr(object):
    '''
    This class is used for suppressing the output produced from Prophet Training (STAN).
    Adapted from https://github.com/facebook/prophet/issues/223
    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def preprocess(data):
    """
    This functions performs several general preprocessing steps
    that are required for all models that we test. These steps are:
        
        - filter certain columns that we have decided to exclude
        - resample to daily resolution with meaningful methods
        - name columns correctly
        - guarantee there will not be any nones in the resulting frame

    After general preprocessing, model-wise preprocessing might be 
    required.
    """

    # Targets
    total_daily_arrivals = data.arrivals.resample("D").sum()
    total_daily_arrivals.name = "total_daily_arrivals"
    daily_peak_occupancy = data.occupancy.resample("D").max()
    daily_peak_occupancy.name = "daily_peak_occupancy"

    targets = pd.concat([total_daily_arrivals,
                        daily_peak_occupancy], axis=1)
    targets = targets.fillna(0)

    # Output capacity
    beds = data.iloc[:,data.columns.str.startswith("hospital")]
    beds = beds.resample("D").last()
    beds = beds.fillna(0)

    # Calendar variables are naturally contained in the index
    weekdays = pd.get_dummies(data.index.day_name(), prefix="weekday", prefix_sep=":")
    weekdays.index = data.index
    months = pd.get_dummies(data.index.month_name(), prefix="month", prefix_sep=":")
    months.index = data.index
    holidays = data[["holiday_t-1", "holiday_t+0", "holiday_t+1"]].fillna(0).astype(int)
    holiday_name = pd.get_dummies(data.holiday_name, prefix="holiday_name", prefix_sep=":")
    is_working_day = data.is_working_day.astype(int)

    cal_vars = pd.concat([weekdays, months, holidays, holiday_name, is_working_day], axis=1)
    cal_vars = cal_vars.resample("D").last()

    # Weather variables
    weather_cols = ["cloud_count", 
                    "air_pressure", 
                    "rel_hum", 
                    "rain_intensity", 
                    "snow_depth", 
                    "air_temp", 
                    "dew_point_temp", 
                    "visibility",
                    "slip",
                    "heat"]
    weather = data.loc[:,weather_cols]
    weather["air_temp_min"] = weather.air_temp.resample("D").min()
    weather["air_temp_max"] = weather.air_temp.resample("D").max()
    weather = weather.resample("D").mean()
    weather.columns = "weather:" + weather.columns

    # Internet activity
    internet_cols = ["website_visits_tays",
                    "website_visits_acuta",
                    "ekströms_visits",
                    "google_trends_acuta_monthnorm"]

    web_visits = data.loc[:,internet_cols]
    web_visits = web_visits.resample("D").sum()

    # Public events
    event_cols = data.columns.str.startswith("public")
    events = data.loc[:,event_cols]
    events = events.resample("D").first()
    events = events.fillna(0)
    events = events.astype(int)

    # Shift necessary features
    web_visits = web_visits.shift(-1, fill_value=0)
    beds = beds.shift(-1, fill_value=0)

    # Collect features
    df = pd.concat([targets, beds, cal_vars, weather, web_visits, events], axis=1)

    # Fill nulls
    df = df.fillna(0)
    
    return df

def get_variables(df, variable_left_out="None"):
    """ Select desired variables which are given to the Prophet model as "holidays"
    :param df: The entire dataset as a dataframe
    :param variable_left_out: This is used for variable selection function
    :return: Dataframe of the variables
    """
    variables = pd.DataFrame()
    i = 0
    
    for (column_name, _) in df.iteritems():

        # Skip timestamps, total_daily_arrivals and public_events
        if i > 1 and (not ('public_event:' in column_name) and not (variable_left_out in column_name)):
            print(column_name)  # Visualize variables included
            occurs = df.loc[df[column_name] != 0]
            dates = np.array(occurs['timestamp'].astype(str))
            variable = pd.DataFrame({
                'holiday' : column_name,
                'ds' : pd.to_datetime(dates)
            })
            if i == 2:
                variables = variable
            else:
                variables = pd.concat((variables, variable))
        i += 1
    return variables
            
def compute_mape(y, y_hat):
    """ Computes mean absolute percentage error (MAPE)
    :param y: Ground truth
    :paramy y_hat: Predicted value
    :return: Mape value
    """ 
    y, y_hat = np.array(y), np.array(y_hat)
    return np.mean(np.abs((y - y_hat) / y)) * 100

def variable_selection(df, best_params):
    """ This function is used to determine which variable left out has the biggest impact on the error 
        (This function may be left out if there are no additional variables).
    :param df: The dataset
    :param best_params: Best parameters for the model yielded from hyperparameter tuning
    :return variable_sel_results: Returns a result dataframe
    """
    variable_sel_results = pd.DataFrame()

    # Change column names for Prophet
    temp_df = df[['timestamp', 'total_daily_arrivals']]
    temp_df.columns = ['ds', 'y']

    for i, feature_leftout in enumerate(FEATURES):
        variables = get_variables(df, feature_leftout)
        start1 = time.time()
        print('-----------------------\nRound {} without feature: {}'.format(i, feature_leftout))

        with suppress_stdout_stderr():
            model = Prophet(**best_params, uncertainty_samples=0, daily_seasonality=False, yearly_seasonality=True, holidays=variables).fit(temp_df)
            df_crossv = cross_validation(model, initial='1100 days', horizon='1 days', period='1 days', parallel='processes')

        pred = df_crossv['yhat']
        ground_truth = temp_df['y'][1101:]

        mape = compute_mape(ground_truth, pred)
        rmse = np.sqrt(mean_squared_error(ground_truth, pred))
        mae = mean_absolute_error(ground_truth, pred)

        new_row = {'Variable_left_out':feature_leftout,
                'MAPE':mape, 'RMSE':rmse, 'MAE':mae}
        variable_sel_results = variable_sel_results.append(new_row, ignore_index=True)
        end1 = time.time()
        print('Training and validation took {} minutes. Feature leftout: {}. Yielded MAPE: {}'.format((end1-start1)/60, feature_leftout, mape))
    
    for index, row in variable_sel_results.iterrows():
        print(row['Variable_left_out'], row['MAPE'], row['RMSE'], row['MAE'])
    return variable_sel_results

def hyperparameter_tuning(df, variables):
    """ This function is used to determine which hyperparameters suite the model the best.
    :param df: The dataset
    :param variables: Additional variables for the model
    :return best_params: Best hyperparameters for the model
    """
    # Make all combinations of parameters
    all_parameters = [dict(zip(PARAM_GRID.keys(), v)) for v in itertools.product(*PARAM_GRID.values())]
    params_result_matrix = pd.DataFrame()
    start = time.time()
    for i, param in enumerate(all_parameters):
        
        start1 = time.time()
        # Suppress Prophet printing
        with suppress_stdout_stderr():
            model = Prophet(**param, uncertainty_samples=0, daily_seasonality=False, yearly_seasonality=True, holidays=variables).fit(df)
            df_crossv = cross_validation(model, initial='1100 days', horizon='1 days', period='1 days', parallel='processes')

        pred = df_crossv['yhat']
        ground_truth = df['y'][1101:]

        mape = compute_mape(ground_truth, pred)
        rmse = np.sqrt(mean_squared_error(ground_truth, pred))
        mae = mean_absolute_error(ground_truth, pred)

        new_row = {'changepoint_prior_scale':param['changepoint_prior_scale'], 
                'seasonality_prior_scale':param['seasonality_prior_scale'],
                'holidays_prior_scale':param['holidays_prior_scale'],
                'seasonality_mode':param['seasonality_mode'],
                'MAPE':mape, 'RMSE':rmse, 'MAE':mae}
        params_result_matrix = params_result_matrix.append(new_row, ignore_index=True)
        end1 = time.time()
        print('{}. epoch took {} minutes with parameters {}. Yielded MAPE: {}'.format(i, (end1-start1)/60, param, mape))
    
    end = time.time()
    print("------------------------------\nTraining and cross validation took {} minutes".format((end-start)/60))
    params_result_matrix = params_result_matrix.sort_values(by=['MAPE'])
    print(params_result_matrix.head(10))
    best_params = params_result_matrix.loc[0, ['changepoint_prior_scale', 'holidays_prior_scale', 'seasonality_prior_scale', 'seasonality_mode']]
    return best_params

def iterate_prophet(data, split_date, variables, best_params):
    """ This function is completely optional, and it is only used for plotting the model components, because Prophet
        does not allow plotting model components from cross validation.
    :param data: The dataset
    :param split_date: The date which splits the training and test set
    :param variables: Additional variables for the model
    :param best_params Best hyperparameters for the model
    :return model, forecast: The model and the forecast for plotting model components
    """
    
    while split_date != data.ds.values[-1]:
        train = data[:split_date]
        
        with suppress_stdout_stderr():
            model = Prophet(**best_params, uncertainty_samples=0, daily_seasonality=False, yearly_seasonality=True, holidays=variables).fit(train)
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)
        split_date = split_date + datetime.timedelta(days = 1)
    return model, forecast

def main():
    # Ignore Prophet logs
    logging.getLogger('fbprophet').setLevel(logging.WARNING)
    # Load the dataset
    df = pd.read_pickle(DATA_PATH)['2015-05':]
    df = preprocess(df)
    
    # Add additional timestamp column for Prophet
    df.insert(0, 'timestamp', df.index)
    variables = get_variables(df)

    temp_df = df
    # Change column names for Prophet
    df = df[['timestamp', 'total_daily_arrivals']]
    df.columns = ['ds', 'y']

    best_parameters = hyperparameter_tuning(df, variables)
    #variable_sel_results = variable_selection(temp_df, best_parameters)

    split_date = pd.to_datetime('2018-05-06')
    model, forecast = iterate_prophet(df, split_date, variables, best_parameters)

    fig1 = model.plot(forecast)
    plt.show()
    fig2 = model.plot_components(forecast)
    plt.show()

    pred = forecast['yhat'][1101:]
    ground_truth = df['y'][1101:]
    
    # Evaluation metrics
    mape = compute_mape(ground_truth, pred)
    print('MAPE: {}'.format(mape))
    rmse = np.sqrt(mean_squared_error(ground_truth, pred))
    print('RMSE: {}'.format(rmse))
    mae = mean_absolute_error(ground_truth, pred)
    print('MAE: {}'.format(mae))
    
if __name__ == '__main__':
    main()