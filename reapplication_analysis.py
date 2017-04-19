import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.discrete.discrete_model import Logit
from sklearn.metrics import roc_curve
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import train_test_split
import string
import datetime as dt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydot

pd.options.mode.chained_assignment = None  # default='warn'

def load_data(saved=False):
    '''
    INPUT:
        - saved: boolean
    OUTPUT:
        - lead_df: DataFrame
        - email_df: DataFrame
        - hist_df: DataFrame
        - task_df: DataFrame
    If saved=False, calls four functions to load and do initial cleaning and
    feature engineering on the four main csv files with important data:
    Lead.csv, EmailMessage.csv, EntityHistory.csv, and Task.csv.  If
    saved=True, loads pre-cleaned versions of those files for efficiency.
    '''
    if saved:
        lead_df = pd.read_csv('lead_data.csv', parse_dates=[2, 3, 4])
        email_df = pd.read_csv('email_data.csv')
        hist_df = pd.read_csv('hist_data.csv', parse_dates=[2])
        task_df = pd.read_csv('task_data.csv', parse_dates=[2])
    else:
        lead_df = convert_lead()
        email_df = convert_email()
        hist_df = convert_hist()
        task_df = convert_task()
    return lead_df, email_df, hist_df, task_df

def convert_lead():
    '''
    INPUT: None
    OUTPUT:
        - lead_df: DataFrame
    Loads Lead.csv into a DataFrame, selects columns to use, does initial
    cleaning and feature engineering, and returns the resulting DataFrame.
    Saves the output to another csv file.
    '''
    lead_df = pd.read_csv('data/Lead.csv', parse_dates=[39, 41, 69, 114], low_memory=False) #Main data file
    lead_df = lead_df[['Id', #StudentId
                        'CreatedDate',
                        'LastModifiedDate',
                        'Last_Attendance__c',
                        'University__c',
                        'OwnerId', #CoachId
                        'Phone',
                        'MobilePhone',
                        'Email', #Email address
                        'Age__c',
                        'High_School_GPA__c',
                        'UG_GPA__c',
                        'SAT_Score__c',
                        'ACT_Score__c',
                        'Units_Attempted__c',
                        'Units_Completed__c',
                        'Units_Transferred__c',
                        'SAP_Standing__c',
                        'Finance_Status__c',
                        'Inbound_Applicants__c',
                        'Program_of_Study__c',
                        'Approved_SMS__c',
                        'Student_Motivation__c', #Few values for this column
                        'WGU_Notes__c',
                        'Bad_Contact_Info__c',
                        'Status']]
    lead_df.dropna(subset=['University__c'], inplace=True)
    lead_df['Age__c'] = 2017 - lead_df['Age__c'].dt.year
    lead_df.loc[lead_df['Last_Attendance__c'] > dt.datetime.today(), 'Last_Attendance__c'] = pd.NaT
    lead_df['WeeksSinceLastAttendance'] = (dt.datetime.today() - lead_df['Last_Attendance__c']).dt.days / 7.
    lead_df['Units_Unfinished'] = lead_df['Units_Attempted__c'] - lead_df['Units_Completed__c']
    lead_df.loc[:,['High_School_GPA__c', 'SAT_Score__c', 'ACT_Score__c',
                'Units_Completed__c', 'Units_Transferred__c']].replace(0, np.nan, inplace=True)
    lead_df.fillna({'Approved_SMS__c':0,
                    'Bad_Contact_Info__c':0},
                    inplace=True)
    lead_df.to_csv('lead_data.csv')
    return lead_df

def convert_email():
    '''
    INPUT: None
    OUTPUT:
        - email_df: DataFrame
    Loads EmailMessage.csv into a DataFrame, selects columns to use, and returns
    the resulting DataFrame.  Saves the output to another csv file.
    '''
    email_df = pd.read_csv('data/EmailMessage.csv', low_memory=False) #Email content
    email_df = email_df[['Id', #EmailMessageId
                            'ToAddress', #Email address; use this instead of ParentId
                            'MessageDate',
                            'Subject',
                            'TextBody',
                            'MessageSize']]
    email_df.to_csv('email_data.csv')
    return email_df

def convert_hist():
    '''
    INPUT: None
    OUTPUT:
        - hist_df: DataFrame
    Loads EntityHistory.csv into a DataFrame, selects columns to use, and returns
    the resulting DataFrame.  Saves the output to another csv file.
    '''
    hist_df = pd.read_csv('data/EntityHistory.csv', parse_dates=[5], low_memory=False) #Status changes
    hist_df = hist_df[[#'Id', #ChangeId
                        'ParentId', #StudentId
                        'CreatedDate',
                        'FieldName',
                        'OldvalString',
                        'NewvalString']]
    hist_df.to_csv('hist_data.csv')
    return hist_df

def convert_task():
    '''
    INPUT: None
    OUTPUT:
        - task_df: DataFrame
    Loads Task.csv into a DataFrame, selects columns to use, does initial
    cleaning and feature engineering, and returns the resulting DataFrame.
    Saves the output to another csv file.
    '''
    task_df = pd.read_csv('data/Task.csv', parse_dates=[6], low_memory=False) #Phone records
    for value in task_df['Subject'].unique():
        words = value.split()
        formatted = [word.strip(string.punctuation).lower() for word in words]
        if 'call' in formatted or 'phone' in formatted:
            task_df.loc[task_df['Subject']==value, 'Subject'] = 'Calls'
        elif 'email' in formatted:
            if 'mass' in formatted:
                task_df.loc[task_df['Subject']==value, 'Subject'] = 'MassEmails'
            else:
                task_df.loc[task_df['Subject']==value, 'Subject'] = 'Emails'
        elif 'text' in formatted or 'sms' in formatted:
            task_df.loc[task_df['Subject']==value, 'Subject'] = 'Texts'
        else:
            task_df.loc[task_df['Subject']==value, 'Subject'] = 'OtherOutreach'
    task_df.loc[(task_df['Subject']=='Calls') & (task_df['CallType']=='Inbound'), 'Subject'] = 'InboundCalls'
    task_df = task_df[['WhoId', #StudentId
                        'ActivityDate',
                        'Subject',
                        'Description',
                        'EmailMessageId',
                        'CallDurationInSeconds']]
    task_df.to_csv('task_data.csv')
    return task_df

def merge_all(lead_df, hist_df, task_df):
    '''
    INPUT:
        - lead_df: DataFrame
        - hist_df: DataFrame
        - task_df: DataFrame
    OUTPUT:
        - lead_df: DataFrame
    Merges relevant event-related data from both hist_df and task_df with
    the main DataFrame lead_df and returns merged lead_df.  Uses helper
    functions add_date_column and add_contact_column to engineer a number of
    date and contact related features.  Specifics given in comments throughout
    the function.
    '''
    status_changes = hist_df[hist_df['FieldName']=='Status']

    def add_date_column(lead_df, date_df, col_name, asc=False,
                        status=None, right_on='ParentId', date='CreatedDate'):
        if status:
            date_df = date_df[date_df['NewvalString']==status]
        date_rows = date_df.sort_values(date, ascending=asc).drop_duplicates(right_on)
        date_rows[col_name] = date_rows[date]
        date_rows = date_rows[[right_on, col_name]]
        return pd.merge(lead_df, date_rows, left_on='Id', right_on='ParentId', how='left')
    lead_df = add_date_column(lead_df, status_changes, 'FirstStatusUpdate', asc=True)
    lead_df = add_date_column(lead_df, status_changes, 'LastEntryDate')
    lead_df = add_date_column(lead_df, status_changes, 'NurturingDate', status='Nurturing')
    lead_df = add_date_column(lead_df, status_changes, 'AppliedDate', status='Applied')
    lead_df = add_date_column(lead_df, status_changes, 'StartDate', status='Start')

    #Removes StartDate from those who don't have 'Start' as their final status
    lead_df.loc[lead_df['Status']!='Start', 'StartDate'] = pd.NaT

    #Filters task_df to exclude contact dates after engaged status date
    merged = pd.merge(lead_df, task_df, left_on='Id', right_on='WhoId', how='left')
    task_df = merged[merged['ActivityDate'] < merged['LastEntryDate']]

    first_contacts = task_df.sort_values('ActivityDate').drop_duplicates('WhoId')
    first_contacts['FirstContactDate'] = first_contacts['ActivityDate']
    first_contacts['FirstContactType'] = first_contacts['Subject']
    first_contacts = first_contacts[['WhoId', 'FirstContactDate', 'FirstContactType']]
    lead_df = pd.merge(lead_df, first_contacts, left_on='Id', right_on='WhoId', how='left')

    #Fields for days it takes to reach engaged status from first contact
    lead_df['NewToApplied'] = (lead_df['AppliedDate'] - lead_df['FirstContactDate']).dt.days
    lead_df['NewToNurturing'] = (lead_df['NurturingDate'] - lead_df['FirstContactDate']).dt.days
    lead_df['AppliedToStart'] = (lead_df['StartDate'] - lead_df['FirstContactDate']).dt.days
    lead_df['NurturingToApplied'] = (lead_df['AppliedDate'] - lead_df['NurturingDate']).dt.days
    lead_df['DaysSinceFirstContact'] = (dt.datetime.today() - lead_df['FirstContactDate']).dt.days

    #Splits task_df Subject into separate columns for each value and counts
    contact_counts = task_df.groupby(['WhoId', 'Subject']).count()
    contact_counts.reset_index(inplace=True)

    def add_contact_column(lead_df, contact_counts, type_str):
        type_rows = contact_counts[contact_counts['Subject']==type_str]
        type_rows[type_str] = type_rows['ActivityDate']
        type_rows = type_rows[['WhoId', type_str]]
        return pd.merge(lead_df, type_rows, left_on='Id', right_on='WhoId', how='left', copy=False)
    call_types = ['Calls', 'InboundCalls', 'Emails', 'MassEmails', 'Texts', 'OtherOutreach']
    for type_str in call_types:
        lead_df = add_contact_column(lead_df, contact_counts, type_str)

    lead_df.drop(['ParentId_x', 'ParentId_y', 'WhoId', 'WhoId_x', 'WhoId_y'], axis=1, inplace=True)
    lead_df.loc[:,call_types] = lead_df[call_types].fillna(0)
    lead_df['TouchPoints'] = lead_df[call_types].sum(axis=1)
    return lead_df

def plot_histograms(df, lead_type='Outbound', window_size=60):
    '''
    INPUT:
        - df: DataFrame
        - lead_type: string
        - window_size: int
    OUTPUT: None
    Plots a histogram for those with engaged status and another for those
    without engaged status for each of the features, linear and non-linear, in
    the feature matrix.  To be used to aid in feature selection.
    '''
    X = clean_data(df, inbound=False, window_size=window_size, split=False)
    engaged_string = 'Engaged Status for {}: Mean = {}, StdDev = {} over {} data points'
    non_engaged_string = 'Non-engaged Status for {}: Mean = {}, StdDev = {} over {} data points'
    for column in X.columns:
        engaged = X[(X['Status']==1) & (X[column] != -1000)][column]
        non_engaged = X[(X['Status']==0) & (X[column] != -1000)][column]
        print engaged_string.format(column, engaged.mean(), engaged.std(), len(engaged))
        print non_engaged_string.format(column, non_engaged.mean(), non_engaged.std(), len(non_engaged)) + '\n'

        if engaged.count() != 0:
            engaged_bins = int(engaged.max() - engaged.min() + 1)
            if X[column].dtype == float:
                engaged_bins = 50
            plt.hist(engaged, bins=engaged_bins)
            plt.xlabel(column)
            plt.ylabel('Number of Leads')
            plt.title(column + ' for Engaged Statuses: ' + lead_type)
            plt.show()
        if non_engaged.count() != 0:
            non_engaged_bins = int(non_engaged.max() - non_engaged.min() + 1)
            if X[column].dtype == float:
                non_engaged_bins = 50
            plt.hist(non_engaged, bins=non_engaged_bins)
            plt.xlabel(column)
            plt.ylabel('Number of Leads')
            plt.title(column + ' for Non-Engaged Statuses: ' + lead_type)
            plt.show()

def windowize(df, window_size=60):
    '''
    INPUT:
        - df: DataFrame
        - window_size: int
    OUTPUT:
        - window: DataFrame (restricted to window_size days)
    Helper function for clean_data() prior to prepping data for regression.
    Restricts data to time window in days given by window_size keyword
    argument.  This controls for the fact that the future status of leads that
    have been entered into the database recently is uncertain.
    '''
    window = df[(df['LastEntryDate'].max() - df['FirstContactDate']).dt.days > window_size]
    window.loc[df['NewToNurturing'] > window_size, 'NewToNurturing'] = np.nan
    window.loc[df['NewToApplied'] > window_size, 'NewToApplied'] = np.nan
    window.loc[df['AppliedToStart'] > window_size, 'AppliedToStart'] = np.nan
    return window

def show_time_windows(outbound, show_total=True):
    '''
    INPUT:
        - outbound: DataFrame (must exclude inbound leads)
        - show_total: boolean (whether to plot total number of leads included)
    OUTPUT: None
    Plots number of total data points as well as number of leads who have
    reached a goal status against different window sizes and saves the graph to
    a .png file: time_windows_with_total if show_total = True and
    time_windows_without_total if show_total = False.  This function should be
    used to select an appropriate time window for restricting the data and is
    not necessary for the final workflow pipeline.
    '''
    window_counts = []
    nurturing_counts = []
    applied_counts = []
    nurt_applied_counts = []
    window_sizes = range(150)

    for window_size in window_sizes:
        window = outbound[(outbound['LastEntryDate'].max() - outbound['FirstContactDate']).dt.days > window_size]
        nurturing_counts.append(window['NewToNurturing'].count())
        applied_counts.append(window['NewToApplied'].count())
        nurt_applied_counts.append(window['NurturingToApplied'].count())
        if show_total:
            window_counts.append(window['Id'].count())

    plt.plot(window_sizes, nurturing_counts, label='First Contact to Nurturing Status')
    plt.plot(window_sizes, applied_counts, label='First Contact to Applied Status')
    plt.plot(window_sizes, nurt_applied_counts, label='Nurturing to Applied Status')
    if show_total:
        plt.plot(window_sizes, window_counts, label='Total Number of Leads')

    plt.xlabel('Size of Window')
    plt.ylabel('Number of Leads')
    plt.legend()

    if show_total:
        plt.savefig('time_windows_with_total.png')
    else:
        plt.savefig('time_windows_without_total.png')
    plt.close()


def clean_data(df, window_size=60, inbound=False, split=True):
    '''
    INPUT:
        - df: DataFrame
        - window_size: int (number of days for window)
        - inbound: boolean (whether df consists of inbound students)
        - split: boolean (whether to split linear columns into bins)
    OUTPUT:
        - window: DataFrame
    Subsets the the merged data from merge_all() by window_size and preps the
    data for fitting to a model such as Logit or DecisionTreeClassifier.  Use
    prior to splitting data into feature matrix X and target vector y.
    '''
    window = windowize(df, window_size=window_size)

    if inbound:
        status_binary = window['Status'] == 'Start'
    else:
        status_binary = (window['Status'] == 'Nurturing') | (window['Status'] == 'Applied')
        window = window[window['Status'] != 'New']
    window['Status'] = status_binary.astype(int)

    window.loc[window['FirstContactType'].isin(['Calls', 'InboundCalls', 'Texts']), 'FirstContactType'] = 'Phone'
    window.loc[window['SAP_Standing__c']=='Warning', 'SAP_Standing__c'] = 'Probation'

    linear_cols, indicator_cols, other_cols, dummy_cols, null_dummy_cols = get_columns()
    columns = list(linear_cols)

    for column in indicator_cols:
        window[column] = (~window[column].isnull()).astype(int)
    for column in columns:
        window[column] = window[column].fillna(-1000)
        if not split:
            new_col = column + 'NotNull'
            window[new_col] = (window[column] != -1000).astype(int)
            linear_cols.append(new_col)

    if split:
        window = split_columns(window)
    window = window[other_cols + linear_cols + indicator_cols]

    if split:
        dummy_cols += linear_cols
    window = pd.get_dummies(window, columns=dummy_cols, drop_first=True)
    window = pd.get_dummies(window, columns=null_dummy_cols).dropna()
    return window


def logreg(X, y, train_test=True, roc=True):
    '''
    INPUT:
        - X: 2-D feature matrix
        - y: target vector
        - train_test: boolean
        - roc: boolean
    OUTPUT:
        - fitted: fitted LogitResults
    Runs statsmodels Logistic Regression and prints summary.  Uses
    train_test_split to split data if train_test = True.  Plots and shows ROC
    curve if roc = True.  Returns fitted Logistic Regression model.
    '''

    if train_test:
        X_train, X_test, y_train, y_test = train_test_split(X, y)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    vifs, filtered = get_vifs(X_train)
    X_train, X_test = X_train[filtered], X_test[filtered]

    log_reg = Logit(y_train, add_constant(X_train, has_constant='add'))
    fitted = log_reg.fit(method='bfgs', maxiter=500)
    try:
        print fitted.summary()
    except:
        return logreg(X, y)
    if roc:
        plot_roc(y_test, fitted.predict(add_constant(X_test, has_constant='add')))

    return fitted

def get_vifs(X, verbose=False):
    '''
    INPUT:
        - X: feature matrix
        - verbose: boolean
    OUTPUT:
        - vifs: list of floats
        - columns: list of strings (same length as vifs)
    Filters columns in X by removing the feature with the highest variance
    inflation factor with the other features until all VIFs are less than 5.
    Returns the list of VIFs for the remaining columns and the list of columns
    selected by the function.  If verbose=True, prints the VIFs before every
    elimination up through the VIFs for the final selection.
    '''
    vifs = []
    columns = list(X.columns)
    while True:
        for i, column in enumerate(columns):
            vif = variance_inflation_factor(add_constant(X[columns], has_constant='add').values, i+1)
            vifs.append(vif)
            if verbose:
                print 'Variance Inflation Factor for {}: {}'.format(column, vif)
        if verbose:
            print ''
        if max(vifs) > 5:
            del columns[np.argmax(vifs)]
            vifs = []
        elif min(vifs) == 0:
            del columns[np.argmin(vifs)]
            vifs = []
        else:
            break
    return vifs, columns

def split_columns(df):
    '''
    INPUT:
        - df: 2-D feature matrix
    OUTPUT:
        - df: modified 2-D feature matrix
    Splits linear columns in the feature matrix into appropriate bins and
    returns the feature matrix.
    '''
    df['UG_GPA__c'] = pd.cut(df['UG_GPA__c'], bins=[-1000, 0, 3, 3.99, 4], include_lowest=True)
    df['WeeksSinceLastAttendance'] = pd.cut(df['WeeksSinceLastAttendance'], bins=[-1000, 0, 50, 100, 200, 1000], include_lowest=True)
    df['Units_Attempted__c'] = pd.cut(df['Units_Attempted__c'], bins=[-1000, -1, 15, 45, 75, 1000], include_lowest=True)
    df['Units_Completed__c'] = pd.cut(df['Units_Completed__c'], bins=[-1000, -1, 15, 45, 75, 1000], include_lowest=True)
    df['Units_Transferred__c'] = pd.cut(df['Units_Transferred__c'], bins=[-1000, -1, 15, 45, 75, 1000], include_lowest=True)
    df['Units_Unfinished'] = pd.cut(df['Units_Unfinished'], bins=[-1000, -1, 0, 1000], include_lowest=True)
    df['Age__c'] = pd.cut(df['Age__c'], bins=[-1000, 0, 25, 35, 50, 100], include_lowest=True)
    return df

def get_tree(X, y, train_test=True, roc=False):
    '''
    INPUT:
        - X: 2-D feature matrix
        - y: target vector
        - train_test: boolean
        - roc: boolean
    OUTPUT:
        - tree: DecisionTreeClassifier
    Runs a DecisionTreeClassifier on the cleaned data.  Uses train_test_split to
    split data if train_test = True.  Plots and shows ROC curve if roc = True.
    Exports visual representation of tree using pydot if pydot is installed.
    Returns the fitted DecisionTreeClassifier.
    '''
    tree = DecisionTreeClassifier(min_samples_leaf=20)

    if train_test:
        X_train, X_test, y_train, y_test = train_test_split(X, y)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
    tree.fit(X_train, y_train)

    if roc:
        plot_roc(y_test, tree.predict_proba(X_test)[:,1])

    export_graphviz(tree, out_file='tree.dot')
    try:
        (graph,) = pydot.graph_from_dot_file('tree.dot')
        graph.write_png('tree.png')
    except:
        pass

    return tree

def plot_roc(y_true, probabilities):
    '''
    INPUT:
        - y_true: target vector
        - probabilities: numpy array (same length as y_true)
    OUTPUT: None
    Helper function for logreg() and get_tree().  Plots the ROC curve for target
    variable based on probabilities given by an estimator such as Logit or
    DecisionTreeClassifier.
    '''
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--', color='k')
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("ROC plot")
    plt.show()

def univ_and_coach_breakdowns():
    '''
    INPUT: None
    OUTPUT: None
    Just prints out some value counts that are useful to know.
    '''
    print lead_df['University__c'].value_counts()
    print lead_df[lead_df['University__c']=='00141000002xrFJAAY']['Status'].value_counts()
    print lead_df[lead_df['University__c']=='00141000002xrFOAAY']['Status'].value_counts()
    print lead_df[lead_df['University__c']=='00141000002xrFTAAY']['Status'].value_counts()
    print lead_df[lead_df['University__c']=='00141000002xrFJAAY']['OwnerId'].value_counts()
    print lead_df[lead_df['University__c']=='00141000002xrFOAAY']['OwnerId'].value_counts()
    print lead_df[lead_df['University__c']=='00141000002xrFTAAY']['OwnerId'].value_counts()
    print lead_df[lead_df['Status']=='Start']['OwnerId'].value_counts()

def get_columns():
    '''
    INPUT: None
    OUTPUT:
        - linear_cols: list of strings
        - indicator_cols: list of strings
        - other_cols: list of strings
        - dummy_cols: list of strings
        - null_dummy_cols: list of strings
    Returns list of columns to be used in the model.  Modify this function to
    choose which columns to use in the final model.
    '''
    #Linear features with or without null values.
    linear_cols = [#'High_School_GPA__c',
                        'UG_GPA__c',
                        'WeeksSinceLastAttendance',
                        #'SAT_Score__c',
                        #'ACT_Score__c',
                        #'Units_Attempted__c',
                        'Units_Completed__c',
                        #'Units_Unfinished',
                        #'Units_Transferred__c',
                        'Age__c']
    #Features to be turned into binary variables: 1 if not null, 0 if null
    indicator_cols = [#'Email',
                            'Student_Motivation__c',
                            'WGU_Notes__c',
                            'Phone',
                            'MobilePhone']
    #Features that need no preprocessing prior to making dummies
    other_cols = ['OwnerId',
                        'University__c',
                        'FirstContactType',
                        'SAP_Standing__c',
                        'Approved_SMS__c',
                        #'Bad_Contact_Info__c',
                        'Status']
    #Features from other_cols with no null values to make dummies
    dummy_cols = ['OwnerId',
                        'FirstContactType',
                        'University__c']
    #Features from other_cols with null values to make dummies)
    null_dummy_cols = ['SAP_Standing__c']
    return linear_cols, indicator_cols, other_cols, dummy_cols, null_dummy_cols

def main(saved=False, inbound=False, window_size=30, roc=True):
    '''
    INPUT:
        - saved: boolean
        - inbound: boolean
        - window_size: int
        - roc: boolean
    OUTPUT:
        - lead_df: DataFrame
        - email_df: DataFrame
        - hist_df: DataFrame
        - task_df: DataFrame
        - X: 2D feature matrix (after cleaning)
        - y: 1D target vector
        - filtered: list of strings (column names in final model)
        - fitted: fitted Logit model
    The main pipeline to clean and combine the data and run it through the
    logistic regression model.
    '''
    lead_df, email_df, hist_df, task_df = load_data(saved=saved)
    lead_df = merge_all(lead_df, hist_df, task_df)
    if inbound:
        lead_df = lead_df[lead_df['Inbound_Applicants__c']==1]
    else:
        lead_df = lead_df[lead_df['Inbound_Applicants__c']!=1]

    X = clean_data(lead_df, inbound=inbound, split=True, window_size=30)
    y = X.pop('Status')
    vifs, filtered = get_vifs(X, verbose=False)
    fitted = logreg(X[filtered], y, train_test=False, roc=roc)

    return lead_df, email_df, hist_df, task_df, X, y, filtered, fitted


if __name__ == '__main__':
    '''
    saved: Use saved=False if you have not run the model before.  Use
    saved=True if you have already run the model and the initial cleaned
    DataFrames have been saved.

    inbound: Use inbound=True if running the model on inbound students; use
    inbound=False if running the model on non-inbound students.

    window_size: Time window in number of days to be considered for the model.
    Default is 30, but we can increase this number as more data comes in.

    roc: Use roc=True if you want to view the ROC curve; use roc=False
    otherwise.
    '''
    lead_df, email_df, hist_df, task_df, X, y, filtered, fitted = \
    main(saved=False, inbound=False, window_size=30, roc=True)
