# Analysis of College Reapplication Data

Worked with data from a company that helps students reapply to college after having dropped out.  Created a Logistic Regression model to determine which factors are most influential in determining whether a student will reapply.  Used pandas library to query and clean the data.

### Motivation

The company would like to know how to use its resources to most effectively and efficiently reach out to students.  To do so, they need to know when students are more likely or less likely to apply so that they can prioritize accordingly.

### Overview

Data is read from four main csv files:  
--Lead.csv: The main file; contains data for the students themselves (23,691 rows, one for each lead; 138 columns)  
--EntityHistory.csv: Contains data that logs and timestamps certain changes in the database, including entry of new leads into the database and status changes of those leads (18 columns)  
--Task.csv: Contains data for tasks related to engaging leads, including calls, e-mails, and other points of contact (45 columns)  
--EmailMessage.csv: Contains data on e-mails sent to leads, including e-mail text (28 columns)  

Each file is read into a pandas DataFrame and relevant columns are selected.  Features are engineered from the EntityHistory, Task, and EmailMessage files by aggregating, combining, or selecting events pertaining to each lead.  These features are then merged with the main Lead DataFrame.

After merging, the data undergoes additional cleaning and prepping, including variance inflation factor analysis to reduce covariance between features.  A time window is selected to make valid comparisons between the statuses of leads.  Once cleaned, the data is used to train a Logistic Regression model.

### Time Window

Suppose we have two students who were contacted for the first time a week ago.  If a student reaches Nurturing status, that student receives a 1 indicating success.  If the student stayed at attempted contact status, that student receives a zero indicating failure.  However, there’s a problem with this; we don’t know what the future status of someone who has not yet reached Nurturing status may be.  The second student might reach Nurturing status a week from now.  Do we really want to call that a failure just because they have not reached that status within a week?  That’s not a lot of time to make such an important decision.

The solution is to choose a time window.  I used a 30 day time window, which means that if the student reaches Nurturing status within thirty days of the first attempted contact, we call it a success.  We ignore anything that happens after the 30 day window.  Students who were first contacted less than 30 days ago will not be included in the analysis, since they do not have a full 30 day window to look at.

### Results

Question:

Using a time window of 30 days, the Logistic Regression model yields the following tentative insights:

--Students in the 35-50 age range are more likely to reach Nurturing status  
--Students who last attended college less than a year ago are less likely to reach Nurturing status  
--Students with an undergrad GPA under 3.0 are more likely to reach Nurturing status  
--Students with mobile phones listed in the database are more likely to reach Nurturing status  
--Students who were first contacted by phone are more likely to reach Nurturing status than those first contacted by e-mail  
--Different coaches do not appear to have a large effect on whether students reach Nurturing status

However, it is uncertain that these conclusions will remain valid for larger time windows.  As of now, data only exists over a period of a few months, so we cannot draw any conclusions about whether or not students will apply given a six-month timeframe, which would be a more reasonable timeframe given the importance of the decision.  The company should return to this analysis once more data has come in.
