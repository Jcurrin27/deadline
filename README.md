    Intro
This project creates a model for student payment deadline predictions. The model uses historic data to predict whether a student will be dropped for non-payment on an upcoming deadline. 

    Notes on Sample Data
The data used for the model was artificially created by me. I used data types that are available to higher ed administrators, but the actual data values are not real. The data includes some intentional skews in order to create patterns the model can use to identify students at risk. For example, some students in the historic data file that were dropped have higher outstanding balances and no FAFSA on file. However, the sample data files are simply holding places for actual data files. If authentic data is gathered in the provided format, then the model could be tested using real student data. Additionally, if authentic data is gathered in a slightly different format (with different data fields), then the model would only require slight adjustments to function properly. 

    Design
The program is designed to run against records of all students who have outstanding balances for an upcoming payment deadline, and have at least one year of payment/enrollment history. The program would require historic payment data to be gathered for each student, and using this historic data, the program can predict whether they will be dropped for the upcoming deadline using a k-nearest neighbor classification model.  
    
