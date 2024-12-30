import csv
import sys
import os
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def main():
    data = load_data(filename='historic_data_1.csv')
    # print(type(data))
    # print(len(data['evidence']))
    # first_record = data['evidence']['student_id']
    # print(first_record)
    total = len(data['evidence'])
    print(f"total = {total}")
    for i in total:
        print(data['evidence']['student_id'][i])
        i += 1
    # for variable in data['evidence']:
    #     i = 0
    #     for i in (len(record) - 19995):
    #         print(record[i])
    #         i += 1


def load_data(filename=None):
    """
    Load student deadline data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - student_id, an integer
        - year, an integer
        - semester, a string
        - term, a string
        - deadline, a string
        - days_to_deadline, an integer
        - balance, a floating point number
        - fafsa, an integer
        - verification, an integer
        - academic_level, a string
        - international, an integer
        - waiver, an integer
        - total_cc, an integer
        - total_check_cash, an integer
        - flywire, an integer
        - sponsored, an integer
        - returned_checks, an integer

    labels should be the corresponding list of labels, where each label
    is 1 if the student dropped, and 0 otherwise.
    """

    while True:
        if filename:
            folder = 'C:/Users/19402/deadline/sample_data'
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                break
        else:
            filename = str(input("\nEnter the sample filename: "))
            if os.path.isfile(file_path):
                break
            elif filename == 'exit':
                sys.exit("Exiting program.")
            else:
                print(f"invalid file directory: {file_path}. Remember to include the file extension. Type 'exit' to exit the program.")

    print(f"Loading data...")
    with open(file_path) as f:
        if file_path.lower().endswith('.json'):
            raise NotImplementedError("JSON file load not supported yet.")

        elif file_path.lower().endswith('.csv'):
            reader = csv.reader(f)
            next(reader)  # Skip header row

        data = {}

        # Initialize label encoders for categorical features
        semester_encoder = LabelEncoder()
        term_encoder = LabelEncoder()
        deadline_encoder = LabelEncoder()
        academic_level_encoder = LabelEncoder()

        # Collect all categorical data for fitting the encoders
        semesters = []
        terms = []
        deadlines = []
        academic_levels = []

        for row in reader:
            semesters.append(row[3])
            terms.append(row[4])
            deadlines.append(row[5])
            academic_levels.append(row[10])

        # Fit the encoders
        semester_encoder.fit(semesters)
        term_encoder.fit(terms)
        deadline_encoder.fit(deadlines)
        academic_level_encoder.fit(academic_levels)

        # Reset the reader to read the file again
        f.seek(0)
        next(reader)

        rows_loaded = 0
        for row in reader:
            rows_loaded += 1
            print(f"Loading row {rows_loaded}...\r", end="")
            evidence = {
                "student_id": int(row[0]), 
                "year": int(row[2]),
                "semester": semester_encoder.transform([row[3]])[0],
                "term": term_encoder.transform([row[4]])[0],
                "deadline": deadline_encoder.transform([row[5]])[0],
                "days_to_deadline": int(row[6]),
                "balance": float(row[7]),
                "fafsa": int(row[8]),
                "verification": int(row[9]),
                "academic_level": academic_level_encoder.transform([row[10]])[0],
                "international": int(row[11]),
                "waiver": int(row[12]),
                "total_cc": int(row[13]),
                "total_check_cash": int(row[14]),
                "flywire": int(row[15]),
                "sponsored": int(row[16]),
                "returned_checks": int(row[17])
            }
            label = 1 if row[1] == '1' else 0

            # Store the evidence and label in the dictionary
            data = {
                "evidence": evidence,
                "label": label
            }
        return data
    
if __name__ == '__main__':
    main()