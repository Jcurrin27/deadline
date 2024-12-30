import csv
import sys
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data()

    X_train, X_test, y_train, y_test = train_test_split(evidence, labels, test_size=TEST_SIZE)
    save_list(X_train, 'X_train.txt'); save_list(X_test, 'X_test.txt'); save_list(y_train, 'y_train.txt'); save_list(y_test, 'y_test.txt')

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

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
        if filename and os.path.isfile(filename):
            break
        else:
            filename = str(input("\nEnter the sample filename: "))
            folder = 'C:/Users/19402/deadline/sample_data'
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                break
            elif filename == 'exit':
                sys.exit("Exiting program.")
            else:
                print(f"invalid file directory: {file_path}. Remember to include the file extension. Type 'exit' to exit the program.")

    with open(file_path) as f:
        if file_path.lower().endswith('.json'):
            raise NotImplementedError("JSON file load not supported yet.")

        elif file_path.lower().endswith('.csv'):
            reader = csv.reader(f)
            next(reader)

            evidence = []
            labels = []

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

            for row in reader:
                # Convert and append the evidence, excluding 'dropped'
                evidence.append([
                    int(row[0]),                           # student_id
                    int(row[2]),                           # year
                    semester_encoder.transform([row[3]])[0],  # semester
                    term_encoder.transform([row[4]])[0],      # term
                    deadline_encoder.transform([row[5]])[0],  # deadline
                    int(row[6]),                           # days_to_deadline
                    float(row[7]),                         # balance
                    int(row[8]),                           # fafsa
                    int(row[9]),                           # verification
                    academic_level_encoder.transform([row[10]])[0],  # academic_level
                    int(row[11]),                          # international
                    int(row[12]),                          # waiver
                    int(row[13]),                          # total_cc
                    int(row[14]),                          # total_check_cash
                    int(row[15]),                          # flywire
                    int(row[16]),                          # sponsored
                    int(row[17])                           # returned_checks
                ])
                # Convert and append the label
                labels.append(1 if row[1] == '1' else 0) 

    return (evidence, labels)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    knn = KNeighborsClassifier(n_neighbors=1)

    model = knn.fit(evidence, labels)

    return model

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    match_positives = 0
    match_negatives = 0
    
    for i in range(len(labels)):
        if labels[i] == 1 and predictions[i] == 1:
            match_positives += 1
        elif labels[i] == 0 and predictions[i] == 0:
            match_negatives += 1

    positives = len([value for value in labels if value == 1])
    negatives = len([value for value in labels if value == 0])

    if (match_positives == 0) or (match_negatives == 0):
        print(f"match_positives: {match_positives}")
        print(f"match_negative: {match_negatives}")
        sys.exit("Error: Sensitivity or Specificity is 0")

    sensitivity = float(match_positives / positives)
    specificity = float(match_negatives / negatives)

    return (sensitivity, specificity)

def save_csv(df):
    df.to_csv('output.csv', index=False)

def save_list(list, filename=None):
    if filename == None:
        with open('output.txt', 'w') as f:
            for item in list:
                f.write("%s\n" % item)
    else:
        with open(filename, 'w') as f:
            for item in list:
                f.write("%s\n" % item)
if __name__ == "__main__":
    main()
