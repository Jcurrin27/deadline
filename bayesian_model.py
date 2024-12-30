import csv
import sys
import os
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

    filename = 'historic_data_1.csv'
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

def get_prior(data):
    """
    Calculate the prior probability of a student dropping the course.
    """
    num_dropped = 0
    num_students = len(data)

    for key in data:
        if data[key]['label'] == 1:
            num_dropped += 1

    return num_dropped / num_students

def get_likelihood(data, feature):
    num_dropped = 0
    num_students = len(data)
    num_feature_dropped = 0
    num_feature_not_dropped = 0

    for key in data:
        if data[key]['label'] == 1:
            num_dropped += 1
            if data[key]['evidence'][feature] == 1:
                num_feature_dropped += 1
        else:
            if data[key]['evidence'][feature] == 1:
                num_feature_not_dropped += 1

    # Add a small smoothing factor (e.g., 1) to avoid zero probabilities
    smoothing_factor = 1

    p_dropped = (num_dropped + smoothing_factor) / (num_students + 2 * smoothing_factor)
    p_not_dropped = 1 - p_dropped
    p_feature_given_dropped = (num_feature_dropped + smoothing_factor) / (num_dropped + 2 * smoothing_factor)
    p_feature_given_not_dropped = (num_feature_not_dropped + smoothing_factor) / (num_students - num_dropped + 2 * smoothing_factor)

    p_feature = p_feature_given_dropped * p_dropped + p_feature_given_not_dropped * p_not_dropped
    likelihood = p_feature_given_dropped * p_dropped / p_feature

    return likelihood

def get_evidence(data, feature):
    """
    Calculate the probability of a specific feature.
    """
    num_feature = 0
    num_students = len(data)

    for key in data:
        if data[key]['evidence'][feature] == 1:
            num_feature += 1

    return num_feature / num_students

def bayes_theorem(prior, likelihood, evidence):

    posterior = (likelihood * prior) / evidence
    return posterior

def predict(data, record_key):

    record = data[record_key]
    evidence = record['evidence']

    # calculate priors
    prior_dropped = get_prior(data)
    prior_not_dropped = 1 - prior_dropped

    # initialize posteriors
    posterior_dropped = prior_dropped
    posterior_not_dropped = prior_not_dropped

    # iterate over features to update posteriors
    for feature, value in evidence.items():
        likelihood_dropped = get_likelihood(data, feature) if value == 1 else 1 - get_likelihood(data, feature)
        likelihood_not_dropped = 1 - likelihood_dropped

        # update posteriors
        posterior_dropped *= likelihood_dropped
        posterior_not_dropped *= likelihood_not_dropped

    # normalize to ensure probabilities sum to 1
    normalization_factor = posterior_dropped + posterior_not_dropped
    posterior_dropped /= normalization_factor
    posterior_not_dropped /= normalization_factor

    return 1 if posterior_dropped > posterior_not_dropped else 0

def main():
    # intiate timer
    start_time = time.time()

    data = load_data()
    # print(data['evidence'])
    # print(f"data keys = {data.keys()}")
    train_data, test_data = train_test_split(list(data.keys()), test_size=0.3, random_state=42)

    # Validate that all keys exist in data
    assert all(key in data for key in train_data), "Mismatch in train_data keys"
    assert all(key in data for key in test_data), "Mismatch in test_data keys"
    
    # train model on training data
    trained_data = {key: data[key] for key in train_data}

    # test model on test data
    correct = 0
    total = len(test_data)
    predicted_counter = 0
    errors = 0

    for key in test_data:
        for i in range(len(data['key'])):
            if predicted_counter > 100:
                break
            try:
                actual = data[key][i]
                predicted = predict(data, key)
                predicted_counter += 1
                if actual == predicted:
                    correct += 1
                print(f"Predicted count: {predicted_counter}/{total}", end='\r')
            except KeyError:
                print(f"KeyError: {key}")
                errors += 1
                continue
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Errors: {errors}")

    # example prediction
    # example_key = test_data[0]
    # print(f"Prediction for record {example_key}: {predict(trained_data, example_key)}")
    # print(f"Actual label: {data[example_key]['label']}")

    # end timer and print runtime
    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()