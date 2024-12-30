import csv
import json
import os

def main():
    # datafile = input("Enter file name for sample data: ")
    datafile = "current_student_1.csv"
    file_path = os.path.join(r'sample_data', datafile)

    csv_data = read_csv(file_path)
    json_data = convert_to_json(csv_data)
    export(json_data)

def read_csv(file_path):
    """Reads a CSV file and returns a list of dictionaries."""
    data = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def convert_to_json(csv_data): 
    """Converts CSV data (list of dictionaries) to a JSON string."""
    return json.dumps(csv_data, indent=4)

def export(json):
    datafile = "current_student_1.json"
    file_path = os.path.join(r'sample_data', datafile)
    with open(file_path, mode='w', encoding='utf-8') as file:
        file.write(json)
    print(f"Data has been successfully exported to {file_path}")

if __name__ == "__main__":
    main()
