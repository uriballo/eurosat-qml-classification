import re
import os

def extract_test_accuracies_from_log(log_text):
    # Regular expression pattern to match test accuracy lines
    pattern = r"Test Accuracy: (\d+\.\d+)%"

    # Extract test accuracies
    test_accuracies = re.findall(pattern, log_text)

    # Convert accuracies to floats
    test_accuracies = [float(acc) for acc in test_accuracies]

    return test_accuracies

folder_path = "logs/"

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path) and filename.endswith(".log"):  # Adjust file extension as needed
        with open(file_path, "r") as file:
            log_text = file.read()
            test_accuracies = extract_test_accuracies_from_log(log_text)
            print(f"{filename[:-4]} = {test_accuracies}\n\n")