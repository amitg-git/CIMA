import csv
import os
from config import Config


class UserManager:
    def __init__(self):
        self.csv_file = Config.USERS_CSV_FILE

        # Check if the file exists, if not, create it
        if not os.path.exists(self.csv_file):
            file = open(self.csv_file, 'w', newline='')
            file.close()

    def user_exists(self, user_id):
        with open(self.csv_file, 'r', newline='') as file:
            reader = csv.reader(file)
            return any(row[0] == user_id for row in reader)

    def add_user(self, user_id):
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([user_id])
