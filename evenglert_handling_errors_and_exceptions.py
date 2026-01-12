# # Handling Errors and Exception: Homework
# # Author: Evgeniya Englert
# # Last update: 2025-07-30
# ## Instructions:
# #### Building on the previous lesson's homework, your task is to modify the program you created to handle errors and exceptions. For example, your program should be able to handle the following errors and exceptions:
# * Invalid user input: The user may enter invalid input, such as a string, when an integer is expected.
# * Removing a student who is not in the database: If the user tries to remove a student who is not in the database, you can handle this by checking if the student exists before removing them.
# * Searching for a student who does not exist: If the user tries to search for a student who does not exist, you can handle this by checking if the student exists before displaying their information.
# * File or database errors: If your program is storing data in a file or database, there may be errors related to reading or writing to the file or database.
# * All of these errors and exceptions should be handled by your program using user-defined exceptions. 
# * If an error or exception occurs, your program should display an appropriate message to the user and allow them to continue using it.

# %%
import os

DATABASE_FILE = "bootcamp.txt"

# --- User-Defined Exceptions ---
class StudentNotFoundError(Exception):
    """Custom exception raised when a student is not found in the database."""
    pass

class InvalidInputError(Exception):
    """Custom exception raised for invalid user input (e.g., non-integer where int is expected)."""
    pass

class DatabaseError(Exception):
    """Custom exception raised for errors related to file/database operations."""
    pass

# --- Helper Functions (with error handling) ---
def load_students():
    """Loads student data from the database file, handling file errors."""
    students = []
    if not os.path.exists(DATABASE_FILE):
        return students # Return empty list if file doesn't exist yet

    try:
        with open(DATABASE_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 3: # Ensure correct format
                    name, age_str, email = parts
                    try:
                        age = int(age_str)
                        students.append({"name": name, "age": age, "email": email})
                    except ValueError:
                        print(f"Warning: Skipping malformed student record (invalid age): {line.strip()}")
                else:
                    print(f"Warning: Skipping malformed student record (incorrect format): {line.strip()}")
        return students
    except IOError as e:
        raise DatabaseError(f"Error reading from database file: {e}")

def save_students(students):
    """Saves student data to the database file, handling file errors."""
    try:
        with open(DATABASE_FILE, 'w') as f:
            for student in students:
                f.write(f"{student['name']},{student['age']},{student['email']}\n")
    except IOError as e:
        raise DatabaseError(f"Error writing to database file: {e}")

# --- Core Program Functions (with error handling) ---

def add_student(students):
    """Prompts the user for student details and adds them to the list, with input validation."""
    try:
        name = input("Enter student's name: ").strip()
        if not name:
            raise InvalidInputError("Student name cannot be empty.")

        try:
            age = int(input("Enter student's age: "))
            if age <= 0:
                raise InvalidInputError("Student age must be a positive number.")
        except ValueError:
            raise InvalidInputError("Invalid age. Please enter an integer.")

        email = input("Enter student's email address: ").strip()
        if not email or "@" not in email:
            raise InvalidInputError("Invalid email address.")

        students.append({"name": name, "age": age, "email": email})
        save_students(students)
        print(f"\nStudent '{name}' added successfully!\n")
    except InvalidInputError as e:
        print(f"Error: {e}\n")
    except DatabaseError as e:
        print(f"Database Error: {e}\n")

def remove_student(students):
    """Removes a student from the list based on their name, handling StudentNotFoundError."""
    try:
        name_to_remove = input("Enter the name of the student to remove: ").strip()
        if not name_to_remove:
            raise InvalidInputError("Student name for removal cannot be empty.")

        original_len = len(students)
        # Create a new list excluding the student to remove
        updated_students = [s for s in students if s["name"].lower() != name_to_remove.lower()]

        if len(updated_students) == original_len:
            raise StudentNotFoundError(f"Student '{name_to_remove}' not found in the database.")
        else:
            students[:] = updated_students # Update the original list
            save_students(students)
            print(f"\nStudent '{name_to_remove}' removed successfully!\n")
    except InvalidInputError as e:
        print(f"Error: {e}\n")
    except StudentNotFoundError as e:
        print(f"Error: {e}\n")
    except DatabaseError as e:
        print(f"Database Error: {e}\n")

def view_all_students(students):
    """Displays a list of all students and their details."""
    if not students:
        print("\nNo students in the bootcamp yet.\n")
        return
    print("\n--- All Students ---")
    for student in students:
        print(f"Name: {student['name']}, Age: {student['age']}, Email: {student['email']}")
    print("--------------------\n")

def search_student(students):
    """Searches for a student by name and displays their information, handling StudentNotFoundError."""
    try:
        name_to_search = input("Enter the name of the student to search for: ").strip()
        if not name_to_search:
            raise InvalidInputError("Student name for search cannot be empty.")

        found_student = None
        for student in students:
            if student["name"].lower() == name_to_search.lower():
                found_student = student
                break

        if found_student:
            print("\n--- Student Found ---")
            print(f"Name: {found_student['name']}, Age: {found_student['age']}, Email: {found_student['email']}")
            print("---------------------\n")
        else:
            raise StudentNotFoundError(f"Student '{name_to_search}' not found in the database.")
    except InvalidInputError as e:
        print(f"Error: {e}\n")
    except StudentNotFoundError as e:
        print(f"Error: {e}\n")

def main():
    """Main function to run the bootcamp management program."""
    try:
        students = load_students()
    except DatabaseError as e:
        print(f"Critical Error: {e}. Cannot load student data. Program will start with an empty list.\n")
        students = [] # Start with an empty list if loading fails

    while True:
        print("Bootcamp Management System")
        print("1. Add a student")
        print("2. Remove a student")
        print("3. View all students")
        print("4. Search for a student")
        print("5. Exit the program")

        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            add_student(students)
        elif choice == '2':
            remove_student(students)
        elif choice == '3':
            view_all_students(students)
        elif choice == '4':
            search_student(students)
        elif choice == '5':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.\n")

if __name__ == "__main__":
    main()


