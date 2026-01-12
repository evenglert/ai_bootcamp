# # Introduction to Machine Learning Homework
# ## Author: Evgeniya Englert
# ## Last update: 2025-07-29
# ## Instructions:
# ### Your task is to create a Python program that helps manage a bootcamp. 
# ### Your program should allow the user to perform the following tasks:
# 
# * Add a student: The program should prompt the user to enter the student's name, age, and email address. The program should then add the student to the bootcamp's database.
# * Remove a student: The program should prompt the user to enter the name of the student they want to remove. The program should then remove the student from the database.
# * View all students: The program should display a list of all the students in the bootcamp, along with their names, ages, and email addresses.
# * Search for a student: The program should prompt the user to enter the student's name they want to search for. The program should then display the student's information if they are found in the database.
# * Exit the program: The program should allow the user to exit the program.
# * Your program should use the input() and print() functions to get input from the user and display output to the user. You should also use lists and dictionaries to store and manipulate data. The database should be read and stored in a file called bootcamp.txt.

# %%
import os

DATABASE_FILE = "bootcamp.txt"

# %%
# Load existing students from the database file
def load_students():
    """Loads student data from the database file."""
    students = []
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'r') as f:
            for line in f:
                name, age, email = line.strip().split(',')
                students.append({"name": name, "age": int(age), "email": email})
    return students

# %%
# Save students to the database file
def save_students(students):
    """Saves student data to the database file."""
    with open(DATABASE_FILE, 'w') as f:
        for student in students:
            f.write(f"{student['name']},{student['age']},{student['email']}\n")

# %%
# Add a student: The program should prompt the user to enter the student's name, age, and email address. The program should then add the student to the bootcamp's database.
def add_student(students):
    """Prompts the user for student details and adds them to the list."""
    name = input("Enter student's name: ").strip()
    age = int(input("Enter student's age: "))
    email = input("Enter student's email address: ").strip()
    students.append({"name": name, "age": age, "email": email})
    save_students(students)
    print(f"\nStudent '{name}' added successfully!\n")

# %%
# Remove a student: The program should prompt the user to enter the name of the student they want to remove. The program should then remove the student from the database.
def remove_student(students):
    """Removes a student from the list based on their name."""
    name_to_remove = input("Enter the name of the student to remove: ").strip()
    original_len = len(students)
    students[:] = [s for s in students if s["name"].lower() != name_to_remove.lower()]
    if len(students) < original_len:
        save_students(students)
        print(f"\nStudent '{name_to_remove}' removed successfully!\n")
    else:
        print(f"\nStudent '{name_to_remove}' not found.\n")

# %%
# View all students: The program should display a list of all the students in the bootcamp, along with their names, ages, and email addresses.
def view_all_students(students):
    """Displays a list of all students and their details."""
    if not students:
        print("\nNo students in the bootcamp yet.\n")
        return
    print("\n--- All Students ---")
    for student in students:
        print(f"Name: {student['name']}, Age: {student['age']}, Email: {student['email']}")
    print("--------------------\n")

# %%
# Search for a student: The program should prompt the user to enter the student's name they want to search for. The program should then display the student's information if they are found in the database.
def search_student(students):
    """Searches for a student by name and displays their information."""
    name_to_search = input("Enter the name of the student to search for: ").strip()
    found = False
    for student in students:
        if student["name"].lower() == name_to_search.lower():
            print("\n--- Student Found ---")
            print(f"Name: {student['name']}, Age: {student['age']}, Email: {student['email']}")
            print("---------------------\n")
            found = True
            break
    if not found:
        print(f"\nStudent '{name_to_search}' not found.\n")

# %%
# Your program should use the input() and print() functions to get input from the user and display output to the user. You should also use lists and dictionaries to store and manipulate data. The database should be read and stored in a file called bootcamp.txt.
def main():
    """Main function to run the bootcamp management program."""
    students = load_students()

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
        # Exit the program: The program should allow the user to exit the program.
        elif choice == '5':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

# %%
# Run the menu and interact with a student database
if __name__ == "__main__":
    main()


