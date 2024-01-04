import pandas as pd
import time
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Function to load and preprocess data from CSV files
def load_and_preprocess_data(grocery_file, retail_file):
    # Load grocery data and retail data from CSV files
    grocery_data = pd.read_csv(grocery_file)
    retail_data = pd.read_csv(retail_file)
    return grocery_data, retail_data

# Function to create a user-category matrix from grocery data
def create_user_category_matrix(grocery_data):
    # Pivot the grocery data to create a matrix with customers as rows, categories as columns
    # Replace missing values with 0 and count occurrences of each category per customer
    user_category_matrix = grocery_data.pivot_table(index='Customer Name', columns='Category', aggfunc='size', fill_value=0)
    return user_category_matrix

# Function to get top categories for a user using the KNN algorithm
def get_top_categories(user, user_category_matrix, top_n=10):
    # Initialize a KNN model with cosine similarity metric
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    # Fit the model on the user-category matrix
    model_knn.fit(csr_matrix(user_category_matrix.values))

    # If the user is not in the matrix, return an empty list
    if user not in user_category_matrix.index:
        return []

    # Find the nearest neighbors (similar users) for the given user
    distances, indices = model_knn.kneighbors(user_category_matrix.loc[user, :].values.reshape(1, -1), n_neighbors=top_n + 1)
    
    # Collect top categories from similar users
    top_categories = set()
    for i in range(1, len(distances.flatten())):
        similar_user = user_category_matrix.index[indices.flatten()[i]]
        top_categories.update(user_category_matrix.loc[similar_user].nlargest(top_n).index.tolist())

    return list(top_categories)[:top_n]

# Function to recommend items from retail data based on top categories
def recommend_items(user, top_categories, grocery_data, retail_data, top_n=10):
    # Get the items that the user has already purchased
    purchased_items = grocery_data[grocery_data['Customer Name'] == user]['Sub Category'].unique().tolist()
    
    recommended_items = []
    for category in top_categories:
        # Extract items under each category
        items = retail_data[retail_data['Category'] == category]['Description'].unique().tolist()
        # Exclude items that the user has already purchased
        items = [item for item in items if item not in purchased_items]
        recommended_items.extend(items)
    
    return recommended_items[:top_n]

# Main function to run the recommendation system
def run_recommendation_system(grocery_file, retail_file, test_user):
    # Initialize all variables to their empty states
    grocery_data = pd.DataFrame()
    retail_data = pd.DataFrame()
    user_category_matrix = pd.DataFrame()
    top_categories = []
    recommendations = []

    # Load data and create matrices
    grocery_data, retail_data = load_and_preprocess_data(grocery_file, retail_file)
    user_category_matrix = create_user_category_matrix(grocery_data)
    # Get top categories for the user
    top_categories = get_top_categories(test_user, user_category_matrix)
    # Get item recommendations based on top categories
    recommendations = recommend_items(test_user, top_categories, grocery_data, retail_data)
    
    return recommendations

# File paths for the data
grocery_file_path = 'grocery_sells.csv'
retail_file_path = 'Online_Retail_Categorized.csv'

# Function to interact with the user and get the selected customer name
def get_user_choice(grocery_file_path):
    grocery_data = pd.read_csv(grocery_file_path)
    customer_names = grocery_data['Customer Name'].unique()
    print("List of customers:")
    # Display the list of customers for selection
    for idx, name in enumerate(customer_names, start=1):
        print(f"{idx}. {name}")
    try:
        choice = int(input("Enter the number of the customer you want recommendations for: "))
        # check if the choice is an integer and within the range of the list
        if (choice>len(customer_names) or choice<1 or isinstance(choice, float)):
            print("Invalid choice. Please enter a number within 1-50.")
            # add timer to wait for 2 seconds
            time.sleep(2)
            return get_user_choice(grocery_file_path)
    except ValueError:
        print("Invalid choice. Please enter a valid number.")
        # add timer to wait for 2 seconds
        time.sleep(2)
        return get_user_choice(grocery_file_path)

    return customer_names[choice - 1]
    

# Main function to execute the recommendation system
def main():
    user_input = input("Do you want to run the recommendation system? (yes/no): ").strip().lower()
    if user_input == 'yes':
        # Get the user's choice and run the recommendation system
        selected_customer = get_user_choice(grocery_file_path)
        recommendations = run_recommendation_system(grocery_file_path, retail_file_path, selected_customer)
        print(f"Top recommendations for {selected_customer}: {recommendations}")
        input ("Press enter to exit")
        
    elif user_input == 'no':
        print("Exiting the recommendation system.")
    else:
        print("Invalid input. Please enter yes or no.")
        main()

    

main()