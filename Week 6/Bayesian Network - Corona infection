import pandas as pd

# Read the CSV file
data = pd.read_csv('country_wise_latest.csv')

# Create a DataFrame
country_data = pd.DataFrame(data)

# Print the first few rows of the DataFrame
print(country_data.head())

# Get the number of rows and columns in the DataFrame
print(f"Number of rows: {country_data.shape[0]}")
print(f"Number of columns: {country_data.shape[1]}")

# Get the column names
print("\nColumn names:")
print(country_data.columns)

# Get the data types of the columns
print("\nData types:")
print(country_data.dtypes)

# Get the unique countries/regions
print("\nUnique countries/regions:")
print(country_data['Country/Region'].unique())

# Get the total confirmed cases
total_confirmed = country_data['Confirmed'].sum()
print(f"\nTotal confirmed cases: {total_confirmed}")

# Get the total deaths
total_deaths = country_data['Deaths'].sum()
print(f"Total deaths: {total_deaths}")

# Get the total recovered cases
total_recovered = country_data['Recovered'].sum()
print(f"Total recovered cases: {total_recovered}")

# Calculate the mortality rate
mortality_rate = (total_deaths / total_confirmed) * 100
print(f"\nMortality rate: {mortality_rate:.2f}%")

# Calculate the recovery rate
recovery_rate = (total_recovered / total_confirmed) * 100
print(f"Recovery rate: {recovery_rate:.2f}%")

# Read the new dataset
new_data = pd.read_csv('new_dataset.csv')
new_data_df = pd.DataFrame(new_data)

# Print the first few rows of the new dataset
print("\nNew dataset:")
print(new_data_df.head())

# Get the number of rows and columns in the new dataset
print(f"\nNumber of rows: {new_data_df.shape[0]}")
print(f"Number of columns: {new_data_df.shape[1]}")

# Get the column names of the new dataset
print("\nColumn names:")
print(new_data_df.columns)

# Get the data types of the columns in the new dataset
print("\nData types:")
print(new_data_df.dtypes)

# Get the unique values in the new dataset
print("\nUnique values:")
print(new_data_df['Column1'].unique())
