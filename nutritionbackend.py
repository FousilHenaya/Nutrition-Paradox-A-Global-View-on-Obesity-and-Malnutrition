PROJECT 2 Nutrition Paradox: A Global View on Obesity and Malnutrition

Step 1:Dataset Overview & Collection

import requests
import numpy as np
import pandas as pd
import pycountry
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import mysql.connector as db

url1 = "https://ghoapi.azureedge.net/api/NCD_BMI_30C"
result1= requests.get(url1)
data1=(result1.json())
df1= pd.DataFrame(data1["value"])

url2= "https://ghoapi.azureedge.net/api/NCD_BMI_PLUS2C"
result2= requests.get(url2)
data2=(result2.json())
df2= pd.DataFrame(data2["value"])

url3 = "https://ghoapi.azureedge.net/api/NCD_BMI_18C"
result3= requests.get(url3)
data3=(result3.json())
df3= pd.DataFrame(data3["value"])

url4 = "https://ghoapi.azureedge.net/api/NCD_BMI_MINUS2C"
result4= requests.get(url4)
data4=(result4.json())
df4= pd.DataFrame(data4["value"])

df1["age_group"]="adult"
df2["age_group"]="children"
df3["age_group"]="adult"
df4["age_group"]="children"

df_obesity = pd.concat([df1, df2], ignore_index=True)
df_malnutrition = pd.concat([df3, df4], ignore_index=True)

df_obesity = df_obesity.drop(df_obesity[df_obesity['TimeDim'] < 2012].index)
df_obesity.reset_index(drop=True, inplace=True)

df_malnutrition = df_malnutrition.drop(df_malnutrition[df_malnutrition['TimeDim'] < 2012].index)
df_malnutrition.reset_index(drop=True, inplace=True)

Step 2: 🧹 Data Cleaning & Feature Engineering

columns_to_keep= ["ParentLocation","Dim1","TimeDim","Low", "High", "NumericValue","SpatialDim","age_group"]
df_obesity = df_obesity[columns_to_keep]
df_malnutrition = df_malnutrition[columns_to_keep]

df_obesity = df_obesity.rename(columns={
    "ParentLocation":"Region",
    "Dim1":"Gender",
    "TimeDim":"Year",
    "Low":"LowerBound", 
    "High":"UpperBound", 
    "NumericValue":"Mean_Estimate",
    "SpatialDim":"Country"})

df_malnutrition = df_malnutrition.rename(columns={
    "ParentLocation":"Region",
    "Dim1":"Gender",
    "TimeDim":"Year",
    "Low":"LowerBound", 
    "High":"UpperBound", 
    "NumericValue":"Mean_Estimate",
    "SpatialDim":"Country"})

df_obesity["Gender"] = df_obesity["Gender"].replace({
    "SEX_MLE": "Male",
    "SEX_FMLE": "Female",
    "SEX_BTSX": "Both"
})

df_malnutrition["Gender"] = df_malnutrition["Gender"].replace({
    "SEX_MLE": "Male",
    "SEX_FMLE": "Female",
    "SEX_BTSX": "Both"
})

!pip install pycountry

special_cases = {
    'GLOBAL': 'Global',
    'WB_LMI': 'Low & Middle Income',
    'WB_HI': 'High Income',
    'WB_LI': 'Low Income',
    'EMR': 'Eastern Mediterranean Region',
    'EUR': 'Europe',
    'AFR': 'Africa',
    'SEAR': 'South-East Asia Region',
    'WPR': 'Western Pacific Region',
    'AMR': 'Americas Region',
    'WB_UMI': 'Upper Middle Income'
}

def get_country_name(code):
    
    if code in special_cases:
        return special_cases[code]
    
    try:
        return pycountry.countries.get(alpha_3=code).name
    except AttributeError:
        return "Unknown"

df_obesity["Country"] = df_obesity["Country"].apply(get_country_name)

df_malnutrition["Country"] = df_malnutrition["Country"].apply(get_country_name)

# Calculate CI_Width
df_obesity["CI_Width"] = df_obesity["UpperBound"] - df_obesity["LowerBound"]
df_malnutrition["CI_Width"] = df_malnutrition["UpperBound"] - df_malnutrition["LowerBound"]

# Define obesity levels
def categorize_obesity(value):
    if value >= 30:
        return "High"
    elif 25 <= value < 30:
        return "Moderate"
    else:
        return "Low"

# Apply categorization for obesity level
df_obesity["obesity_level"] = df_obesity["Mean_Estimate"].apply(categorize_obesity)

# Define malnutrition levels
def categorize_malnutrition(value):
    if value >= 20:
        return "High"
    elif 10 <= value < 20:
        return "Moderate"
    else:
        return "Low"

# Apply categorization for malnutrition level
df_malnutrition["malnutrition_level"] = df_malnutrition["Mean_Estimate"].apply(categorize_malnutrition)

Step:3 🧮 Exploratory Data Analysis (EDA) with Python

# Check the number of rows and columns
print(df_obesity.shape)

# Display the first few rows to inspect data
print(df_obesity.head())

# Check for data types of each column
print(df_obesity.dtypes)

# Summary statistics to get an overview of the numeric columns
print(df_obesity.describe())

# Check for missing values
print(df_malnutrition.isnull().sum())

# Check for duplicates
print(df_malnutrition.duplicated().sum())

# Check for unusual or outlier values (e.g., negative values for measurements)
print(df_malnutrition[df_malnutrition['Mean_Estimate'] < 0])

# Distribution of 'Mean_Estimate'
plt.figure(figsize=(8, 6))
sns.histplot(df_obesity['Mean_Estimate'], kde=True)
plt.title('Distribution of Mean Estimate')
plt.xlabel('Mean Estimate')
plt.ylabel('Frequency')
plt.show()

# Distribution of 'CI_Width'
plt.figure(figsize=(8, 6))
sns.histplot(df_obesity['CI_Width'], kde=True, color='orange')
plt.title('Distribution of Confidence Interval Width')
plt.xlabel('Confidence Interval Width')
plt.ylabel('Frequency')
plt.show()

# Distribution of 'Mean_Estimate'
plt.figure(figsize=(8, 6))
sns.histplot(df_malnutrition['Mean_Estimate'], kde=True)
plt.title('Distribution of Mean Estimate')
plt.xlabel('Mean Estimate')
plt.ylabel('Frequency')
plt.show()

# Distribution of 'CI_Width'
plt.figure(figsize=(8, 6))
sns.histplot(df_malnutrition['CI_Width'], kde=True, color='orange')
plt.title('Distribution of Confidence Interval Width')
plt.xlabel('Confidence Interval Width')
plt.ylabel('Frequency')
plt.show()

#Trends of Mean Estimate Over Time by Region
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Mean_Estimate', data=df_obesity, hue='Region')
plt.title('Trends of Mean Estimate Over Time by Region')
plt.xlabel('Year')
plt.ylabel('Mean Estimate')
plt.show()

#Trends of Mean Estimate Over Time by Region
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Mean_Estimate', data=df_malnutrition, hue='Region')
plt.title('Trends of Mean Estimate Over Time by Region')
plt.xlabel('Year')
plt.ylabel('Mean Estimate')
plt.show()

# Box plot for Mean Estimate across Regions
plt.figure(figsize=(12, 6))
sns.boxplot(x='Region', y='Mean_Estimate', data=df_obesity)
plt.title('Distribution of Mean Estimate of obesity by Region')
plt.xlabel('Region')
plt.ylabel('Mean Estimate')
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(x='Region', y='Mean_Estimate', data=df_malnutrition)
plt.title('Distribution of Mean Estimate of malnutrition by Region')
plt.xlabel('Region')
plt.ylabel('Mean Estimate')
plt.xticks(rotation=90)
plt.show()


# Box plot for Mean Estimate across Age Groups
plt.figure(figsize=(8, 6))
sns.boxplot(x='age_group', y='Mean_Estimate', data=df_obesity)
plt.title('Distribution of Mean Estimate of obesity by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Mean Estimate')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='age_group', y='Mean_Estimate', data=df_malnutrition)
plt.title('Distribution of Mean Estimate  of malnutrition by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Mean Estimate')
plt.show()

# Bar plot for Obesity Levels by Region
plt.figure(figsize=(10, 6))
sns.countplot(x='Region', hue='obesity_level', data=df_obesity)
plt.title('Obesity Levels by Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Region', hue='malnutrition_level', data=df_malnutrition)
plt.title('malnutrition Levels by Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

# Correlation heatmap for numeric columns
correlation_matrix = df_obesity[['Mean_Estimate', 'CI_Width']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap')
plt.show()

correlation_matrix = df_malnutrition[['Mean_Estimate', 'CI_Width']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap')
plt.show()

# Interactive scatter plot for Mean Estimate vs CI Width
fig = px.scatter(df_obesity, x='Mean_Estimate', y='CI_Width', color='Region', title='Mean Estimate vs CI Width')
fig.show()

fig = px.scatter(df_malnutrition, x='Mean_Estimate', y='CI_Width', color='Region', title='Mean Estimate vs CI Width')
fig.show()

pip install mysql-connector-python

connection = db.connect(
    host = 'localhost',
    user = 'root',
    password ='Shaffie0000' ,
    database = 'healthdata'
)

curr = connection.cursor()

curr.execute(
    """
    CREATE TABLE obesity (
    id INT AUTO_INCREMENT PRIMARY KEY,
    Region VARCHAR(255),
    Country VARCHAR(255),
    Year INT,
    Gender VARCHAR(50),
    LowerBound FLOAT,
    UpperBound FLOAT,
    Mean_Estimate FLOAT,
    CI_Width FLOAT,
    age_group VARCHAR(50),
    obesity_level VARCHAR(50)
    );
    
    """
)

# SQL Insert Query
insert_query = """
    INSERT INTO obesity (Region, Country, Year, Gender, LowerBound, UpperBound, Mean_Estimate, CI_Width, age_group, obesity_level) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

# Counter for inserted rows
inserted_rows = 0

# Iterate over DataFrame rows
try:
    for index, row in df_obesity.iterrows():
        # Convert each row to a tuple
        values = (
            row["Region"],
            row["Country"],
            int(row["Year"]),
            row["Gender"],
            float(row["LowerBound"]),
            float(row["UpperBound"]),
            float(row["Mean_Estimate"]),
            float(row["CI_Width"]),
            row["age_group"],
            row["obesity_level"]
        )
        
        # Execute insertion
        curr.execute(insert_query, values)
        inserted_rows += 1

    # Commit the transaction
    connection.commit()
    print(f"Inserted {inserted_rows} records successfully.")

except Exception as e:
    # Rollback in case of error
    if 'connection' in locals():
        connection.rollback()
    print(f"Error occurred: {e}")

curr.execute(
    """
    CREATE TABLE malnutrition (
    id INT AUTO_INCREMENT PRIMARY KEY,
    Region VARCHAR(255),
    Country VARCHAR(255),
    Year INT,
    Gender VARCHAR(50),
    LowerBound FLOAT,
    UpperBound FLOAT,
    Mean_Estimate FLOAT,
    CI_Width FLOAT,
    age_group VARCHAR(50),
    Malnutrition_Level VARCHAR(50)
    );
    
    """
)

# SQL Insert Query
insert_query = """
    INSERT INTO malnutrition (Region, Country, Year, Gender, LowerBound, UpperBound, Mean_Estimate, CI_Width, age_group, Malnutrition_Level) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

# Counter for inserted rows
inserted_rows = 0

# Iterate over DataFrame rows
try:
    for index, row in df_malnutrition.iterrows():
        # Convert each row to a tuple
        values = (
            row["Region"],
            row["Country"],
            int(row["Year"]),
            row["Gender"],
            float(row["LowerBound"]),
            float(row["UpperBound"]),
            float(row["Mean_Estimate"]),
            float(row["CI_Width"]),
            row["age_group"],
            row["malnutrition_level"]
        )
        
        # Execute insertion
        curr.execute(insert_query, values)
        inserted_rows += 1

    # Commit the transaction
    connection.commit()
    print(f"Inserted {inserted_rows} records successfully.")

except Exception as e:
    # Rollback in case of error
    if 'connection' in locals():
        connection.rollback()
    print(f"Error occurred: {e}")
    
#queries

#1 Top 5 regions with the highest average obesity levels in the most recent year(2022)
query = """
    SELECT Region, AVG(Mean_Estimate) AS Avg_Obesity_Level
    FROM obesity
    WHERE Year = 2022
    GROUP BY Region
    ORDER BY Avg_Obesity_Level DESC
    LIMIT 5;
"""

# Execute the query
curr.execute(query)
result = curr.fetchall()

# Display the results
print("Top 5 Regions with Highest Average Obesity Levels in 2022:")
for row in result:
    print(f"Region: {row[0]}, Average Obesity Level: {row[1]:.2f}")
    
#2 Top 5 countries with highest obesity estimates

query = """
    SELECT Country, AVG(Mean_Estimate) AS Avg_Obesity_Level
    FROM obesity
    GROUP BY Country
    ORDER BY Avg_Obesity_Level DESC
    LIMIT 5;
"""

curr.execute(query)
result = curr.fetchall()

print("Top 5 Countries with Highest Obesity Estimates:")
for row in result:
    print(f"Country: {row[0]}, Average Obesity Estimate: {row[1]:.2f}")

#3 Obesity trend in India over the years(Mean_estimate)

query = """
    SELECT Year, AVG(Mean_Estimate) AS Avg_Obesity_Level
    FROM obesity
    WHERE Country = 'India'
    GROUP BY Year
    ORDER BY Year ASC;
"""

curr.execute(query)
result = curr.fetchall()

df_trend = pd.DataFrame(result, columns=["Year", "Avg_Obesity_Level"])

plt.figure(figsize=(10, 6))
plt.plot(df_trend["Year"], df_trend["Avg_Obesity_Level"], marker='o', linestyle='-', color='blue')
plt.title("Obesity Trend in India (Mean Estimate) Over the Years")
plt.xlabel("Year")
plt.ylabel("Average Obesity Estimate")
plt.grid(alpha=0.3)
plt.xticks(df_trend["Year"])
plt.tight_layout()
plt.show()

#4 Average obesity by gender

query = """
    SELECT Gender, AVG(Mean_Estimate) AS Avg_Obesity_Level
    FROM obesity
    GROUP BY Gender;
"""

curr.execute(query)
result = curr.fetchall()

df_gender = pd.DataFrame(result, columns=["Gender", "Avg_Obesity_Level"])


plt.figure(figsize=(8, 6))
plt.bar(df_gender["Gender"], df_gender["Avg_Obesity_Level"], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title("Average Obesity by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Obesity Estimate")
plt.grid(alpha=0.3, linestyle='--', axis='y')
plt.tight_layout()
plt.show()

#5 Country count by obesity level category and age group

query = """
    SELECT obesity_level, age_group, COUNT(DISTINCT Country) AS Country_Count
    FROM obesity
    GROUP BY obesity_level, age_group;
"""

curr.execute(query)
result = curr.fetchall()

df_count = pd.DataFrame(result, columns=["Obesity_Level", "Age_Group", "Country_Count"])

pivot_data = df_count.pivot(index="Age_Group", columns="Obesity_Level", values="Country_Count")
print(pivot_data)


# Plotting
plt.figure(figsize=(12, 8))
obesity_levels = pivot_data.columns
bottom = np.zeros(len(pivot_data))

for level in obesity_levels:
    plt.bar(pivot_data.index, pivot_data[level], bottom=bottom, label=level)
    bottom += pivot_data[level].fillna(0)

plt.xlabel("Age Group")
plt.ylabel("Country Count")
plt.title("Country Count by Obesity Level and Age Group (Stacked Bar Chart)")
plt.xticks(rotation=45)
plt.legend(title="Obesity Level")
plt.tight_layout()
plt.show()

#6 Top 5 countries least reliable countries(with highest CI_Width) and Top 5 most consistent countries (smallest average CI_Width)

query_least_reliable = """
    SELECT Country, AVG(CI_Width) AS Avg_CI_Width
    FROM obesity
    GROUP BY Country
    ORDER BY Avg_CI_Width DESC
    LIMIT 5;
"""

curr.execute(query_least_reliable)
least_reliable = curr.fetchall()

# Query for most consistent countries (lowest CI_Width)
query_most_consistent = """
    SELECT Country, AVG(CI_Width) AS Avg_CI_Width
    FROM obesity
    GROUP BY Country
    ORDER BY Avg_CI_Width ASC
    LIMIT 5;
"""

curr.execute(query_most_consistent)
most_consistent = curr.fetchall()

df_least_reliable = pd.DataFrame(least_reliable, columns=["Country", "Avg_CI_Width"])
df_most_consistent = pd.DataFrame(most_consistent, columns=["Country", "Avg_CI_Width"])

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot Least Reliable Countries
axs[0].barh(df_least_reliable["Country"], df_least_reliable["Avg_CI_Width"], color='red')
axs[0].set_title("Top 5 Least Reliable Countries (Highest CI_Width)")
axs[0].set_xlabel("Average CI_Width")
axs[0].invert_yaxis()

# Plot Most Consistent Countries
axs[1].barh(df_most_consistent["Country"], df_most_consistent["Avg_CI_Width"], color='green')
axs[1].set_title("Top 5 Most Consistent Countries (Lowest CI_Width)")
axs[1].set_xlabel("Average CI_Width")
axs[1].invert_yaxis()

plt.tight_layout()
plt.show()

#7 Average obesity by age group

query = """
    SELECT age_group, AVG(Mean_Estimate) AS Avg_Obesity_Level
    FROM obesity
    GROUP BY age_group;
"""

curr.execute(query)
result = curr.fetchall()

df_age_group = pd.DataFrame(result, columns=["Age_Group", "Avg_Obesity_Level"])

plt.figure(figsize=(10, 6))
plt.bar(df_age_group["Age_Group"], df_age_group["Avg_Obesity_Level"], color=['#1f77b4', '#ff7f0e'])
plt.title("Average Obesity by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Average Obesity Level")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#8 Top 10 Countries with consistent low obesity (low average + low CI)over the years

query = """
    SELECT Country, 
           AVG(Mean_Estimate) AS Avg_Obesity, 
           AVG(CI_Width) AS Avg_CI_Width
    FROM obesity
    GROUP BY Country
    HAVING AVG(Mean_Estimate) < 25 AND AVG(CI_Width) < 5
    ORDER BY Avg_Obesity ASC, Avg_CI_Width ASC
    LIMIT 10;
"""
curr.execute(query)
result = curr.fetchall()

df_consistent = pd.DataFrame(result, columns=["Country", "Avg_Obesity", "Avg_CI_Width"])

plt.figure(figsize=(12, 8))
width = 0.4  # Width of the bars

x = range(len(df_consistent["Country"]))

# Bar for Avg Obesity
plt.bar(x, df_consistent["Avg_Obesity"], width=width, color="#1f77b4", label="Avg Obesity")

# Bar for Avg CI Width (shifted by width for grouping effect)
plt.bar([i + width for i in x], df_consistent["Avg_CI_Width"], width=width, color="#ff7f0e", label="Avg CI Width")

plt.xlabel("Country")
plt.ylabel("Values")
plt.title("Top 10 Countries with Consistent Low Obesity and CI Width")
plt.xticks([i + width / 2 for i in x], df_consistent["Country"], rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.grid(alpha=0.3, linestyle="--", axis="y")
plt.show()

#9 Countries where female obesity exceeds male by large margin (same year)
query = """
    SELECT 
        o1.Country,
        AVG(o1.Mean_Estimate - o2.Mean_Estimate) AS Avg_Obesity_Diff
    FROM 
        obesity AS o1
    INNER JOIN 
        obesity AS o2 
        ON o1.Country = o2.Country AND o1.Year = o2.Year
    WHERE 
        o1.Gender = 'Female' AND 
        o2.Gender = 'Male'
    GROUP BY 
        o1.Country
    HAVING 
        Avg_Obesity_Diff > 1
    ORDER BY 
        Avg_Obesity_Diff DESC
    LIMIT 10;
"""

curr.execute(query)
result = curr.fetchall()

df_diff = pd.DataFrame(result, columns=["Country", "Avg_Obesity_Diff"])
print(df_diff)
# Plotting - Horizontal Bar Chart
plt.figure(figsize=(12, 8))
plt.barh(df_diff["Country"], df_diff["Avg_Obesity_Diff"], color="#ff7f0e", alpha=0.7)
plt.xlabel("Average Obesity Difference (Female - Male)")
plt.title("Top 10 Countries Where Female Obesity Exceeds Male by a Large Margin (Average Over Years)")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#10 Global average obesity percentage per year

query = """
    SELECT 
        Year,
        AVG(Mean_Estimate) AS Global_Avg_Obesity
    FROM obesity
    GROUP BY Year
    ORDER BY Year;
"""

curr.execute(query)
result = curr.fetchall()

df_global_avg = pd.DataFrame(result, columns=["Year", "Global_Avg_Obesity"])
print(df_global_avg )
# Plotting
plt.figure(figsize=(12, 8))
plt.plot(df_global_avg["Year"], df_global_avg["Global_Avg_Obesity"], marker='o', linestyle='-', color="#1f77b4")
plt.title("Global Average Obesity Percentage Per Year")
plt.xlabel("Year")
plt.ylabel("Global Average Obesity (%)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(df_global_avg["Year"], rotation=45)
plt.tight_layout()
plt.show()

#11  Avg. malnutrition by age group

query = """
    SELECT 
        age_group,
        AVG(Mean_Estimate) AS Avg_Malnutrition
    FROM malnutrition
    GROUP BY age_group
    ORDER BY Avg_Malnutrition DESC;
"""

curr.execute(query)
result = curr.fetchall()

df_malnutrition = pd.DataFrame(result, columns=["Age_Group", "Avg_Malnutrition"])

plt.figure(figsize=(3,6))
plt.bar(df_malnutrition["Age_Group"], df_malnutrition["Avg_Malnutrition"], color="#ff7f0e")
plt.xlabel("Age Group")
plt.ylabel("Average Malnutrition (%)")
plt.title("Average Malnutrition by Age Group")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#12 Top 5 countries with highest malnutrition(mean_estimate)

query = """
    SELECT 
        Country,
        AVG(Mean_Estimate) AS Avg_Malnutrition
    FROM malnutrition
    GROUP BY Country
    ORDER BY Avg_Malnutrition DESC
    LIMIT 5;
"""

curr.execute(query)
result = curr.fetchall()

df_top_malnutrition = pd.DataFrame(result, columns=["Country", "Avg_Malnutrition"])
print(df_top_malnutrition)
plt.figure(figsize=(12,4))
plt.barh(df_top_malnutrition["Country"], df_top_malnutrition["Avg_Malnutrition"], color="#d62728")
plt.xlabel("Average Malnutrition (%)")
plt.title("Top 5 Countries with Highest Malnutrition")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.gca().invert_yaxis()  # Invert y-axis for descending order
plt.tight_layout()
plt.show()

#13 Malnutrition trend in African region over the years

query = """
    SELECT 
        Year,
        AVG(Mean_Estimate) AS Avg_Malnutrition
    FROM malnutrition
    WHERE Region = 'Africa'
    GROUP BY Year
    ORDER BY Year;
"""
curr.execute(query)
result = curr.fetchall()


df_africa_trend = pd.DataFrame(result, columns=["Year", "Avg_Malnutrition"])

plt.figure(figsize=(12, 8))
plt.plot(df_africa_trend["Year"], df_africa_trend["Avg_Malnutrition"], marker='o', linestyle='-', color="#d62728")
plt.xlabel("Year")
plt.ylabel("Average Malnutrition (%)")
plt.title("Malnutrition Trend in Africa Over the Years")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(df_africa_trend["Year"], rotation=45)
plt.tight_layout()
plt.show()

#14  Gender-based average malnutrition
query = """
    SELECT 
        Gender,
        AVG(Mean_Estimate) AS Avg_Malnutrition
    FROM malnutrition
    GROUP BY Gender
    ORDER BY Avg_Malnutrition DESC;
"""

# Execute the query
curr.execute(query)
result = curr.fetchall()

df_gender_malnutrition = pd.DataFrame(result, columns=["Gender", "Avg_Malnutrition"])

plt.figure(figsize=(5, 6))
plt.bar(df_gender_malnutrition["Gender"], df_gender_malnutrition["Avg_Malnutrition"], color="#1f77b4")
plt.xlabel("Gender")
plt.ylabel("Average Malnutrition (%)")
plt.title("Gender-Based Average Malnutrition")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#15 Malnutrition level-wise (average CI_Width by age group)

query = """
    SELECT 
        malnutrition_level,
        age_group,
        AVG(CI_Width) AS Avg_CI_Width
    FROM malnutrition
    GROUP BY malnutrition_level, age_group
    ORDER BY malnutrition_level, Avg_CI_Width DESC;
"""
curr.execute(query)
result = curr.fetchall()

df_malnutrition_ci = pd.DataFrame(result, columns=["Malnutrition_Level", "Age_Group", "Avg_CI_Width"])
print(df_malnutrition_ci)

plt.figure(figsize=(10, 8))

for level in df_malnutrition_ci["Malnutrition_Level"].unique():
    subset = df_malnutrition_ci[df_malnutrition_ci["Malnutrition_Level"] == level]
    plt.plot(subset["Age_Group"], subset["Avg_CI_Width"], marker='o', linestyle='-', label=level)

plt.xlabel("Age Group")
plt.ylabel("Average CI_Width")
plt.title("Average CI_Width by Malnutrition Level and Age Group")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.legend(title="Malnutrition Level", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#16 Yearly malnutrition change in specific countries(India, Nigeria, Brazil)

query = """
    SELECT 
        Year,
        Country,
        AVG(Mean_Estimate) AS Avg_Malnutrition
    FROM malnutrition
    WHERE Country IN ('India', 'Nigeria', 'Brazil')
    GROUP BY Year, Country
    ORDER BY Year, Country;
"""
curr.execute(query)
result = curr.fetchall()

df_malnutrition_change = pd.DataFrame(result, columns=["Year", "Country", "Avg_Malnutrition"])
print(df_malnutrition_change )
plt.figure(figsize=(14, 8))

for country in df_malnutrition_change["Country"].unique():
    subset = df_malnutrition_change[df_malnutrition_change["Country"] == country]
    plt.plot(subset["Year"], subset["Avg_Malnutrition"], marker='o', linestyle='-', label=country)

plt.xlabel("Year")
plt.ylabel("Average Malnutrition (%)")
plt.title("Yearly Malnutrition Change in India, Nigeria, and Brazil")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(sorted(df_malnutrition_change["Year"].unique()), rotation=45)
plt.legend(title="Country")
plt.tight_layout()
plt.show()

#17 Regions with lowest malnutrition averages
query = """
    SELECT 
        Region,
        AVG(Mean_Estimate) AS Avg_Malnutrition
    FROM malnutrition
    GROUP BY Region
    ORDER BY Avg_Malnutrition ASC
    LIMIT 6;
"""

curr.execute(query)
result = curr.fetchall()

df_low_malnutrition = pd.DataFrame(result, columns=["Region", "Avg_Malnutrition"])

df_low_malnutrition["Region"].fillna("Unknown", inplace=True)
print(df_low_malnutrition)
plt.figure(figsize=(12, 8))
plt.barh(df_low_malnutrition["Region"], df_low_malnutrition["Avg_Malnutrition"], color="#2ca02c")
plt.xlabel("Average Malnutrition (%)")
plt.title("Top 5 Regions with Lowest Malnutrition Averages")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.gca().invert_yaxis()  # Invert y-axis for descending order
plt.tight_layout()
plt.show()

#18 Countries with increasing malnutrition 

query = """
    SELECT 
        Country,
        MIN(Year) AS Earliest_Year,
        MAX(Year) AS Latest_Year,
        MIN(Mean_Estimate) AS Min_Estimate,
        MAX(Mean_Estimate) AS Max_Estimate,
        (MAX(Mean_Estimate) - MIN(Mean_Estimate)) AS Malnutrition_Change
    FROM malnutrition
    GROUP BY Country
    HAVING Malnutrition_Change > 0
    ORDER BY Malnutrition_Change DESC
    LIMIT 10;
"""

curr.execute(query)
result = curr.fetchall()

df_increasing_malnutrition = pd.DataFrame(result, columns=["Country", "Earliest_Year", "Latest_Year", "Min_Estimate", "Max_Estimate", "Malnutrition_Change"])

plt.figure(figsize=(14, 8))
plt.barh(df_increasing_malnutrition["Country"], df_increasing_malnutrition["Malnutrition_Change"], color="#d62728")
plt.xlabel("Increase in Malnutrition (%)")
plt.title("Top 10 Countries with Increasing Malnutrition Levels")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#19  Min/Max malnutrition levels year-wise comparison


query = """
    SELECT 
        Year,
        MIN(Mean_Estimate) AS Min_Malnutrition,
        MAX(Mean_Estimate) AS Max_Malnutrition
    FROM malnutrition
    GROUP BY Year
    ORDER BY Year ASC;
"""

curr.execute(query)
result = curr.fetchall()

df_malnutrition_yearwise = pd.DataFrame(result, columns=["Year", "Min_Malnutrition", "Max_Malnutrition"])

plt.figure(figsize=(14, 8))
plt.plot(df_malnutrition_yearwise["Year"], df_malnutrition_yearwise["Min_Malnutrition"], marker='o', linestyle='-', label="Min Malnutrition", color="#1f77b4")
plt.plot(df_malnutrition_yearwise["Year"], df_malnutrition_yearwise["Max_Malnutrition"], marker='o', linestyle='-', label="Max Malnutrition", color="#ff7f0e")
plt.xlabel("Year")
plt.ylabel("Malnutrition Level (%)")
plt.title("Year-Wise Min/Max Malnutrition Levels")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#20  High CI_Width flags for monitoring(CI_width > 5)
query = """
    SELECT DISTINCT Region, Country, Year, Gender, CI_Width
    FROM obesity
    WHERE CI_Width > 5;
"""

curr.execute(query)
result = curr.fetchall()

columns = ["Region", "Country", "Year", "Gender", "CI_Width"]
df_high_ci_flags = pd.DataFrame(result, columns=columns)

plt.figure(figsize=(12, 6))
df_high_ci_flags.sort_values("CI_Width", ascending=False).head(10).plot(
    kind="barh",
    x="Country",
    y="CI_Width",
    color="#ff7f0e",
    legend=False,
    title="Top 10 Countries with High CI_Width (> 5)",
    xlabel="CI_Width",
    ylabel="Country"
)
plt.tight_layout()
plt.show()

#21  Obesity vs malnutrition comparison by country(any 5 countries)

query = """
    SELECT 
        o.Country,
        o.Year,
        o.Mean_Estimate AS Obesity_Estimate,
        m.Mean_Estimate AS Malnutrition_Estimate
    FROM 
        obesity o
    JOIN 
        malnutrition m ON o.Country = m.Country AND o.Year = m.Year
    WHERE 
        o.Country IN ('India', 'Nigeria', 'Brazil', 'USA', 'China')
    ORDER BY 
        o.Country, o.Year;
"""

curr.execute(query)
result = curr.fetchall()

columns = ["Country", "Year", "Obesity_Estimate", "Malnutrition_Estimate"]
df_comparison = pd.DataFrame(result, columns=columns)

plt.figure(figsize=(10, 6))

for country in df_comparison['Country'].unique():
    country_data = df_comparison[df_comparison['Country'] == country]
    
    plt.scatter(country_data['Obesity_Estimate'], country_data['Malnutrition_Estimate'],
                label=country, s=100, alpha=0.6)

plt.xlabel('Obesity Estimate (%)', fontsize=14)
plt.ylabel('Malnutrition Estimate (%)', fontsize=14)
plt.title('Obesity vs Malnutrition Comparison by Country', fontsize=16)
plt.legend(title="Country", loc='upper left', fontsize=12)

plt.grid(True)
plt.tight_layout()
plt.show()

#22 Gender-based disparity in both obesity and malnutrition

query = """
    SELECT 
        o.Country,
        o.Gender,
        AVG(o.Mean_Estimate) AS Avg_Obesity,
        AVG(m.Mean_Estimate) AS Avg_Malnutrition
    FROM 
        obesity o
    JOIN 
        malnutrition m ON o.Country = m.Country AND o.Year = m.Year AND o.Gender = m.Gender
    GROUP BY 
        o.Country, o.Gender
    ORDER BY 
        o.Country, o.Gender;
"""

curr.execute(query)
result = curr.fetchall()

columns = ["Country", "Gender", "Avg_Obesity", "Avg_Malnutrition"]
df_gender_disparity = pd.DataFrame(result, columns=columns)

top_countries = df_gender_disparity["Country"].unique()[:5]
df_filtered = df_gender_disparity[df_gender_disparity["Country"].isin(top_countries)]


df_filtered_melted = df_filtered.melt(
    id_vars=["Country", "Gender"],
    value_vars=["Avg_Obesity", "Avg_Malnutrition"],
    var_name="Indicator",
    value_name="Mean_Estimate"
)

# Plotting using Seaborn
g = sns.catplot(
    data=df_filtered_melted,
    x="Gender",
    y="Mean_Estimate",
    hue="Indicator",
    col="Country",
    kind="bar",
    height=5,
    aspect=0.8,
    palette=["#1f77b4", "#ff7f0e"]
)

g.set_titles("{col_name}")
g.set_axis_labels("Gender", "Mean Estimate (%)")
g.add_legend(title="Indicator")
plt.suptitle("Gender-Based Disparity in Obesity and Malnutrition (Top 5 Countries)", y=1.05)
plt.tight_layout()
plt.show()

#22 Gender-based disparity in both obesity and malnutrition

query = """
    SELECT 
        o.Country,
        o.Gender,
        AVG(o.Mean_Estimate) AS Avg_Obesity,
        AVG(m.Mean_Estimate) AS Avg_Malnutrition
    FROM 
        obesity o
    JOIN 
        malnutrition m ON o.Country = m.Country AND o.Year = m.Year AND o.Gender = m.Gender
    GROUP BY 
        o.Country, o.Gender
    ORDER BY 
        o.Country, o.Gender;
"""

curr.execute(query)
result = curr.fetchall()

columns = ["Country", "Gender", "Avg_Obesity", "Avg_Malnutrition"]
df_gender_disparity = pd.DataFrame(result, columns=columns)

top_countries = df_gender_disparity["Country"].unique()[:5]
df_filtered = df_gender_disparity[df_gender_disparity["Country"].isin(top_countries)]


df_filtered_melted = df_filtered.melt(
    id_vars=["Country", "Gender"],
    value_vars=["Avg_Obesity", "Avg_Malnutrition"],
    var_name="Indicator",
    value_name="Mean_Estimate"
)

# Plotting using Seaborn
g = sns.catplot(
    data=df_filtered_melted,
    x="Gender",
    y="Mean_Estimate",
    hue="Indicator",
    col="Country",
    kind="bar",
    height=5,
    aspect=0.8,
    palette=["#1f77b4", "#ff7f0e"]
)

g.set_titles("{col_name}")
g.set_axis_labels("Gender", "Mean Estimate (%)")
g.add_legend(title="Indicator")
plt.suptitle("Gender-Based Disparity in Obesity and Malnutrition (Top 5 Countries)", y=1.05)
plt.tight_layout()
plt.show()

#23 Region-wise avg estimates side-by-side(Africa and America)
query = """
    SELECT 
        o.Region,
        o.Gender,
        AVG(o.Mean_Estimate) AS Avg_Obesity,
        AVG(m.Mean_Estimate) AS Avg_Malnutrition
    FROM 
        obesity o
    JOIN 
        malnutrition m ON o.Country = m.Country AND o.Year = m.Year AND o.Gender = m.Gender
    WHERE 
        o.Region IN ('Africa', 'Americas')
    GROUP BY 
        o.Region, o.Gender;
"""

curr.execute(query)
result = curr.fetchall()

columns = ["Region", "Gender", "Avg_Obesity", "Avg_Malnutrition"]
df_region_comparison = pd.DataFrame(result, columns=columns)

regions = ["Africa", "Americas"]
genders = ["Male", "Female"]

import seaborn as sns
# Side-by-side plots for Obesity and Malnutrition
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

# Plot for Obesity
sns.barplot(
    data=df_region_comparison,
    x="Gender",
    y="Avg_Obesity",
    hue="Region",
    palette="Blues",
    ci=None,
    ax=axes[0]
)
axes[0].set_title("Average Obesity by Gender in Africa and Americas")
axes[0].set_ylabel("Mean Estimate (%)")
axes[0].grid(axis="y", linestyle="--", alpha=0.5)

# Plot for Malnutrition
sns.barplot(
    data=df_region_comparison,
    x="Gender",
    y="Avg_Malnutrition",
    hue="Region",
    palette="Oranges",
    ci=None,
    ax=axes[1]
)
axes[1].set_title("Average Malnutrition by Gender in Africa and Americas")
axes[1].set_ylabel("")  # Remove y-label to avoid repetition
axes[1].grid(axis="y", linestyle="--", alpha=0.5)

# Adjust layout and show the plots
plt.tight_layout()
plt.show()

#24 Countries with obesity up & malnutrition down
query = """
SELECT 
    o.Country,
    (MAX(o.Mean_Estimate) - MIN(o.Mean_Estimate)) AS Obesity_Change,
    (MIN(m.Mean_Estimate) - MAX(m.Mean_Estimate)) AS Malnutrition_Change
FROM 
    obesity o
INNER JOIN 
    malnutrition m ON o.Country = m.Country AND o.Year = m.Year
GROUP BY 
    o.Country
LIMIT 10;
"""

curr.execute(query)
data = curr.fetchall()

# Creating DataFrame
df_change = pd.DataFrame(data, columns=["Country", "Obesity_Change", "Malnutrition_Change"])

# Check if data is fetched
print(df_change.head())

# Plotting
plt.figure(figsize=(14, 8))
width = 0.35
x = range(len(df_change))

plt.bar([pos - width/2 for pos in x], df_change["Obesity_Change"], width=width, label="Obesity Increase", color="#1f77b4")
plt.bar([pos + width/2 for pos in x], df_change["Malnutrition_Change"], width=width, label="Malnutrition Decrease", color="#ff7f0e")

plt.xticks(x, df_change["Country"], rotation=45, ha="right")
plt.xlabel("Country")
plt.ylabel("Change in Levels")
plt.title("Countries with Increasing Obesity and Decreasing Malnutrition")
plt.axhline(0, color='grey', linewidth=0.5, linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

#25 Age-wise trend analysis


# Query obesity trend
curr.execute("""
SELECT
    Year,
    age_group,
    AVG(Mean_Estimate) AS Avg_Obesity
FROM obesity
GROUP BY Year, age_group
ORDER BY Year, age_group;
""")
obesity_data = curr.fetchall()

df_obesity = pd.DataFrame(obesity_data, columns=["Year", "Age_Group", "Avg_Obesity"])

# Query malnutrition trend
curr.execute("""
SELECT
    Year,
    age_group,
    AVG(Mean_Estimate) AS Avg_Malnutrition
FROM malnutrition
GROUP BY Year, age_group
ORDER BY Year, age_group;
""")
malnutrition_data = curr.fetchall()

df_malnutrition = pd.DataFrame(malnutrition_data, columns=["Year", "Age_Group", "Avg_Malnutrition"])

plt.figure(figsize=(14, 6))
print(df_malnutrition)
# Obesity trend
plt.subplot(1, 2, 1)
sns.lineplot(data=df_obesity, x="Year", y="Avg_Obesity", hue="Age_Group", marker="o")
plt.title("Age-wise Obesity Trend Over Years")
plt.ylabel("Average Obesity (%)")
plt.xlabel("Year")
plt.legend(title="Age Group")

# Malnutrition trend
plt.subplot(1, 2, 2)
sns.lineplot(data=df_malnutrition, x="Year", y="Avg_Malnutrition", hue="Age_Group", marker="o")
plt.title("Age-wise Malnutrition Trend Over Years")
plt.ylabel("Average Malnutrition (%)")
plt.xlabel("Year")
plt.legend(title="Age Group")

plt.tight_layout()
plt.show()

curr.close()
connection.close()