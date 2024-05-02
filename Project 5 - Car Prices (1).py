#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


inpPath = 'C:\\Users\\ashyc\\Downloads\\'
data = pd.read_csv(inpPath + 'car_prices.csv', delimiter=',')
data


# In[8]:


# Checking for missing values and removing any rows with missing data
data_cleaned = data.dropna()


# In[9]:


# Remove duplicates
data_cleaned = data_cleaned.drop_duplicates()


# In[10]:


data_cleaned


# In[11]:


# Convert year to datetime format
data_cleaned['year'] = pd.to_datetime(data_cleaned['year'], format='%Y')


# In[16]:


data_cleaned['sellingprice'] = pd.to_numeric(data_cleaned['sellingprice'], errors='coerce')


# In[18]:


# Creating a Histogram of car prices
plt.figure(figsize=(10, 6))
sns.histplot(data_cleaned['sellingprice'], bins=30, color='green')
plt.title('Distribution of Car Prices')
plt.xlabel('Price ($)')
plt.ylabel('Number of Cars')
plt.show()


# In[20]:


# Scatter plot of mileage vs selling price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='odometer', y='sellingprice', data=data_cleaned)
plt.title('Mileage vs. Price')
plt.xlabel('Mileage')
plt.ylabel('Price ($)')
plt.show()


# In[22]:


# Bar chart of average price by make
avg_price_by_make = data_cleaned.groupby('make')['sellingprice'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 7))
avg_price_by_make.plot(kind='bar', color='orange')
plt.title('Average Price by Car Make')
plt.xlabel('Make')
plt.ylabel('Average Price ($)')
plt.xticks(rotation=45)
plt.show()


# In[25]:


# Removing outliers for price for more accurate analysis
Q1 = data_cleaned['sellingprice'].quantile(0.25)
Q3 = data_cleaned['sellingprice'].quantile(0.75)
IQR = Q3 - Q1
data_filtered = data_cleaned[(data_cleaned['sellingprice'] >= Q1 - 1.5 * IQR) & (data_cleaned['sellingprice'] <= Q3 + 1.5 * IQR)]


# In[26]:


# Visualization of price distribution once outliers are removed
plt.figure(figsize=(10, 6))
sns.histplot(data_filtered['sellingprice'], bins=30, kde=True, color='blue')
plt.title('Filtered Distribution of Car Prices')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.show()


# In[27]:


# Scatter plot to show the relationship between mileage and price once outliers are removed
plt.figure(figsize=(10, 6))
sns.scatterplot(x='odometer', y='sellingprice', data=data_filtered)
plt.title('Price vs. Mileage')
plt.xlabel('Mileage')
plt.ylabel('Price ($)')
plt.show()


# In[29]:


# Boxplot to examine prices across different car makes
plt.figure(figsize=(12, 8))
sns.boxplot(x='make', y='sellingprice', data=data_filtered)
plt.title('Price Distribution by Car Make')
plt.xlabel('Car Make')
plt.ylabel('Price ($)')
plt.xticks(rotation=45)  
# Rotates the car make labels for better readability
plt.show()


# In[30]:


# Bar chart to compare average prices by year
avg_price_by_year = data_filtered.groupby('year')['sellingprice'].mean()
plt.figure(figsize=(12, 8))
avg_price_by_year.plot(kind='bar', color='green')
plt.title('Average Car Prices by Year')
plt.xlabel('Year')
plt.ylabel('Average Price ($)')
plt.xticks(rotation=45)
plt.show()


# In[31]:


# Correlation Heatmap of numerical features
numerical_features = ['sellingprice', 'odometer', 'year']  
# assume mileage is another numerical column
correlation_matrix = data_cleaned[numerical_features].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap Among Numerical Features')
plt.show()

