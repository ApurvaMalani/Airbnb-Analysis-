#importing important packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from collections import Counter

import warnings
warnings.filterwarnings('ignore')
from scipy.stats import zscore

from IPython.core.display import HTML
from datascience import *
from mpl_toolkits.basemap import Basemap

#reading the dataset
df=pd.read_csv("sample_airbnb.csv")
df.head(4)

df.shape

df.columns

columns = ['price',
           'summary',
           'neighborhood_overview',
           'property_type',
           'room_type',
           'price',
           'number_of_reviews',
           'beds',
           'bedrooms',
           'room_type',
           'minimum_nights',
           'maximum_nights',
           'security_deposit',
           'cleaning_fee',
           'extra_people',
           'guests_included',
           'bathrooms',
           'accommodates',
           'cancellation_policy',
           'address.location.coordinates[0]',
           'address.location.coordinates[1]',
           'reviews_per_month']
airbnb = pd.read_csv(listings_file, usecols=columns)
# address.location.coordinates[0] is longitude
# address.location.coordinates[1] is lat

DATA CLEANING

airbnb.describe()

airbnb.isnull().sum()

# replacing NaN values with 0
airbnb.fillna(0, inplace=True)

airbnb.property_type.unique()

airbnb.cancellation_policy.unique()

# airbnb['cancellation_policy']=airbnb['cancellation_policy'].replace('super_strict_30','super_strict')
# airbnb['cancellation_policy']=airbnb['cancellation_policy'].replace('super_strict_60','super_strict')
# airbnb['cancellation_policy']=airbnb['cancellation_policy'].replace('strict_14_with_grace_period','super')

airbnb.accommodates.describe()

# extract the 'prices' from the table
price = airbnb['price']
prices = []

# the values are stored as '$7,000' instead of 7000.
# data cleaning to make the values into floats
for element in price:
    # element = float(element[1:].replace(',',''))
    prices.append(element)

# replacing the current column into a new one for future use
airbnb['price'] = prices

# excluding listings with "0" for price, bedrooms, etc
airbnb = airbnb[airbnb.price > 0]
airbnb = airbnb[airbnb.bedrooms > 0]
airbnb = airbnb[airbnb.beds > 0]
# airbnb = airbnb[airbnb.review_scores_rating > 0]
airbnb = airbnb[airbnb.reviews_per_month > 0]
airbnb = airbnb[airbnb.accommodates > 0]
airbnb = airbnb[airbnb.bathrooms > 0]

# we only need latitude and longitude to determine the location
location = airbnb[['address.location.coordinates[1]', 'address.location.coordinates[0]']]

airbnb.room_type.unique()

# Red will represent Entire home / apartments
# Blue will represent private rooms
# Green will represent shared rooms

longitude='address.location.coordinates[0]'
latitude='address.location.coordinates[1]'

plt.figure(figsize=(12,12))

# the longitude and latitude limits of San Francisco
map_extent = [-160, -35, 152, 45.833]
themap = Basemap(llcrnrlon=map_extent[0],
                 llcrnrlat=map_extent[1],
                 urcrnrlon=map_extent[2],
                 urcrnrlat=map_extent[3],
                 projection='gall',
                 resolution='f', epsg=4269)

# creating outlines for the basemap
themap.drawcoastlines()
themap.drawcountries()
themap.fillcontinents(color = 'gainsboro')
themap.drawmapboundary(fill_color='steelblue')

# splitting each type of property
home = airbnb[(airbnb.room_type == 'Entire home/apt')]
private = airbnb[(airbnb.room_type == 'Private room')]
shared = airbnb[(airbnb.room_type == 'Shared room')]

# then splitting them based on longitude and latitude
a, b = themap(home['address.location.coordinates[0]'], home['address.location.coordinates[1]'])
c, d = themap(private['address.location.coordinates[0]'], private['address.location.coordinates[1]'])
e, f = themap(shared['address.location.coordinates[0]'], shared['address.location.coordinates[1]'])

# plotting using different colors
themap.plot(a, b, 'o',
            color='Red',
            markersize=4)
themap.plot(c, d, 'o',
            color='Green',
            markersize=4)
themap.plot(e, f, 'o',
            color='Blue',
            markersize=4)

# using Counter to analyze frequency of each listing based on neighborhood
nh = Counter(airbnb['neighbourhood_cleansed'])

nh

nh_df = pd.DataFrame.from_dict(nh, orient='index').sort_values(by=0)
nh_df.plot(kind='bar',
           color = 'LightBlue',
           figsize =(15,8),
           title = 'SF Neighborhood Frequency',
           legend = False)

average_price = sum(airbnb.price) / float(len(airbnb.price))
average_price

# extracting the names
neighborhood_names = list(nh.keys())

# 2 column table of neighborhood names and prices
nh_prices = airbnb[['neighbourhood_cleansed', 'price']]
nh_prices.columns = ['neighbourhood', 'price']

# we pick out the rows which have neighborhood names with 400+ listings.
nh_prices = nh_prices[nh_prices['neighbourhood'].isin(neighborhood_names)]

# group by neighbourhood and then aggreate the prices based on mean
nh_prices_group = nh_prices.groupby('neighbourhood')
nh_prices = nh_prices_group['price'].agg(np.mean)

# turn dictionary's keys and values into a table for easy read
nh_prices = nh_prices.reset_index()
nh_prices['number of listings'] = nh.values()

nh_prices

p = nh_prices.sort_values(by = 'price')
p.plot(x = "neighbourhood",
       y = "price",
       kind='bar',
       color = 'LightBlue',
       figsize =(15,8),
       title = 'Price per Neighorhood',
       legend = False)

room = airbnb.room_type
r = Counter(room)

room_df = pd.DataFrame.from_dict(r, orient='index').sort_values(by=0)
room_df.columns = ['room_type']
room_df.plot.pie(y = 'room_type',
                 colormap = 'Blues_r',
                 figsize=(8,8),
                 fontsize = 20, autopct = '%.2f',
                 legend = False,
                 title = 'Room Type Distribution')

property = airbnb.property_type
p = Counter(property)

property_df = pd.DataFrame.from_dict(p, orient='index').sort_values(by=0)
property_df.columns = ['property_type']
property_df.plot.bar(y= 'property_type',
                     color = 'LightBlue',
                     fontsize = 20,
                     legend = False,
                     figsize= (13, 7),
                     title = "Property type distribution")

prop_room = airbnb[['property_type', 'room_type', 'price']]

# first ten of the table
prop_room[0:10]

# Grouping by porperty and room type, and then aggregating them
# using mean of the price
prop_room_group = prop_room.groupby(['property_type', 'room_type']).mean()

# resetting the index in order to turn the lists into a readable table
p = prop_room_group.reset_index()

# pivoting the table based on the 3 factors, in order.
p = p.pivot('property_type', 'room_type', 'price')

# replacing the NaN values with 0
p.fillna(0.00, inplace=True)

p

summary = airbnb[['summary']]

# gets rid of NaN values with pandas' notnull function
summary = summary[pd.notnull(summary['summary'])]
summary

words = []

# accessing each summary, then each word, and putting it into an empty list
for detail in summary['summary']:
    if detail != 0:
        for word in detail.split():
            words.append(word)

# turning the list into a counter (dictionary) for frequency, then into a pandas dataframe
words = Counter(words)
word_count = pd.DataFrame.from_dict(words, orient='index').sort_values(by=0)

# renaming the column
word_count.columns = ['summary']

# sorting it from highest to lowest
word_count = word_count.sort_values(by=['summary'], ascending=False)
word_count

price_review = airbnb[['number_of_reviews', 'price']].sort_values(by = 'price')

price_review.plot(x = 'price',
                  y = 'number_of_reviews',
                  style = 'o',
                  figsize =(12,8),
                  legend = False,
                  title = 'Reviews based on Price')

sum(airbnb.number_of_reviews)

np.mean(airbnb.reviews_per_month)

cancel = airbnb.cancellation_policy
c = Counter(cancel)

# cleaning up small values
c.pop("super_strict_30", None)
c.pop("super_strict_60", None)
cancel_df = pd.DataFrame.from_dict(c, orient='index').sort_values(by=0)
cancel_df.columns = ['Cancellation Policy']
cancel_df.plot.pie(y = 'Cancellation Policy',
                   colormap = 'Blues_r',
                   figsize=(8,8),
                   fontsize = 20,
                   autopct = '%.2f',
                   legend = False,
                   title = "Cancellation Policy Distribution")

data = airbnb[['price',
           'room_type',
           'accommodates',
           'bathrooms',
           'bedrooms',
           'beds',
           'review_scores_rating',
           'instant_bookable',
           'cancellation_policy',
           'amenities']]

# cancellation policy and instant bookable factors are little bit more complicated.
# we use pandas get_dummies function to convert the categorical variable into indicator variables

cancel_policy = pd.get_dummies(data.cancellation_policy).astype(int)
instant_booking = pd.get_dummies(data.instant_bookable, prefix = 'instant_booking').astype(int)
room_type = pd.get_dummies(data.room_type).astype(int)

# ib has 2 columns, so we can just drop one of them.
instant_booking = instant_booking.drop('instant_booking_f', axis = 1)


# we drop the original columns and replace them with indicator columns
data = data.drop(['cancellation_policy', 'instant_bookable', 'room_type'], axis = 1)
data = pd.concat((data, cancel_policy, instant_booking, room_type), axis = 1)

# splitting the amenities list to draw out how many amenities each listing has

amenities_list = []

for element in data.amenities:
    element = element[1:]
    element = element[:-1]
    x = element.split()
    amenities_list.append(len(x))

data.amenities = amenities_list

data

#FURTHER ANALYSIS AND VISULAIZATION IS DONE USING TABLEAU/POWER BI USING THIS DATA
