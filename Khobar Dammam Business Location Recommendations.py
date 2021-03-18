#!/usr/bin/env python
# coding: utf-8

# # The Battle of Neighborhoods - Coursera Capstone Project
# 
# This notebook constitutes of the capstone project for the IBM Data Science Professional Certificate provided by Coursera.
# 
# The goal of this notebook is to determine optimal opening locations for different kinds of businesses, focused on the twin cities of Al-Khobar and Dammam, Saudi Arabia.
# 
# The end result will beproviding a simple way to get recommendations on where to open a certain business.
# 
# ## Installing Dependencies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import os
import time
import json
import requests
from copy import deepcopy
from dotenv import load_dotenv
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import matplotlib.colors as colors
from folium.plugins import HeatMap
import uuid
from IPython.display import display_javascript, display_html, display
import collections
import streamlit as st
from streamlit_folium import folium_static

# ## Creating GeoJSON for Saudi Arabia
# 
# To do any kind of analysis, we need to segregate the map into cities and districts, define boundries for the same and mark the centerpoints of each district.
# 
# Saudi Arabia can be neatly divided into provinces, cities and districts. We are particularly interested in Al-Khobar and Dammam, both of which are in the Eastern Province of Saudi Arabia.
# 
# Fortunately, a contributer on GitHub has already gathered the coordinate data, saving us the time to scrape it ourselves. However, we will have to build the GeoJSON ourselves. 
# 
# ### Importing coordinates JSON
# 
# Source: https://github.com/homaily/Saudi-Arabia-Regions-Cities-and-Districts
# 
# Let us import that into pandas.

# In[2]:

st.title('Opening A Business in Khobar/Dammam - Location Recommendations')

with open(r"json/cities.json", 'r', encoding='utf8', errors='ignore') as file:
    cities = json.load(file)
    
with open(r"json/districts.json", 'r', encoding='utf8', errors='ignore') as file:
    districts = json.load(file)

#st.write("Improted CSVs for boundary data.")

# Let's take a look at how the JSON is structured. 
# 
# We can see city_id and district_id serve as the primary keys.

# In[3]:


#cities[1] #showing only 1 record out of many


# In[4]:


#districts[1] #showing only 1 record out of many


# ### Getting city_id of Al Khobar and Dammam

# In[5]:


#for city in cities:
#    if "Khobar" in city["name_en"]:
#        print("The city_id of " + city["name_en"] + " is " + str(city["city_id"]) + ".")
#    if "Dammam" in city["name_en"]:
#        print("The city_id of " + city["name_en"] + " is " + str(city["city_id"]) + ".")


# ### Getting districts in Al Khobar and Dammam
# 
# Now we can get the districts that constitute Al Khobar and Dammam.

# In[6]:


khobar_districts = []
dammam_districts = []

for district in districts:
    if district["city_id"] == 31:
        khobar_districts.append(district)
        
for district in districts:
    if district["city_id"] == 13:
        dammam_districts.append(district)


# ### Converting into GeoJSON format
# 
# We will create a copy of the data so we can convert latitude,longitude coordinates to longitude,latitude coordinates supported by GeoJSON.
# 
# More information here, see "Position" section: https://macwright.com/2015/03/23/geojson-second-bite.html 

# In[7]:


khobar_districts_xy = deepcopy(khobar_districts)
dammam_districts_xy = deepcopy(dammam_districts)


# Reversing boundaries coordinates as per GeoJSON format

# In[8]:


for district in range(len(khobar_districts_xy)):
    for _ in khobar_districts_xy[district]["boundaries"][0]:
        _.reverse()
        
for district in range(len(dammam_districts_xy)):
    for _ in dammam_districts_xy[district]["boundaries"][0]:
        _.reverse()


# Let's also define the centerpoints of the district

# In[9]:


for district in range(len(khobar_districts_xy)):
    khobar_districts_xy[district]["center"] = [sum(x)/len(x) for x in zip(*khobar_districts_xy[district]["boundaries"][0])]
    khobar_districts_xy[district]["latitude"] = khobar_districts_xy[district]["center"][0]
    khobar_districts_xy[district]["longitude"] = khobar_districts_xy[district]["center"][1]    
    
for district in range(len(khobar_districts)):
    khobar_districts[district]["center"] = [sum(x)/len(x) for x in zip(*khobar_districts[district]["boundaries"][0])]
    khobar_districts[district]["latitude"] = khobar_districts[district]["center"][0]
    khobar_districts[district]["longitude"] = khobar_districts[district]["center"][1]
    
for district in range(len(dammam_districts_xy)):
    dammam_districts_xy[district]["center"] = [sum(x)/len(x) for x in zip(*dammam_districts_xy[district]["boundaries"][0])]
    dammam_districts_xy[district]["latitude"] = dammam_districts_xy[district]["center"][0]
    dammam_districts_xy[district]["longitude"] = dammam_districts_xy[district]["center"][1]
    
for district in range(len(dammam_districts)):
    dammam_districts[district]["center"] = [sum(x)/len(x) for x in zip(*dammam_districts[district]["boundaries"][0])]
    dammam_districts[district]["latitude"] = dammam_districts[district]["center"][0]
    dammam_districts[district]["longitude"] = dammam_districts[district]["center"][1]


# Let's go ahead and put that into a Pandas DataFrame

# In[10]:


khobar = pd.DataFrame(khobar_districts)
dammam = pd.DataFrame(dammam_districts)

khobar_xy = pd.DataFrame(khobar_districts_xy)
dammam_xy = pd.DataFrame(dammam_districts_xy)


#khobar_xy


# Finally, we can parse all the above data into the GeoJSON format

# In[11]:


features = []

#create Khobar GeoJSON
for district in range(len(khobar_districts_xy)):
    feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": khobar_districts_xy[district]["boundaries"]},
            "properties": {
                "district_id": khobar_districts_xy[district]["district_id"],
                "city_id": khobar_districts_xy[district]["city_id"],
                "name_en": khobar_districts_xy[district]["name_en"]}
        }

    features.append(feature)
    
khobar_geojson = {
    "type": "FeatureCollection",
    "features": features
}

khobar_geojson = json.dumps(khobar_geojson)


# In[12]:


features = []

#create Dammam GeoJSON
for district in range(len(dammam_districts_xy)):
    feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": dammam_districts_xy[district]["boundaries"]},
            "properties": {
                "district_id": dammam_districts_xy[district]["district_id"],
                "city_id": dammam_districts_xy[district]["city_id"],
                "name_en": dammam_districts_xy[district]["name_en"]}
        }

    features.append(feature)
    
dammam_geojson = {
    "type": "FeatureCollection",
    "features": features
}

dammam_geojson = json.dumps(dammam_geojson)


# The GeoJSON is now in the proper format for usage.

# ### Plotting districts and boundaries

# In[13]:


khobar_data = khobar[["name_en", "district_id", "center"]]
dammam_data = dammam[["name_en", "district_id", "center"]]


# In[14]:


# create a plain map
khobar_map = folium.Map(location=[26.2172,50.1971], zoom_start=12)

folium.GeoJson(khobar_geojson).add_to(khobar_map)

#Add title to Map
loc = "Al Khobar"

title_html = '''<h3 align="center" style="font-size:16px"><b>{}</b></h3>'''.format(loc)   

khobar_map.get_root().html.add_child(folium.Element(title_html))

#"TODO: Add District Labels"

# display map
#"Khobar Districts"
#folium_static(khobar_map)


# In[52]:


# create a plain map
dammam_map = folium.Map(location=[26.4207,50.0888], zoom_start=12)

folium.GeoJson(dammam_geojson).add_to(dammam_map)

#Add title to Map
loc = "Dammam"

title_html = '''
             <h3 align="center" style="font-size:16px"><b>{}</b></h3>
             '''.format(loc)   

dammam_map.get_root().html.add_child(folium.Element(title_html))

#"TODO: Add District Labels"

# display map
#"Dammam Districts"
#folium_static(dammam_map)


# ## Using Foursquare API to retrieve popular venues in each district 
# 
# Having registered beforehand for the Foursqaure developer program (https://developer.foursquare.com/), we can use the API to get a list of popular venues in each district.
# 
# But first, security. We will the dotenv package to safely import our public and private keys to pass to the Foursquare API.

# In[16]:


#using python-dotenv to protect Foursqaure credentials
#get_ipython().run_line_magic('load_ext', 'dotenv')
#get_ipython().run_line_magic('dotenv', '')
#import os

#CLIENT_ID = os.getenv("CLIENT_ID") # your Foursquare ID
#CLIENT_SECRET = os.getenv("CLIENT_SECRET") # your Foursquare Secret
CLIENT_ID = "FTYQ3NXJY3OWGWAVEBSUZJQZMDQI5PNORHKXL5ZHO3UMUC2T"
CLIENT_SECRET = "QYB1ASVT3RVFTBMTCGGGXQEGN332C2ENSJDF10SH2DBBIOSS"
VERSION = '20180605' # Foursquare API version
LIMIT = 100 # A default Foursquare API limit value

#print('Your credentials:')
#print('CLIENT_ID SIZE: ' + str(len(CLIENT_ID)))
#print('CLIENT_SECRET SIZE: ' + str(len(CLIENT_SECRET)))


# ### Function to get nearby popular venues
# 
# This function will be called recursively to retrieve nearby venues. 

# In[17]:


def getNearbyVenues(
    names,
    latitudes,
    longitudes,
    radius=500,
    ):

    venues_list = []
    for (name, lat, lng) in zip(names, latitudes, longitudes):

        # create the API request URL

        url =             'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID,
            CLIENT_SECRET,
            VERSION,
            lat,
            lng,
            radius,
            LIMIT,
            )

        # make the GET request

        results = requests.get(url).json()['response']['groups'][0]['items']

        # return only relevant information for each nearby venue

        venues_list.append([(
            name,
            lat,
            lng,
            v['venue']['name'],
            v['venue']['location']['lat'],
            v['venue']['location']['lng'],
            v['venue']['categories'][0]['name'],
            ) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list
                                 for item in venue_list])
    nearby_venues.columns = [
        'District',
        'District Latitude',
        'District Longitude',
        'Venue',
        'Venue Latitude',
        'Venue Longitude',
        'Venue Category',
        ]

    return nearby_venues


# ### Getting nearby venues
# 
# Let's run our function and utilize the Foursqaure API to get nearby venues for each District.

# In[18]:

if st.sidebar.checkbox("Get latest data? (takes some time)"):

	st.write("Please hold, getting latest data...")

	khobar_venues = getNearbyVenues(names=khobar['name_en'],
        	                           latitudes=khobar['latitude'],
                	                   longitudes=khobar['longitude']
                        	          )

	khobar_venues.to_csv("cached_data/khobar_venues.csv")


	# In[19]:


	dammam_venues = getNearbyVenues(names=dammam['name_en'],
        	                           latitudes=dammam['latitude'],
                	                   longitudes=dammam['longitude']
                        	          )

	dammam_venues.to_csv("cached_data/dammam_venues.csv")

	"Got data! Thank you for waiting :)"

	# ### Unique Venue Categories
	# 
	# Let's merge and list the unique categories gathered from both the citites.

	# In[20]:


	khobar_unique_cat = khobar_venues['Venue Category'].unique()

	with open('cached_data/khobar_unique_cat.txt', 'w') as f:
	    for item in khobar_unique_cat:
        	f.write("%s\n" % item)

	#st.write('There are {} unique categories in Al-Khobar.'.format(len(khobar_unique_cat)))

	#khobar_unique_cat


	# In[21]:


	dammam_unique_cat = dammam_venues['Venue Category'].unique()

	with open('cached_data/dammam_unique_cat.txt', 'w') as f:
	    for item in dammam_unique_cat:
        	f.write("%s\n" % item)


	#st.write('There are {} unique categories in Al-dammam.'.format(len(dammam_unique_cat)))
	
	#dammam_unique_cat


	# In[22]:


	unique_venue_categories = khobar_unique_cat.tolist() + dammam_unique_cat.tolist()

	unique_venue_categories = np.unique(unique_venue_categories)

	#print('There are {} unique categories in both Al Khobar and Dammam overall.'.format(len(unique_venue_categories)))

	#unique_venue_categories

else:

	with open('cached_data/khobar_unique_cat.txt', 'r') as f:
		khobar_unique_cat = [line.rstrip() for line in f]
	
	with open('cached_data/dammam_unique_cat.txt', 'r') as f:
		dammam_unique_cat = [line.rstrip() for line in f]

	khobar_venues = pd.read_csv("cached_data/khobar_venues.csv")
	dammam_venues = pd.read_csv("cached_data/dammam_venues.csv")


#Enter search criteria
CITY = st.sidebar.selectbox("Which city would you like to open your business in?", ["Khobar","Dammam"])


#Choose between Khobar and Dammam
if CITY == "Khobar":

    DISTRICT =  st.sidebar.selectbox("Select your district:", khobar_xy["name_en"])
    VENUE_CATEGORY = st.sidebar.selectbox("Select your business category:", khobar_unique_cat)

if CITY == "Dammam":

    DISTRICT = st.sidebar.selectbox("Select your district:", dammam_xy["name_en"])
    VENUE_CATEGORY = st.sidebar.selectbox("Select your business category:", dammam_unique_cat)

# ### Onehot encoding
# 
# We can do one-hot encoding to convert our variables into integers that we can easily analyze with our algorithms

# In[23]:


## KHOBAR
# one hot encoding
khobar_onehot = pd.get_dummies(khobar_venues[['Venue Category']], prefix="", prefix_sep="")

# add District column back to dataframe
khobar_onehot['District'] = khobar_venues['District'] 

# move District column to the first column
khobar_onehot = khobar_onehot[ ['District'] + [ col for col in khobar_onehot.columns if col != 'District' ] ]


## DAMMAM
# one hot encoding
dammam_onehot = pd.get_dummies(dammam_venues[['Venue Category']], prefix="", prefix_sep="")

# add District column back to dataframe
dammam_onehot['District'] = dammam_venues['District'] 

# move District column to the first column
dammam_onehot = dammam_onehot[ ['District'] + [ col for col in dammam_onehot.columns if col != 'District' ] ]



#khobar_onehot.head()


# In[24]:


khobar_grouped = khobar_onehot.groupby('District').mean().reset_index()
dammam_grouped = dammam_onehot.groupby('District').mean().reset_index()

#khobar_grouped


# ### Getting most common venues in each district
# 
# The venue data gathered up to this point is just a list. We can group that by frequency of the occurence of each venue within a particular district to get an idea of the kind of place the district is. 

# In[25]:


#Function to return most common venues in each District
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[26]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['District']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))


# ### Top Venues in Khobar's Districts

# In[27]:


# create a new dataframe
khobar_venues_sorted = pd.DataFrame(columns=columns)
khobar_venues_sorted['District'] = khobar_grouped['District']

for ind in np.arange(khobar_grouped.shape[0]):
    khobar_venues_sorted.iloc[ind, 1:] = return_most_common_venues(khobar_grouped.iloc[ind, :], num_top_venues)

#khobar_venues_sorted


# ### Top Venues in Dammam's Districts

# In[28]:


# create a new dataframe
dammam_venues_sorted = pd.DataFrame(columns=columns)
dammam_venues_sorted['District'] = dammam_grouped['District']

for ind in np.arange(dammam_grouped.shape[0]):
    dammam_venues_sorted.iloc[ind, 1:] = return_most_common_venues(dammam_grouped.iloc[ind, :], num_top_venues)

#dammam_venues_sorted


# ## K-means Clustering
# 
# We can utilize the K-means Clustering machine learning algorithm to group the different districts into clusters based on their most common venues.
# 
# We will use 5 clusters, any higher than that leads to ineffective clutering. You can rerun this notebook with a different number of k clusters and observe the changes on the maps below.

# In[29]:


khobar_data = khobar_data.rename(columns={"name_en": "District"})
dammam_data = dammam_data.rename(columns={"name_en": "District"})


# ### Khobar Clusters

# In[30]:


# set number of clusters
kclusters = 5

khobar_grouped_clustering = khobar_grouped.drop('District', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(khobar_grouped_clustering)

# check cluster labels generated for each row in the dataframe
#kmeans.labels_

# add clustering labels
khobar_venues_sorted.insert(0, 'Cluster Label', kmeans.labels_)

khobar_merged = khobar_data

# merge khobar_grouped with khobar_data to add latitude/longitude for each District
khobar_merged = khobar_merged.join(khobar_venues_sorted.set_index('District'), on='District')

khobar_merged.dropna(axis=0, inplace = True)

#khobar_merged.head() # check the last columns!


# ### Dammam Clusters

# In[31]:


# set number of clusters
kclusters = 5

dammam_grouped_clustering = dammam_grouped.drop('District', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(dammam_grouped_clustering)

# check cluster labels generated for each row in the dataframe
#kmeans.labels_

# add clustering labels
dammam_venues_sorted.insert(0, 'Cluster Label', kmeans.labels_)

dammam_merged = dammam_data

# merge dammam_grouped with dammam_data to add latitude/longitude for each District
dammam_merged = dammam_merged.join(dammam_venues_sorted.set_index('District'), on='District')

dammam_merged.dropna(axis=0, inplace = True)

#dammam_merged.head() # check the last columns!


# In[32]:


#Adding Latitude and Longitude of each District's Center
khobar_merged["Latitude"] = [ x[0] for x in khobar_merged["center"].tolist() ]
khobar_merged["Longitude"] = [ x[1] for x in khobar_merged["center"].tolist() ]

dammam_merged["Latitude"] = [ x[0] for x in dammam_merged["center"].tolist() ]
dammam_merged["Longitude"] = [ x[1] for x in dammam_merged["center"].tolist() ]

#khobar_merged


# ### Plotting Clusters and listing the districts in each cluster
# 
# Let's plot the clusters obtained from our K-means clustering in a Folium map. This provides a nice visual for the different kinds of clusters.
# 
# What do you think each cluster represents? For example, Al Khobar's Cluster 0 seems to represent areas with many restaurants and dining options. What about the rest?
# 
# ### Al Khobar

# In[54]:


# create map
map_clusters = folium.Map(location=[26.2172,50.1971], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(khobar_merged['Latitude'], khobar_merged['Longitude'], khobar_merged['District'], khobar_merged['Cluster Label']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[int(cluster)-1],
        fill=True,
        fill_color=rainbow[int(cluster)-1],
        fill_opacity=0.7).add_to(map_clusters)

#Add title to Map
loc = "Al Khobar Clusters"

title_html = '''
             <h3 align="center" style="font-size:16px"><b>{}</b></h3>
             '''.format(loc)   

map_clusters.get_root().html.add_child(folium.Element(title_html))    

#"Khobar Clusters"

#folium_static(map_clusters)


# In[34]:


# Cluster 0:

#khobar_merged.loc[khobar_merged['Cluster Label'] == 0, khobar_merged.columns[[0] + list(range(4, khobar_merged.shape[1]))]]


# In[35]:


# Cluster 1

#khobar_merged.loc[khobar_merged['Cluster Label'] == 1, khobar_merged.columns[[0] + list(range(4, khobar_merged.shape[1]))]]


# In[36]:


# Cluster 2

#khobar_merged.loc[khobar_merged['Cluster Label'] == 2, khobar_merged.columns[[0] + list(range(4, khobar_merged.shape[1]))]]


# In[37]:


# Cluster 3

#khobar_merged.loc[khobar_merged['Cluster Label'] == 3, khobar_merged.columns[[0] + list(range(4, khobar_merged.shape[1]))]]


# In[38]:


# Cluster 4

#khobar_merged.loc[khobar_merged['Cluster Label'] == 4, khobar_merged.columns[[0] + list(range(4, khobar_merged.shape[1]))]]


# ### Dammam

# In[55]:


# create map
map_clusters = folium.Map(location=[26.4207,50.0888], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(dammam_merged['Latitude'], dammam_merged['Longitude'], dammam_merged['District'], dammam_merged['Cluster Label']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[int(cluster)-1],
        fill=True,
        fill_color=rainbow[int(cluster)-1],
        fill_opacity=0.7).add_to(map_clusters)

#Add title to Map
loc = "Dammam Clusters"

title_html = '''
             <h3 align="center" style="font-size:16px"><b>{}</b></h3>
             '''.format(loc)   

map_clusters.get_root().html.add_child(folium.Element(title_html))       
    
#"Dammam Clusters"

#folium_static(map_clusters)


# In[40]:


# Cluster 0

#dammam_merged.loc[dammam_merged['Cluster Label'] == 0, dammam_merged.columns[[0] + list(range(4, dammam_merged.shape[1]))]]


# In[41]:


# Cluster 1

#dammam_merged.loc[dammam_merged['Cluster Label'] == 1, dammam_merged.columns[[0] + list(range(4, dammam_merged.shape[1]))]]


# In[42]:


# Cluster 2

#dammam_merged.loc[dammam_merged['Cluster Label'] == 2, dammam_merged.columns[[0] + list(range(4, dammam_merged.shape[1]))]]


# In[43]:


# Cluster 3

#dammam_merged.loc[dammam_merged['Cluster Label'] == 3, dammam_merged.columns[[0] + list(range(4, dammam_merged.shape[1]))]]


# In[44]:


# Cluster 4

#dammam_merged.loc[dammam_merged['Cluster Label'] == 4, dammam_merged.columns[[0] + list(range(4, dammam_merged.shape[1]))]]


# ## Recommending Districts, Locations and Clusters
# 
# The final stage will be to recommend district, cluster and possible locations to open any kind of business.

# ## Recommending Districts for Businesses
# 
# A district will be recommended if similar venues are numerous and thriving in the same district. Since Foursquare by default returns only popular venues, we can safely assume that the list of venues represents the popular venues for that area. 
# 
# Hence, we can simply sort the districts by the highest number of venues of the same category and present the results as a top 10 list. 
# 
# We can then output a map showing the recommended districts and their boundaries.

# In[45]:


def get_district_recommendation(city, venue_category, map):

    if city == "Khobar":
        
        top_districts = khobar_grouped.sort_values([venue_category], ascending=[False])

        top_districts_list = top_districts["District"].tolist()
        
    if city == "Dammam":
        
        top_districts = dammam_grouped.sort_values([venue_category], ascending=[False])

        top_districts_list = top_districts["District"].tolist()
    
    #return top 10 districts
    top_districts_list = top_districts_list[:10]
    
    #add districts to map
    for district in top_districts_list:
            
            if city == "Khobar":
                
                #Create district boundary GEoJSON
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": khobar_xy.loc[khobar_xy["name_en"] == district, "boundaries"].item()},
                    "properties": {
                        "district_id": khobar_xy.loc[khobar_xy["name_en"] == district, "district_id"].item(),
                        "city_id": khobar_xy.loc[khobar_xy["name_en"] == district, "city_id"].item(),
                        "name_en": khobar_xy.loc[khobar_xy["name_en"] == district, "name_en"].item()}
                    }

                geojson = {
                    "type": "FeatureCollection",
                    "features": [feature]
                }

                #Add district boundaries to map 
                folium.GeoJson(geojson).add_to(map)
                
            if city == "Dammam":
                
                #Create district boundary GEoJSON
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": dammam_xy.loc[dammam_xy["name_en"] == district, "boundaries"].item()},
                    "properties": {
                        "district_id": dammam_xy.loc[dammam_xy["name_en"] == district, "district_id"].item(),
                        "city_id": dammam_xy.loc[dammam_xy["name_en"] == district, "city_id"].item(),
                        "name_en": dammam_xy.loc[dammam_xy["name_en"] == district, "name_en"].item()}
                    }

                geojson = {
                    "type": "FeatureCollection",
                    "features": [feature]
                }

                #Add district boundaries to map 
                folium.GeoJson(geojson).add_to(map)

    return top_districts_list


# In[46]:

#Choose between Khobar and Dammam
if CITY == "Khobar":

    recommendation_map = folium.Map(location = [26.2172,50.1971], zoom_start = 12)

if CITY == "Dammam":

    recommendation_map = folium.Map(location = [26.4207,50.0888], zoom_start = 12)


#Add title to Map
loc = VENUE_CATEGORY + " Locations  in " + CITY

title_html = '''
             <h3 align="center" style="font-size:16px"><b>{}</b></h3>
             '''.format(loc)   

recommendation_map.get_root().html.add_child(folium.Element(title_html))

st.write(f"We recommend the following districts to open a {VENUE_CATEGORY} in {CITY}:")
"TODO - Add District Marker"
get_district_recommendation(CITY, VENUE_CATEGORY, recommendation_map)

folium_static(recommendation_map)


# ## Recommending Locations for Businesses
# 
# Obviously, if there are other venues in the same area, there is a higher competition. But sometimes we want to open near to other similar businesses to benefit from the footfall. So to recommend possible locations, we will do two things:
# 
# 1) Show the user a heatmap of similar businesses.
# 
# 2) Future Improvement: Take an input Coordinate, a Competition Importance Factor (CIF) and a Footfall Importance Factor (FIF), and return a Recommendation Factor (RF).

# In[47]:


#Function to get a Location Recommandation Map

def get_location_recommendation(city, district, venue_category, map):
    
    #Khobar
    if city == "Khobar":
        
        khobar_venues_copy = khobar_venues[khobar_venues["Venue Category"] == venue_category].copy()
        
        if not district == "NA": 
            khobar_venues_copy = khobar_venues_copy[khobar_venues_copy["District"] == district].copy() 
            
            #Create district boundary GEoJSON
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": khobar_xy.loc[khobar_xy["name_en"] == district, "boundaries"].item()},
                "properties": {
                    "district_id": khobar_xy.loc[khobar_xy["name_en"] == district, "district_id"].item(),
                    "city_id": khobar_xy.loc[khobar_xy["name_en"] == district, "city_id"].item(),
                    "name_en": khobar_xy.loc[khobar_xy["name_en"] == district, "name_en"].item()}
                }

            geojson = {
                "type": "FeatureCollection",
                "features": [feature]
            }
            
            #Add district boundaries to map 
            folium.GeoJson(geojson).add_to(map)
        
        khobar_venues_copy['count'] = 1
        
        #Add heatpoints to map where venues are located
        HeatMap(data=khobar_venues_copy[['Venue Latitude', 'Venue Longitude', 'count']].groupby(['Venue Latitude', 'Venue Longitude']).sum().reset_index().values.tolist(), radius=20).add_to(map)
    
    #Dammam
    if city == "Dammam":
        dammam_venues_copy = dammam_venues[dammam_venues["Venue Category"] == venue_category].copy()
        
        if not district == "NA": 
            dammam_venues_copy = dammam_venues_copy[dammam_venues_copy["District"] == district].copy() 
            
            #Create district boundary GEoJSON
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": dammam_xy.loc[dammam_xy["name_en"] == district, "boundaries"].item()},
                "properties": {
                    "district_id": dammam_xy.loc[dammam_xy["name_en"] == district, "district_id"].item(),
                    "city_id": dammam_xy.loc[dammam_xy["name_en"] == district, "city_id"].item(),
                    "name_en": dammam_xy.loc[dammam_xy["name_en"] == district, "name_en"].item()}
                }

            geojson = {
                "type": "FeatureCollection",
                "features": [feature]
            }

            #Add district boundaries to map 
            folium.GeoJson(geojson).add_to(map)
        
        dammam_venues_copy['count'] = 1
        
        #Add heatpoints to map where venues are located
        HeatMap(data=dammam_venues_copy[['Venue Latitude', 'Venue Longitude', 'count']].groupby(['Venue Latitude', 'Venue Longitude']).sum().reset_index().values.tolist(), radius=20).add_to(map)


# Let's enter our choice of the District from one of the recommended Districts:

# In[48]:


#Enter search criteria
#DISTRICT = "Ad Dawasir Dist."

# Now we can get our recommendation map:

# In[49]:


#Choose between Khobar and Dammam
if CITY == "Khobar":

    recommendation_map = folium.Map(location = (((khobar_xy.loc[khobar_xy["name_en"] == DISTRICT, "longitude"].values)),((khobar_xy.loc[khobar_xy["name_en"] == DISTRICT, "latitude"].values))), zoom_start = 15)

if CITY == "Dammam":
    
    recommendation_map = folium.Map(location = (((dammam_xy.loc[dammam_xy["name_en"] == DISTRICT, "longitude"].values)),((dammam_xy.loc[dammam_xy["name_en"] == DISTRICT, "latitude"].values))), zoom_start = 15)

get_location_recommendation(CITY, DISTRICT, VENUE_CATEGORY, recommendation_map)

#Add title to Map
loc = VENUE_CATEGORY + " Locations In " + DISTRICT

title_html = '''
             <h3 align="center" style="font-size:16px"><b>{}</b></h3>
             '''.format(loc)   

recommendation_map.get_root().html.add_child(folium.Element(title_html))

"Location Recommendation"

#Display map
folium_static(recommendation_map)


# ## Recommending Clusters for Businesses
# 
# Finally, we'd like to get a recommendation for the cluster where the business should open.
# 
# Since there isn't enough data to build the correlation between clusters and business success, we'll recommend the venue if it is in the top 10 venues for that cluster. We'll also assign a recommendation point, which will be the frequency of the venue occuring in the top 10 venues for every district.

# In[50]:


def get_cluster_recommendation(city, venue_category):    
    
    recommended_clusters = []
    recommended_clusters_list = []
    empty = False

    if city == "Khobar":

        khobar_venues_sorted_onehot = khobar_venues_sorted.copy()

        prefixes = ["1st","2nd","3rd","4th","5th","6th","7th","8th","9th","10th"]

        for prefix in prefixes:

            columns = prefix + " Most Common Venue"

            x = [list(a) for a in khobar_venues_sorted_onehot["Cluster Label"][khobar_venues_sorted_onehot[columns].str.contains(venue_category)].items()]

            recommended_clusters.append(x)
            

    if city == "Dammam":

        dammam_venues_sorted_onehot = dammam_venues_sorted.copy()

        prefixes = ["1st","2nd","3rd","4th","5th","6th","7th","8th","9th","10th"]

        for prefix in prefixes:

            columns = prefix + " Most Common Venue"

            x = [list(a) for a in dammam_venues_sorted_onehot["Cluster Label"][dammam_venues_sorted_onehot[columns].str.contains(venue_category)].items()]

            recommended_clusters.append(x)
    
    
    st.write("The recommended clusters are:")
    rc = 0
    for _ in recommended_clusters:
        if not len(_) == 0:
            rc += 1
            recommended_clusters_list.append(_[0][1])
    if rc == 0:
        st.write("\nSorry, could not find a recommended cluster. Please choose another venue category or city.")
    
    ctr = collections.Counter(recommended_clusters_list)
    
    for _ in ctr:
        st.write("Cluster: " + str(_) + "   Recommendation Points: " + str(ctr[_]))
    


# In[51]:

get_cluster_recommendation(CITY,VENUE_CATEGORY)
