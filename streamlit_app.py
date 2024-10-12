# import necessary libraries
import random
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import os
from matplotlib import cm
import streamlit as st
from sodapy import Socrata
import requests
import io
import duckdb
from datetime import datetime
import time
import plotly.express as px
import plotly.graph_objects as go

# set df numeric formatting for floats
pd.set_option('display.float_format', '{:.4f}'.format)

# establish connection to database
con = duckdb.connect()
# con.close()


# collect data for filters: origins, destinations, hours, days of week
hours_df = con.sql("SELECT DISTINCT hour_of_day FROM 'subway-trips.parquet' ORDER BY 1 ASC").df()
day_of_week = con.sql("""
    SELECT DISTINCT day_of_week 
    FROM 'subway-trips.parquet' 
    ORDER BY 
        CASE day_of_week
            WHEN 'Monday' THEN 1
            WHEN 'Tuesday' THEN 2
            WHEN 'Wednesday' THEN 3
            WHEN 'Thursday' THEN 4
            WHEN 'Friday' THEN 5
            WHEN 'Saturday' THEN 6
            WHEN 'Sunday' THEN 7
        END
""").df()
origins_df = con.sql("SELECT DISTINCT origin_station_complex_name FROM 'subway-trips.parquet' ORDER BY 1 ASC").df()
destinations_df = con.sql("SELECT DISTINCT destination_station_complex_name FROM 'subway-trips.parquet' ORDER BY 1 DESC").df()
month_df = con.sql("SELECT DISTINCT month FROM 'subway-trips.parquet' ORDER BY 1 ASC").df()

# get holistic station list
all_stations = pd.concat([origins_df.rename(columns={'origin_station_complex_name': 'station'}),
                          destinations_df.rename(columns={'destination_station_complex_name': 'station'})]).drop_duplicates().reset_index(drop=True)



# Streamlit app content
st.title("MTA Subway Ridership Between Stations in 2023")
st.write(f"This interactive document allows users to explore New York MTA Ridership between different pairs or groupings  "
         f"of stations for the year of 2023. This project was built as a submission to the MTA Open Data Challenge. "
         f"It is recommended to select fewer than 5 stations at a time.")

st.sidebar.header('Filter Data')

# sidebar parameters
stations = st.sidebar.multiselect("Select Stations of Interest", options = all_stations['station'].tolist())
months_of_year = st.sidebar.multiselect("Select Months of Year", options= month_df['month'].tolist(), default=[])
days_of_week = st.sidebar.multiselect("Select Days of Week", options= day_of_week['day_of_week'].tolist(), default=[])
hours_of_day = st.sidebar.multiselect("Select Hours of Day", options= hours_df['hour_of_day'].tolist(), default=[])

agg_option = st.sidebar.selectbox("Aggregation", options=["Total Ridership", "Average Ridership", "Median Ridership"])
present_option = st.sidebar.selectbox("Origin, Destination, or Both", options = ["Both", "Origin", "Destination"])
selected_option = st.sidebar.selectbox("Selected : All or Selected : Selected", options = ["Selected", "All"])
distribution_breakout = st.sidebar.selectbox("Distribution Breakout", options = ["Hour of Day", "Day of Week", "Month"])


# Default conditions

if len(months_of_year)==0:
    months_of_year = month_df['month'].tolist()

if len(days_of_week)==0:
    days_of_week = day_of_week['day_of_week'].tolist()

if len(hours_of_day)==0:
    hours_of_day = hours_df['hour_of_day'].tolist()

if len(stations) < 1:
    stations = random.sample(all_stations['station'].tolist(), 3)

stations_2 = stations
if selected_option == "All":
    stations_2 = all_stations['station'].tolist()

# format all filtering user-inputs into passable strings as needed
hours_of_day_int = ', '.join(f"'{hour}'" for hour in hours_of_day)
months_of_year_int = ', '.join(f"'{month}'" for month in months_of_year)
stations_str = ', '.join(f"'{station.replace("'", "''")}'" for station in stations)
stations_2_str = ', '.join(f"'{station_2.replace("'", "''")}'" for station_2 in stations_2)
days_of_week_str = ', '.join(f"'{day}'" for day in days_of_week)


if distribution_breakout == "Day of Week":
    group_by_option = "day_of_week"
elif distribution_breakout == "Month":
    group_by_option = "month"
else:
    group_by_option = "hour_of_day"


# Dynamically determine the weight column and title based on user selection
if agg_option == "Total Ridership":
    weight_column = 'total_ridership'
    weight_title = "Total Ridership"
elif agg_option == "Average Ridership":
    weight_column = 'average_ridership'
    weight_title = "Average Ridership"
else:
    weight_column = 'median_ridership'
    weight_title = "Median Ridership"




# Table for Packed Bubble Chart
def get_popular_stations():
    if present_option == "Origin":
        query = f"""
        SELECT origin_station_complex_name as station, 
            ROUND(SUM(estimated_average_ridership),2) as total_ridership,
            ROUND(AVG(estimated_average_ridership),2) AS average_ridership,
               ROUND(MEDIAN(estimated_average_ridership),2) AS median_ridership,
               ROUND(VAR_POP(estimated_average_ridership),2) as variance_ridership,
               ROUND(MIN(estimated_average_ridership),2) as min_ridership,
               ROUND(MAX(estimated_average_ridership),2) as max_ridership,
               ROUND(MAX(estimated_average_ridership) - MIN(estimated_average_ridership),2) as range_ridership,
               ROUND(QUANTILE_CONT(estimated_average_ridership, 0.25), 2) AS first_quartile,
            ROUND(QUANTILE_CONT(estimated_average_ridership,0.75), 2) AS third_quartile,
            ROUND(QUANTILE_CONT(estimated_average_ridership,0.75) - QUANTILE_CONT(estimated_average_ridership,0.25), 2) AS iqr,
            ROUND(SKEWNESS(estimated_average_ridership), 2) AS skewness,
            ROUND(KURTOSIS_POP(estimated_average_ridership),2) AS kurtosis
        FROM 'subway-trips.parquet'
        WHERE origin_station_complex_name IN ({stations_str}) 
          AND destination_station_complex_name IN ({stations_2_str})  
          AND "month" IN ({months_of_year_int})  
          AND day_of_week IN ({days_of_week_str})  
          AND hour_of_day IN ({hours_of_day_int})
        GROUP BY 1
        ORDER BY 2 DESC
        """

    elif present_option == "Destination":
        query = f"""
            SELECT destination_station_complex_name as station, 
                ROUND(SUM(estimated_average_ridership),2) as total_ridership,
                ROUND(AVG(estimated_average_ridership),2) AS average_ridership,
               ROUND(MEDIAN(estimated_average_ridership),2) AS median_ridership,
               ROUND(VAR_POP(estimated_average_ridership),2) as variance_ridership,
               ROUND(MIN(estimated_average_ridership),2) as min_ridership,
               ROUND(MAX(estimated_average_ridership),2) as max_ridership,
               ROUND(MAX(estimated_average_ridership) - MIN(estimated_average_ridership),2) as range_ridership,
               ROUND(QUANTILE_CONT(estimated_average_ridership, 0.25), 2) AS first_quartile,
            ROUND(QUANTILE_CONT(estimated_average_ridership,0.75), 2) AS third_quartile,
            ROUND(QUANTILE_CONT(estimated_average_ridership,0.75) - QUANTILE_CONT(estimated_average_ridership,0.25), 2) AS iqr,
            ROUND(SKEWNESS(estimated_average_ridership), 2) AS skewness,
            ROUND(KURTOSIS_POP(estimated_average_ridership),2) AS kurtosis
            FROM 'subway-trips.parquet'
           WHERE origin_station_complex_name IN ({stations_2_str})  
          AND destination_station_complex_name IN ({stations_str})  
          AND "month" IN ({months_of_year_int})  
          AND day_of_week IN ({days_of_week_str}) 
          AND hour_of_day IN ({hours_of_day_int}) 
            GROUP BY 1
            ORDER BY 2 DESC
            """

    elif present_option == "Both":
        query = f"""
        WITH combined_ridership AS (
            SELECT origin_station_complex_name AS station, 
                   estimated_average_ridership
            FROM 'subway-trips.parquet'
            WHERE origin_station_complex_name IN ({stations_str}) 
            AND destination_station_complex_name IN ({stations_2_str})  
            AND "month" IN ({months_of_year_int})  
            AND day_of_week IN ({days_of_week_str})  
            AND hour_of_day IN ({hours_of_day_int})

            UNION ALL

            SELECT destination_station_complex_name AS station, 
                   estimated_average_ridership
            FROM 'subway-trips.parquet'
            WHERE origin_station_complex_name IN ({stations_2_str})  
            AND destination_station_complex_name IN ({stations_str})  
            AND "month" IN ({months_of_year_int})  
            AND day_of_week IN ({days_of_week_str}) 
            AND hour_of_day IN ({hours_of_day_int})
        )
        SELECT station,
               ROUND(SUM(estimated_average_ridership),2) as total_ridership,
               ROUND(AVG(estimated_average_ridership),2) AS average_ridership,
               ROUND(MEDIAN(estimated_average_ridership),2) AS median_ridership,
               ROUND(VAR_POP(estimated_average_ridership),2) as variance_ridership,
               ROUND(MIN(estimated_average_ridership),2) as min_ridership,
               ROUND(MAX(estimated_average_ridership),2) as max_ridership,
               ROUND(MAX(estimated_average_ridership) - MIN(estimated_average_ridership),2) as range_ridership,
               ROUND(QUANTILE_CONT(estimated_average_ridership, 0.25), 2) AS first_quartile,
            ROUND(QUANTILE_CONT(estimated_average_ridership,0.75), 2) AS third_quartile,
            ROUND(QUANTILE_CONT(estimated_average_ridership,0.75) - QUANTILE_CONT(estimated_average_ridership,0.25), 2) AS iqr,
            ROUND(SKEWNESS(estimated_average_ridership), 2) AS skewness,
            ROUND(KURTOSIS_POP(estimated_average_ridership),2) AS kurtosis
        FROM combined_ridership
        GROUP BY 1
        ORDER BY 2 DESC
        """
    df = con.execute(query).fetch_df()
    return df

# Generate popular stations table
popular_stations = get_popular_stations()

# Show the Bubble plot in Streamlit
# Get max values for scaling
max_x_value = popular_stations['average_ridership'].max()
max_y_value = popular_stations['median_ridership'].max()

# min_x_value = popular_stations['average_ridership'].min()
# min_y_value = popular_stations['median_ridership'].min()


st.subheader("Popular Stations")
st.write(f"The packed circles chart below displays the popularity of selected stations by total, "
         f"average, and median ridership levels. Hovering over each bubble will present exact figures for each. ")

# Create Packed Circles chart with Plotly Express
fig = px.scatter(popular_stations,
                 size='total_ridership',
                 x='average_ridership',
                 y='median_ridership',
                 color='total_ridership',
                 color_continuous_scale='darkmint',
                 hover_name='station',  # Show station on hover
                 hover_data={
                     'total_ridership': True,
                     'average_ridership': True,
                     'median_ridership': True,
                     'variance_ridership': False,
                     'station': False},
                 # text='station',
                 size_max=100,  # Max size for bubbles
                 labels={
                     "median_ridership": "Median Ridership",
                     "average_ridership": "Average Ridership",
                     "total_ridership": "Total Ridership"
                 },
                 title=f"Ridership by Station")



fig.update_layout(
    xaxis=dict(showgrid=False, range=[0, max_x_value * 1.3]),
    yaxis=dict(showgrid=False, range=[0, max_y_value * 1.3]),
    autosize=True
    # margin=dict(l=25, r=25, t=50, b=0)
)


st.plotly_chart(fig)

# Rename columns to display to user
popular_stations.rename(columns={
    'station': 'Station',
    'total_ridership': 'Total Ridership',
    'average_ridership': 'Average Ridership',
    'median_ridership': 'Median Ridership',
    'variance_ridership': 'Variance Ridership',
    'min_ridership': 'Min Ridership',
    'max_ridership': 'Max Ridership',
    'range_ridership': 'Range Ridership',
    'first_quartile': 'First Quartile',
    'third_quartile': 'Third Quartile',
    'iqr': 'Interquartile Range',
    'skewness': 'Skewness',
    'kurtosis': 'Kurtosis'
}, inplace = True)

# Display the table below the chart
st.write(f"The table below presents underlying data for the visual above along with several other descriptive "
         f"statistics. ")
st.dataframe(popular_stations)

# Add space between the visual and the next subheader
st.write("")  # One empty line
st.write("")  # Two empty lines
st.write("")  # Three empty lines


# Table for Distribution Tables

def dist_table():
    # Prepare ORDER BY clause
    if group_by_option == 'day_of_week':
        # Custom sorting for days of the week
        order_by_clause = f"""
        ORDER BY 
            CASE {group_by_option}
                WHEN 'Monday' THEN 1
                WHEN 'Tuesday' THEN 2
                WHEN 'Wednesday' THEN 3
                WHEN 'Thursday' THEN 4
                WHEN 'Friday' THEN 5
                WHEN 'Saturday' THEN 6
                WHEN 'Sunday' THEN 7
            END
        """
    else:
        order_by_clause = f"ORDER BY {group_by_option} ASC"

    # Prepare the SQL query and parameters based on present_option
    if present_option == "Origin":
        query = f"""
        SELECT {group_by_option},  
               ROUND(SUM(estimated_average_ridership),2) as total_ridership,
               ROUND(AVG(estimated_average_ridership),2) AS average_ridership,
               ROUND(MEDIAN(estimated_average_ridership),2) AS median_ridership,
               ROUND(MIN(estimated_average_ridership),2) as min_ridership,
               ROUND(MAX(estimated_average_ridership),2) as max_ridership,
               ROUND(MAX(estimated_average_ridership) - MIN(estimated_average_ridership),2) as range_ridership,
               ROUND(QUANTILE_CONT(estimated_average_ridership, 0.25), 2) AS first_quartile,
            ROUND(QUANTILE_CONT(estimated_average_ridership,0.75), 2) AS third_quartile,
            ROUND(QUANTILE_CONT(estimated_average_ridership,0.75) - QUANTILE_CONT(estimated_average_ridership,0.25), 2) AS iqr,
            ROUND(SKEWNESS(estimated_average_ridership), 2) AS skewness,
            ROUND(KURTOSIS_POP(estimated_average_ridership),2) AS kurtosis
        FROM 'subway-trips.parquet'
        WHERE origin_station_complex_name IN ({stations_str}) 
          AND destination_station_complex_name IN ({stations_2_str})  
          AND "month" IN ({months_of_year_int})  
          AND day_of_week IN ({days_of_week_str})  
          AND hour_of_day IN ({hours_of_day_int})  
        GROUP BY {group_by_option}
        {order_by_clause}
        """
    elif present_option == "Destination":
        query = f"""
        SELECT {group_by_option},
                ROUND(SUM(estimated_average_ridership),2) as total_ridership,  
               ROUND(AVG(estimated_average_ridership),2) AS average_ridership,
               ROUND(MEDIAN(estimated_average_ridership),2) AS median_ridership,
               ROUND(MIN(estimated_average_ridership),2) as min_ridership,
               ROUND(MAX(estimated_average_ridership),2) as max_ridership,
               ROUND(MAX(estimated_average_ridership) - MIN(estimated_average_ridership),2) as range_ridership,
               ROUND(QUANTILE_CONT(estimated_average_ridership, 0.25), 2) AS first_quartile,
            ROUND(QUANTILE_CONT(estimated_average_ridership,0.75), 2) AS third_quartile,
            ROUND(QUANTILE_CONT(estimated_average_ridership,0.75) - QUANTILE_CONT(estimated_average_ridership,0.25), 2) AS iqr,
            ROUND(SKEWNESS(estimated_average_ridership), 2) AS skewness,
            ROUND(KURTOSIS_POP(estimated_average_ridership),2) AS kurtosis
        FROM 'subway-trips.parquet'
        WHERE origin_station_complex_name IN ({stations_2_str})  
          AND destination_station_complex_name IN ({stations_str})  
          AND "month" IN ({months_of_year_int})  
          AND day_of_week IN ({days_of_week_str}) 
          AND hour_of_day IN ({hours_of_day_int}) 
        GROUP BY {group_by_option}
        {order_by_clause}
        """

    elif present_option == "Both":
        query = f"""
        WITH combined_ridership AS (
            SELECT origin_station_complex_name AS station, 
                   estimated_average_ridership,
                   {group_by_option}
            FROM 'subway-trips.parquet'
            WHERE origin_station_complex_name IN ({stations_str}) 
            AND destination_station_complex_name IN ({stations_2_str})  
            AND "month" IN ({months_of_year_int})  
            AND day_of_week IN ({days_of_week_str})  
            AND hour_of_day IN ({hours_of_day_int}) 

            UNION ALL

            SELECT destination_station_complex_name AS station, 
                   estimated_average_ridership,
                   {group_by_option}
            FROM 'subway-trips.parquet'
            WHERE origin_station_complex_name IN ({stations_2_str})  
            AND destination_station_complex_name IN ({stations_str})  
            AND "month" IN ({months_of_year_int})  
            AND day_of_week IN ({days_of_week_str}) 
            AND hour_of_day IN ({hours_of_day_int}) 
        )
        SELECT {group_by_option},
                ROUND(SUM(estimated_average_ridership),2) as total_ridership,
               ROUND(AVG(estimated_average_ridership),2) AS average_ridership,
               ROUND(MEDIAN(estimated_average_ridership),2) AS median_ridership,
               ROUND(MIN(estimated_average_ridership),2) as min_ridership,
               ROUND(MAX(estimated_average_ridership),2) as max_ridership,
               ROUND(MAX(estimated_average_ridership) - MIN(estimated_average_ridership),2) as range_ridership,
               ROUND(QUANTILE_CONT(estimated_average_ridership, 0.25), 2) AS first_quartile,
            ROUND(QUANTILE_CONT(estimated_average_ridership,0.75), 2) AS third_quartile,
            ROUND(QUANTILE_CONT(estimated_average_ridership,0.75) - QUANTILE_CONT(estimated_average_ridership,0.25), 2) AS iqr,
            ROUND(SKEWNESS(estimated_average_ridership), 2) AS skewness,
            ROUND(KURTOSIS_POP(estimated_average_ridership),2) AS kurtosis

        FROM combined_ridership
        GROUP BY {group_by_option} 
        {order_by_clause}

        """
    df = con.execute(query).fetch_df()

    return df

# Generate stats table
dist_stats = dist_table()

st.subheader(f"Distribution of {weight_title} by {distribution_breakout}")
# Create and display the Bar chart
plt.figure(figsize=(12, 6))

# Plot a bar chart for the grouped data
ax = sns.barplot(data=dist_stats, x=group_by_option, y=weight_column, color='skyblue', alpha=0.7)
ax.bar_label(ax.containers[0], fontsize=10)

# Remove axes spines (which are the plot borders/box)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Customize the plot
plt.xlabel(distribution_breakout)
plt.ylabel(weight_title)
plt.xticks(rotation=45)  # Rotate x labels if necessary

st.pyplot(plt)

# Rename columns to display to user
dist_stats.rename(columns={
    group_by_option: distribution_breakout,
    'total_ridership': 'Total Ridership',
    'average_ridership': 'Average Ridership',
    'median_ridership': 'Median Ridership',
    'variance_ridership': 'Variance Ridership',
    'min_ridership': 'Min Ridership',
    'max_ridership': 'Max Ridership',
    'range_ridership': 'Range Ridership',
    'first_quartile': 'First Quartile',
    'third_quartile': 'Third Quartile',
    'iqr': 'Interquartile Range',
    'skewness': 'Skewness',
    'kurtosis': 'Kurtosis'
}, inplace = True)

# Display the table below the chart
st.write(f"The table below contains underlying data and descriptive statistics for the above visual.")
st.dataframe(dist_stats)

# Add space between the visual and the next subheader
st.write("")  # One empty line
st.write("")  # Two empty lines
st.write("")  # Three empty lines

# Table for Graph Network
def graph_pairs():
    query = f"""
    SELECT origin_station_complex_name as station_1, destination_station_complex_name as station_2,
            ROUND(SUM(estimated_average_ridership),2) as total_ridership, 
           ROUND(AVG(estimated_average_ridership),2) AS average_ridership,
               ROUND(MEDIAN(estimated_average_ridership),2) AS median_ridership,
               ROUND(MIN(estimated_average_ridership),2) as min_ridership,
               ROUND(MAX(estimated_average_ridership),2) as max_ridership,
               ROUND(MAX(estimated_average_ridership) - MIN(estimated_average_ridership),2) as range_ridership,
               ROUND(QUANTILE_CONT(estimated_average_ridership, 0.25), 2) AS first_quartile,
            ROUND(QUANTILE_CONT(estimated_average_ridership,0.75), 2) AS third_quartile,
            ROUND(QUANTILE_CONT(estimated_average_ridership,0.75) - QUANTILE_CONT(estimated_average_ridership,0.25), 2) AS iqr,
            ROUND(SKEWNESS(estimated_average_ridership), 2) AS skewness,
            ROUND(KURTOSIS_POP(estimated_average_ridership),2) AS kurtosis
    FROM 'subway-trips.parquet'
    WHERE origin_station_complex_name IN ({stations_str})
      AND destination_station_complex_name IN ({stations_str})
      AND "month" IN ({months_of_year_int})
      AND day_of_week IN ({days_of_week_str})
      AND hour_of_day IN ({hours_of_day_int})
    GROUP BY 1, 2
    ORDER BY 1,2
    """
    df = con.execute(query).fetch_df()
    return df

# Generate graph table
graph_pairs = graph_pairs()

st.subheader(f"Graph of Station pairs depicting {agg_option}")

# Create the networkX graph and corresponding plot
# Check if graph_pairs is empty
if graph_pairs.empty:
    st.warning("Please select two connected stations to see the graph.")
else:
    # Create a directed graph
    G = nx.DiGraph()

    # Create a weighted edgelist from the DataFrame
    edgelist = [(row['station_1'], row['station_2'], row[weight_column]) for _, row in graph_pairs.iterrows()]

    # Add edges with weights
    G.add_weighted_edges_from(edgelist)

    try:
        # Calculate Eigenvector Centrality
        centrality = nx.eigenvector_centrality(G, max_iter=10000, tol=1.0e-3, weight='weight')
        centrality_values = list(centrality.values())

        # Scale node sizes based on centrality
        min_size = 1000
        max_size = 5000

        # Normalize centrality values to a scale between min_size and max_size
        scaled_node_sizes = [
            min_size + (centrality[node] - min(centrality.values())) /
            (max(centrality.values()) - min(centrality.values())) * (max_size - min_size)
            for node in G.nodes()
        ]


    except nx.NetworkXPointlessConcept:
        # Fallback to in-degree and out-degree if centrality fails to converge
        in_degrees = dict(G.in_degree(weight='weight'))
        out_degrees = dict(G.out_degree(weight='weight'))

        # Calculate combined in-degree and out-degree for node sizes and colors
        combined_degrees = {node: in_degrees[node] + out_degrees[node] for node in G.nodes()}

        # Normalize combined degrees for node sizes
        min_combined = min(combined_degrees.values())
        max_combined = max(combined_degrees.values())

        scaled_node_sizes = [
            min_size + (combined_degrees[node] - min_combined) /
            (max_combined - min_combined) * (max_size - min_size)
            for node in G.nodes()
        ]

        # Use the combined degrees for coloring the nodes
        centrality_values = list(combined_degrees.values())

    # Set up the colormap and position
    blues_mid = plt.get_cmap('Blues', 256)
    blues_custom = blues_mid(np.linspace(0.2, 0.9, 256))

    # create the graph
    fig2, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, k = 0.5, iterations=1000, weight = 'weight', scale = 1)
    nx.draw(G, pos, with_labels=True,
            node_size=scaled_node_sizes,
            connectionstyle='arc3, rad = 0.1',
            node_color= centrality_values, cmap=cm.colors.ListedColormap(blues_custom))


 #   edge_labels = dict([((u, v,), f'{d["weight"]}\n\n{G.edges[(v,u)]["weight"]}')
  #                  for u, v, d in G.edges(data=True) if pos[u][0] > pos[v][0]])

    edge_labels = {}
    for u, v, d in G.edges(data=True):
        reverse_edge_exists = G.has_edge(v, u)
        edge_labels[(
        u, v)] = f'{d["weight"]}\n\n{G.edges[(v, u)]["weight"]}' if reverse_edge_exists else f'{d["weight"]}\n\nN/A'

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue')


    # Display the plot in Streamlit
    st.pyplot(fig2)

# Rename columns to display to user
graph_pairs.rename(columns={
    'station_1': 'Station 1',
    'station_2': 'Station 2',
    'total_ridership': 'Total Ridership',
    'average_ridership': 'Average Ridership',
    'median_ridership': 'Median Ridership',
    'variance_ridership': 'Variance Ridership',
    'min_ridership': 'Min Ridership',
    'max_ridership': 'Max Ridership',
    'range_ridership': 'Range Ridership',
    'first_quartile': 'First Quartile',
    'third_quartile': 'Third Quartile',
    'iqr': 'Interquartile Range',
    'skewness': 'Skewness',
    'kurtosis': 'Kurtosis'
}, inplace = True)

# Display the table below the chart
st.write(f"The table below contains the underlying data and descriptive statistics for each edge "
         f"in the network displayed above. ")
st.dataframe(graph_pairs)

