import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


#matplolib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

import os


df = pd.read_csv("./data_for_graphics.csv")

df.dropna(inplace=True)

df.isnull().sum()


# Median Age vs GDP
fig = px.scatter(df, 
                 x='Median age', 
                 y='Gini coefficient of income', 
                  
                 color='Country name',
                 trendline='ols',
                 trendline_scope='overall',
                 trendline_color_override='black'
                )
fig.update_layout(title_text='Median Age vs GDP')
fig.show()

#Population of countries with its meadian age
fig = px.scatter(df, 
                 x='Population 2020', 
                 y='Median age',
                 color='Country name')
fig.update_layout(title_text='Population of countries with its meadian age')
fig.show()



# Median Age vs Covid 19 death per 100000 in 2020
d1 = df.loc[:,['Country name','Population 2020','Median age','COVID-19 deaths per 100,000 population in 2020']]
fig = px.scatter(d1,
                 x='Median age',
                 y='COVID-19 deaths per 100,000 population in 2020',
                 size='Population 2020',
                 color='Country name',
                 size_max=50,
                 hover_name='Country name',
                 log_x=True
                )

fig.update_layout(title_text='Meadian age VS deaths per 100,000')
fig.show()

# Index of institutional trust per Country
d2 = df.loc[:,['Country name','Index of institutional trust']].sort_values(ascending=False, by='Index of institutional trust').head(20)
fig = px.bar(d2,
             x='Country name',
             y='Index of institutional trust'
            )
fig.update_layout(title_text='Top 10 countries with highest index of institutional trust')
fig.show()