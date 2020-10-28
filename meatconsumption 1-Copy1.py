#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os


# In[2]:


pwd


# In[3]:


df_prod1=pd.read_csv(r"C:\Users\Khaled Sallemi\Desktop\WorldQuant\Dataset\archive\animals-slaughtered-for-meat.csv")


# In[4]:


df_prod2=pd.read_csv(r"C:\Users\Khaled Sallemi\Desktop\WorldQuant\Dataset\archive\global-meat-production.csv")
df_prod2.head()


# In[5]:


df_prod3=pd.read_csv(r"C:\Users\Khaled Sallemi\Desktop\WorldQuant\Dataset\archive\global-meat-production-by-livestock-type.csv")
df_prod3.head()


# In[6]:


data_prod1=pd.merge(df_prod3,df_prod2, on=['Entity','Year','Code'], how='left')
data_prod1.head()


# In[7]:


data_prod1.dtypes


# In[8]:


df_cons1=pd.read_csv(r"C:\Users\Khaled Sallemi\Desktop\WorldQuant\Dataset\archive\per-capita-meat-consumption-by-type-kilograms-per-year.csv")
df_cons1.head()


# In[9]:


df_cons1.dtypes


# In[10]:


df_cons1['Year']=df_cons1.Year.astype(str)


# In[ ]:





# In[11]:


df_cons2=pd.read_csv(r"C:\Users\Khaled Sallemi\Desktop\WorldQuant\Dataset\archive\meat-consumption-vs-gdp-per-capita.csv")
df_cons2.head()


# In[12]:


df_cons2.dtypes


# In[13]:


data_cons1=pd.merge(df_cons1,df_cons2, on=['Entity','Code','Year'], how='left')
data_cons1.head()


# In[14]:


data_prod1['Year']=data_prod1.Year.astype(str)


# In[15]:


data=pd.merge(data_prod1,data_cons1, on=['Entity','Code','Year'], how='left')
data.head()


# In[16]:


cols=['entity','code','year','sheep_goat_prod(T)','beef_buffalo_prod(T)','pig_prod(T)',
      'wildgame_prod(T)','duck_prod(T)','chiken_prod(T)','horse_prod(T)','camel_prod(T)',
      'goose_guinea_prod(T)','total_meat_prod(T)','sheep_goat_cons(kg/ind)',
      'other_meat_cons(kg/ind)','poultry_cons(kg/ind)','pig_cons(kg/ind)',
      'beef_buffalo_cons(kg/ind)','total_cons(kg/ind)','GDP per capita',
     'Total population']


# In[17]:


data.columns=cols
data.head()


# In[18]:


df_p14=pd.read_csv(r"C:\Users\Khaled Sallemi\Desktop\WorldQuant\Dataset\archive\protein-efficiency-of-meat-and-dairy-production.csv")
df_p14


# In[19]:


df_p5=pd.read_csv(r"C:\Users\Khaled Sallemi\Desktop\WorldQuant\Dataset\archive\energy-efficiency-of-meat-and-dairy-production.csv")
df_p5.head()


# In[20]:


df_efficiency= pd.merge(df_p5,df_p14, on=['Entity','Code','Year'], how='left')
df_efficiency.drop('Code',axis=1, inplace=True)
eff_cols=['entity','year','NRJ_efficiency','PRO_efficiency']
df_efficiency.columns=eff_cols
df_efficiency


# In[21]:


df_c1=pd.read_csv(r"C:\Users\Khaled Sallemi\Desktop\WorldQuant\Dataset\meat_consumption_worldwide.csv")
df_c1.head()


# In[22]:


df_c2=pd.read_csv(r"C:\Users\Khaled Sallemi\Desktop\WorldQuant\Dataset\meat_consumption.csv")
df_c2.head()


# In[23]:


data['year']=data.year.astype(int)


# In[24]:


meat_pro_cons_2010=data[data['year']>=2010]
meat_pro_cons_2010.shape


# In[25]:


list(meat_pro_cons_2010.year.unique())


# In[26]:


meat_pro_cons_2010.columns


# In[27]:


meat_pro_cons_2010.sort_values('total_meat_prod(T)', ascending=False).head()


# In[28]:


meat_pro_cons_2010.entity.unique()
African_countries=['Algeria','Angola','Benin','Botswana','Burkina Faso','Burundi','Cameroon','Cape Verde',
 'Central African Republic','Chad','Comoros','Congo',"Cote d'Ivoire",'Democratic Republic of Congo', 
 'Djibouti','Egypt', 'Equatorial Guinea', 'Eritrea','Estonia', 'Ethiopia', 'Gabon','Gambia', 'Ghana',
 'Guinea', 'Guinea-Bissau', 'Kenya','Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali',
 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia','Niger', 'Nigeria', 'Rwanda',
 'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa',
 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe']
len(African_countries)


# In[29]:


af_meat_2010=meat_pro_cons_2010[meat_pro_cons_2010['entity'].isin(African_countries)]
af_meat_2010.head(10)


# In[30]:


af_meat_2010.sort_values(['year','sheep_goat_prod(T)'], ascending=False).entity.head()


# In[31]:


products=['sheep_goat_prod(T)', 'beef_buffalo_prod(T)','pig_prod(T)', 'chiken_prod(T)','total_meat_prod(T)']
years=list(af_meat_2010.year.unique())
years


# In[32]:


def the_best (item,yea):
    d={}
    for i in item:
        for y in yea:
            top3 = list(af_meat_2010[af_meat_2010['year']==y].sort_values(i, ascending=False).entity.head(3))
            d[(i,y)] = top3
    return pd.DataFrame(d)


# In[33]:


df_prod=the_best(products,years).T
df_prod=df_prod.rename(index={'sheep_goat_prod(T)':'sheep_goat',
                      'beef_buffalo_prod(T)':'beef_buffalo',
                      'pig_prod(T)':'pig',
                      'chiken_prod(T)': 'poultry',
                      'total_meat_prod(T)':'total'})
df_prod.reset_index(inplace=True)
df_prod.columns=['item','year','1st_producer','2nd_producer','3rd_producer']
df_prod


# In[34]:


consumption=['sheep_goat_cons(kg/ind)','beef_buffalo_cons(kg/ind)','pig_cons(kg/ind)',
             'poultry_cons(kg/ind)','total_cons(kg/ind)']


# In[35]:


df_cons=the_best(consumption,years).T
df_cons=df_cons.rename(index={'sheep_goat_cons(kg/ind)':'sheep_goat',
                      'beef_buffalo_cons(kg/ind)':'beef_buffalo',
                      'pig_cons(kg/ind)':'pig',
                      'poultry_cons(kg/ind)': 'poultry',
                      'total_cons(kg/ind)':'total'})
df_cons.reset_index(inplace=True)
df_cons.columns=['item','year','1st_consumer','2nd_consumer','3rd_consumer']
df_cons


# In[36]:


df=pd.merge(df_prod, df_cons, on=['item','year'], how= 'left')
df.head()


# In[37]:


df.groupby('item')['1st_producer'].value_counts()


# In[38]:


df.groupby('1st_producer')['item'].value_counts()


# In[39]:


df.groupby('item')['1st_consumer'].value_counts()


# In[40]:


df.groupby('1st_consumer')['item'].value_counts()


# In[41]:


get_ipython().system('pip install plotly')


# In[42]:


import json


# In[43]:


african_map= json.load(open(r"C:\Users\Khaled Sallemi\Desktop\WorldQuant\Dataset\customafrica.geo.json"))


# In[44]:


country_id_map={}
for feature in african_map['features']:
    feature['id']=feature['properties']['pop_est']
    country_id_map[feature['properties']['name']] = feature['id']


# In[45]:


df_id=pd.DataFrame()
df_id['entity']=country_id_map.keys()
df_id['id']=country_id_map.values()
print(df_id['entity'].unique())
print(af_meat_2010['entity'].unique())
df_id


# In[46]:


df_id.loc[5,'entity']='Central African Republic'
df_id.loc[6,'entity']="Cote d'Ivoire"
df_id.loc[8,'entity']='Democratic Republic of Congo'
df_id.loc[20,'entity']='Equatorial Guinea'
df_id.loc[37,'entity']='South Sudan'


# In[47]:


african_map['features'][0]['properties']['name']


# In[48]:


african_meat=pd.merge(af_meat_2010, df_id, on='entity', how='left')
african_meat.head()


# In[49]:


import plotly.express as px


# In[50]:


fig_prod=px.choropleth(african_meat, locations='id', geojson= african_map, color='sheep_goat_prod(T)', scope='africa')
fig_prod.show()


# In[51]:


fig_cons=px.choropleth(african_meat, locations='id', geojson= african_map, color='sheep_goat_cons(kg/ind)', scope='africa')
fig_cons.show()


# In[52]:


data.head()


# In[53]:


data.groupby('entity')['GDP per capita'].sum()


# In[54]:


data3af=data[data['entity'].isin(African_countries)]
data3af


# In[55]:


data3af.columns


# In[56]:


data3af=data3af[['entity', 'code', 'year', 'sheep_goat_prod(T)', 'beef_buffalo_prod(T)',
       'pig_prod(T)','chiken_prod(T)','total_meat_prod(T)', 'sheep_goat_cons(kg/ind)',
       'beef_buffalo_cons(kg/ind)','pig_cons(kg/ind)','poultry_cons(kg/ind)','total_cons(kg/ind)',
       'GDP per capita','Total population']]


# In[57]:


data3af[ 'GDP per capita'].isna().sum()


# In[58]:


country_med_gdp=data3af.groupby('entity')['GDP per capita'].median()
country_med_gdp


# In[59]:


X1=data3af[data3af.entity=='Mauritania']
X1.head()


# In[60]:


X=X1[['year','sheep_goat_cons(kg/ind)','Total population']]
X.shape


# In[61]:


X['sheep_goat_cons(kg/ind)'].fillna(value= 0, inplace=True)
X['sheep_goat_cons(kg/ind)'].isna().sum()


# In[62]:


y=data3af[data3af.entity=='Mauritania']['sheep_goat_prod(T)']
y.isnull().sum()


# In[63]:


from sklearn.model_selection import train_test_split


# In[64]:


X_train, X_test, y_train, y_test= train_test_split(X,y, random_state=0)


# In[72]:


y_train.fillna(value= 0, inplace=True)
X_train=X_train.dropna(how='any')
X_train.isnull().sum()


# In[67]:


from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[73]:


pipeline1= Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])


# In[ ]:


pipeline1.fit(X_train,y_train)


# In[ ]:


pipeline1.predict(X_test)


# In[ ]:


pipeline1.steps_names_


# In[ ]:




