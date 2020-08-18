# NASAstronauts
An astronaut is a human being who travels in space or is transported by something that does it. Thanks to this dataset we can know many curiosities about these "magic" people. This information relates to the generalities of astronauts from 1959 to 2017. In fact, in April 9, 1959, nine astronauts were selected for the Mercury Progra, one of the firsts SPACE programs.

## Dataset
This dataset (astronauts.csv) has nineteen columns, organized as follows: Name, Year, Group, Status, Birthdate, Birthplace, Gender, Alma mater, Undergraduate and Graduate Major.

## Questions
1) What is the average age of astronauts on duty? <br />
2) What are the most common academic studies? <br />
3) Have most astronauts served in the military? Which branch? <br />
4) What are the astronauts with more space walks? With how many hours? <br />
5) Are there more female or male? <br />
6) What is the average age of death? <br />

## Let's start
First of all, we need to import all the modules, in particular Pandas and Numpy for the data manipulation, then Matplotlib and Seaborn for the data visualization. Datetime and dateutil will be useful for working on ages. 


```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
```

Then we need to import the dataset that we'll use


```python
df = pd.read_csv('CSV/astronauts.csv')
```

Let's see the first input


```python
df.head(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Year</th>
      <th>Group</th>
      <th>Status</th>
      <th>Birth Date</th>
      <th>Birth Place</th>
      <th>Gender</th>
      <th>Alma Mater</th>
      <th>Undergraduate Major</th>
      <th>Graduate Major</th>
      <th>Military Rank</th>
      <th>Military Branch</th>
      <th>Space Flights</th>
      <th>Space Flight (hr)</th>
      <th>Space Walks</th>
      <th>Space Walks (hr)</th>
      <th>Missions</th>
      <th>Death Date</th>
      <th>Death Mission</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Joseph M. Acaba</td>
      <td>2004.0</td>
      <td>19.0</td>
      <td>Active</td>
      <td>5/17/1967</td>
      <td>Inglewood, CA</td>
      <td>Male</td>
      <td>University of California-Santa Barbara; Univer...</td>
      <td>Geology</td>
      <td>Geology</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>3307</td>
      <td>2</td>
      <td>13.0</td>
      <td>STS-119 (Discovery), ISS-31/32 (Soyuz)</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### 1) What is the average age of astronauts on duty? 
We need to focus just on the records with Status = "Active", and then we can calculate the average. We suppose for simplicity that they're currently active in 2020. We'll use another dataframe with just the born date of active astronauts.


```python
df_years = df[df["Status"] == "Active"]
df_years = df_years[['Birth Date']]
df_years.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Birth Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5/17/1967</td>
    </tr>
    <tr>
      <th>11</th>
      <td>8/23/1967</td>
    </tr>
    <tr>
      <th>15</th>
      <td>11/26/1963</td>
    </tr>
  </tbody>
</table>
</div>


Before going on, we have to check if there is any null value


```python
df_years.isnull().sum()
```




    Birth Date    0
    dtype: int64



Perfect, there is no null value, now we can add another column with the equivalent ages. 


```python
end_date = datetime.datetime.now()
df_years["Age"] = df_years.apply(lambda x: 
                                     relativedelta(end_date, 
                                                   datetime.datetime.strptime(x["Birth Date"], '%m/%d/%Y')
                                ).years, axis=1)

df_years.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Birth Date</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5/17/1967</td>
      <td>53</td>
    </tr>
    <tr>
      <th>11</th>
      <td>8/23/1967</td>
      <td>52</td>
    </tr>
    <tr>
      <th>15</th>
      <td>11/26/1963</td>
      <td>56</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_years["Age"].mean()
```




    54.34



**So the average age of astronauts on duty is 53.34 years!**

### 2) What are the most common academic studies? 
In this case we can use the "groupby" function and then count the occurrencies. After that, we can plot as we prefer, in this case I'll use a simple sns.barplot.


```python
undergraduate = df.groupby(["Undergraduate Major"]).count()['Name'].sort_values(ascending=False)[:10]

plt.figure(figsize=(12, 8))
sns.barplot(y=undergraduate.index, x=undergraduate.values, palette="Blues").set_title("Top 10 undergraduate Major")
plt.xlabel("Number of occurrences")
plt.ylabel(None)
```

![Error](https://github.com/francescodisalvo05/NASAstronauts/blob/master/Screen/output_15_1.png?raw=true)



```python
graduate = df.groupby(["Graduate Major"]).count()['Name'].sort_values(ascending=False)[:10]

plt.figure(figsize=(12, 8))
sns.barplot(y=graduate.index, x=graduate.values, palette="Blues").set_title("Top 10 graduate Major")
plt.xlabel("Number of occurrences")
plt.ylabel(None)
```
![Error](https://github.com/francescodisalvo05/NASAstronauts/blob/master/Screen/output_16_1.png?raw=true)


For completeness I wanna also show the top 10 universities and academies that trained these astronauts.


```python
universities = df.groupby(["Alma Mater"]).count()['Name'].sort_values(ascending=False)[:10]

plt.figure(figsize=(12, 8))
sns.barplot(y=universities.index, x=universities.values, palette="Blues").set_title("Top 10 Alma Mater")
plt.xlabel("Number of occurrences")
plt.ylabel(None)
```

![Error](https://github.com/francescodisalvo05/NASAstronauts/blob/master/Screen/output_18_1.png?raw=true)


### 3) Have most astronauts served in the military? Which branch?
This time we need to focus also on "null" values, beacuse it means that that astronauts did not serve in the military. So, for asking to this question we can replace NaN to "None" and considering it as an option. 


```python
df_military = df.fillna({'Military Branch':'None'})
df_military["Military Branch"][:5]
```




    0                      None
    1                      None
    2         US Army (Retired)
    3    US Air Force (Retired)
    4    US Air Force (Retired)
    Name: Military Branch, dtype: object



Now, as before, we can use the groupby function and plot the result in a pieplot.


```python
df_military.groupby(["Military Branch"]).count()['Name'].sort_values(ascending=False)
```




    Military Branch
    None                               146
    US Air Force (Retired)              61
    US Navy (Retired)                   59
    US Navy                             21
    US Air Force                        21
    US Marine Corps (Retired)           17
    US Army (Retired)                   13
    US Army                              4
    US Marine Corps                      3
    US Air Force Reserves (Retired)      3
    US Naval Reserves                    2
    US Marine Corps Reserves             2
    US Coast Guard (Retired)             2
    US Air Force Reserves                2
    US Naval Reserves (Retired)          1
    Name: Name, dtype: int64



Before we plot, we need to replace this "(Retired)" with the classic one. We can do this by the replace function. The first array contains all the value that we wanna replace, and the second one contains all the corresponding ones. 


```python
df_military = df_military.replace(["US Air Force (Retired)","US Air Force Reserves (Retired)",
                                   "US Army (Retired)","US Coast Guard (Retired)",
                                   "US Marine Corps (Retired)","US Naval Reserves (Retired)",
                                   "US Navy (Retired)"],
                                 
                                  ["US Air Force", "US Air Force Reserves","US Army","US Coast Guard",
                                   "US Marine Corps","US Naval Reserves", "US Navy"]
                                 ).groupby(["Military Branch"]).count()['Name'].sort_values(ascending=False)
df_military
```




    Military Branch
    None                        146
    US Air Force                 82
    US Navy                      80
    US Marine Corps              20
    US Army                      17
    US Air Force Reserves         5
    US Naval Reserves             3
    US Marine Corps Reserves      2
    US Coast Guard                2
    Name: Name, dtype: int64



Now we're finally ready to plot our result in a pie chart! For editing the default color, it can be helpful "cm" from Matplotlib, in fact we choose for the "tab20c" palette.


```python
cs=cm.tab20c(np.arange(40))

plt.figure(figsize=(12,12))
plt.pie(df_military.values, colors=cs, autopct='%.1f%%')

plt.ylabel(None)
plt.legend(df_military.index, loc="upper right")
```

![Error](https://github.com/francescodisalvo05/NASAstronauts/blob/master/Screen/output_26_1.png?raw=true)


### 4) What are the astronauts with more space walks? With how many hours?
Let's clean the dataset for making our process a bit easier. For a better visualization we'll consider just the first 10 results.


```python
df_walks = df[['Name','Space Walks']].sort_values('Space Walks', ascending=False)[:10]
df_walks.head(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Space Walks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>203</th>
      <td>Michael E. Lopez-Alegria</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_walks_hr = df[['Name','Space Walks (hr)']].sort_values('Space Walks (hr)', ascending=False)[:10]
df_walks_hr.head(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Space Walks (hr)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>203</th>
      <td>Michael E. Lopez-Alegria</td>
      <td>67.0</td>
    </tr>
  </tbody>
</table>
</div>



Now we can plot these results 


```python
f, axes = plt.subplots(2,1, figsize=(12, 10))
sns.despine(left=True)

sns.barplot(y=df_walks['Name'], x=df_walks['Space Walks'], ax=axes[0], palette="Blues")
sns.barplot(y=df_walks_hr['Name'], x=df_walks_hr['Space Walks (hr)'], ax=axes[1], palette="Blues")
```

![Error](https://github.com/francescodisalvo05/NASAstronauts/blob/master/Screen/output_31_1.png?raw=true)


### 5) Are there more female or male?
Also in this case we have to use the groupby function. We'll plot the results in a piechart and in a barchart to highlight the differences.


```python
df_gender = df.groupby('Gender')['Gender'].count()
df_gender.values
```




    array([ 50, 307], dtype=int64)




```python
f, axes = plt.subplots(1,2, figsize=(14, 8))

sns.despine(left=True)

sns.countplot(x=df['Gender'], ax=axes[0], palette="Blues")

cs=cm.tab20c(np.arange(40))
explode =(0.05,0)
plt.pie(df_gender, colors=cs, autopct='%.1f%%',explode=explode)
plt.legend(df_gender.index, loc="upper right")
```

![Error](https://github.com/francescodisalvo05/NASAstronauts/blob/master/Screen/output_34_1.png?raw=true)


### 6) What is the average age of death? 
We need another column "Age of death" where we'll insert the difference between death date and born that, then we have to calculate the average thanks to the mean() function. To avoid changing the original dataset I prefer to create another copy. 


```python
df_agedeath = df[['Birth Date', 'Death Date']]
df_agedeath.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Birth Date</th>
      <th>Death Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5/17/1967</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3/7/1936</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3/3/1946</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5/20/1951</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1/20/1930</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



As we can see, there are so many astronauts still alive (luckly), so we have to make a choice about null values. Since we need to focus just on the death astronauts I think we can delete this records with the function dropna().


```python
df_agedeath = df_agedeath.dropna()
df_agedeath.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Birth Date</th>
      <th>Death Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>12/25/1959</td>
      <td>2/1/2003</td>
    </tr>
    <tr>
      <th>14</th>
      <td>8/5/1930</td>
      <td>8/25/2012</td>
    </tr>
    <tr>
      <th>24</th>
      <td>12/30/1931</td>
      <td>2/28/1966</td>
    </tr>
    <tr>
      <th>36</th>
      <td>8/12/1951</td>
      <td>7/23/2006</td>
    </tr>
    <tr>
      <th>42</th>
      <td>4/16/1956</td>
      <td>2/1/2003</td>
    </tr>
  </tbody>
</table>
</div>



Now we can calculate di exactly age and the average age of death


```python
df_agedeath['Age of death'] = df_agedeath.apply(
                                            lambda x: relativedelta(
                                                        datetime.strptime(x["Death Date"], '%m/%d/%Y'), 
                                                        datetime.strptime(x["Birth Date"], '%m/%d/%Y')
                                ).years, axis=1)
df_agedeath['Age of death'].mean()
```




    51.17307692307692



Sadly, the average age of death is **51.17 years**. It's probably beacause the death astronauts belongs mostly to the past "era" of astronautics and they were many more accidents than nowadays. In fact, nowadays the average age of the active astronauts is around 54 years, 3 yers greater than the average age of death.

## Conclusion
As we saw before, in the past, astronauts has a shorter average life (around 51 yers), but nowadays luckly the situation has clearly improved. Most of the astronauts haven't served in military, and they're mostly Physicists, Aerospace and Mechanical Engineer, trained mostly from US Naval Academy. 

The most "active" astronauts is Michael E. Lopez-Algeria with ten walks with a duration of around 68 hours. Another intresting point is the differences between female (14%) and male (86%), maybe beacause in the past there were very few girls in the ISS but I'm pretty sure that this value will increase in the future!
