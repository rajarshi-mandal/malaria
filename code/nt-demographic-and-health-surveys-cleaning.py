import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


angola = pd.read_csv("malaria-data-for-modeling-dynamics/Demographic and Health Surveys/raw/Angola 2015-16 Standard DHS, DHS-7.csv")
angola.columns = angola.columns.str.replace('_', ' ')
angola.replace(-9999, np.nan, inplace=True)
cote = pd.read_csv("malaria-data-for-modeling-dynamics/Demographic and Health Surveys/raw/Cote dIvoire 2021 Standard DHS, DHS-8.csv")
cote.columns = cote.columns.str.replace('_', ' ')
cote.replace(-9999, np.nan, inplace=True)
cote.loc[cote['Rainfall 2020'] < 0, 'Rainfall 2020'] = np.nan
ethiopia = pd.read_csv("malaria-data-for-modeling-dynamics/Demographic and Health Surveys/raw/Ethiopia 2016 Standard DHS, DHS-7.csv")
ethiopia.columns = ethiopia.columns.str.replace('_', ' ')
ethiopia.replace(-9999, np.nan, inplace=True)
malawi = pd.read_csv("malaria-data-for-modeling-dynamics/Demographic and Health Surveys/raw/Malawi 2017, MIS DHS-7.csv")
malawi.columns = malawi.columns.str.replace('_', ' ')
malawi.replace(-9999, np.nan, inplace=True)
nigeria = pd.read_csv("malaria-data-for-modeling-dynamics/Demographic and Health Surveys/raw/Nigeria 2021 MIS, DHS-8.csv")
nigeria.columns = nigeria.columns.str.replace('_', ' ')
nigeria.replace(-9999, np.nan, inplace=True)


### DO NOT DELETE THESE
### angola.to_csv("Angola DHS7",index=False)
### cote.to_csv("Cote d'Ivoire DHS8",index=False)
### ethiopia.to_csv("Ethiopia DHS7",index=False)
### malawi.to_csv("Malawi DHS7",index=False)
### nigeria.to_csv("Nigeria DHS8",index=False)


# ## Angola


plt.figure(figsize=(9, 7))
subset = angola[['Rainfall 2015','Wet Days 2015','Annual Precipitation 2015','Aridity 2015','ITN Coverage 2015','Enhanced Vegetation Index 2015','Malaria Incidence 2015','Malaria Prevalence 2015','Mean Temperature 2015', 'Land Surface Temperature 2015','U5 Population 2015','UN Population Count 2015', 'UN Population Density 2015']]
subset.columns = subset.columns.str.replace(' 2015', '', regex=False)
sns.heatmap(subset.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Angola Feature Correlations in 2015")
plt.show()


plt.figure(figsize=(6, 4))
subset = angola[['Rainfall 2015','Wet Days 2015','Annual Precipitation 2015','Aridity 2015','ITN Coverage 2015','Enhanced Vegetation Index 2015','Malaria Incidence 2015','Malaria Prevalence 2015','Mean Temperature 2015', 'Land Surface Temperature 2015','U5 Population 2015','UN Population Count 2015', 'UN Population Density 2015']]
subset.columns = subset.columns.str.replace(' 2015', '', regex=False)
sns.heatmap(subset.corr(), annot=False, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5, linecolor='black')
plt.title("Angola Feature Correlations in 2015")
plt.show()


sns.scatterplot(data=subset,x='ITN Coverage',y='Malaria Prevalence',hue='Enhanced Vegetation Index',palette='Oranges')
plt.title("Angola ITN Coverage and Malaria in 2015")
plt.xlabel("ITN Coverage")
plt.ylabel("Malaria Prevalence")
plt.show()


plt.figure(figsize=(4,3))
subset['Median'] = subset['Malaria Prevalence'] > subset['Malaria Prevalence'].median()
sns.boxplot(x='Median',y='Enhanced Vegetation Index',data=subset)
plt.title("Angola EVI in 2015")
plt.xticks([0, 1], ['Low Malaria', 'High Malaria'])
plt.show()


# ## Cote d'Ivoire


plt.figure(figsize=(9, 7))
subset = cote[['Enhanced Vegetation Index 2020','Malaria Incidence 2020', 'Malaria Prevalence 2020','ITN Coverage 2020','Rainfall 2020','Wet Days 2020','Aridity 2020','Precipitation 2020','U5 Population 2020','UN Population Count 2020', 'UN Population Density 2020','Land Surface Temperature 2020','Mean Temperature 2020']]
subset.columns = subset.columns.str.replace(' 2020', '', regex=False)
sns.heatmap(subset.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Cote d'Ivoire Feature Correlations in 2020")
plt.show()


plt.figure(figsize=(6, 4))
subset = cote[['Enhanced Vegetation Index 2020','Malaria Incidence 2020', 'Malaria Prevalence 2020','ITN Coverage 2020','Rainfall 2020','Wet Days 2020','Aridity 2020','Precipitation 2020','U5 Population 2020','UN Population Count 2020', 'UN Population Density 2020','Land Surface Temperature 2020','Mean Temperature 2020']]
subset.columns = subset.columns.str.replace(' 2020', '', regex=False)
sns.heatmap(subset.corr(), annot=False, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5, linecolor='black')
plt.title("Cote d'Ivoire Feature Correlations in 2020")
plt.show()


# Basically bad data
# Rainfall 900 is already pretty high in most placed
sns.scatterplot(data=subset,x='ITN Coverage',y='Malaria Prevalence',hue='Rainfall',palette='Oranges')
plt.title("Cote d'Iovire ITN Coverage and Malaria in 2020")
plt.xlabel("ITN Coverage")
plt.ylabel("Malaria Prevalence")
plt.show()


plt.figure(figsize=(4,3))
subset['Median'] = subset['Malaria Prevalence'] > subset['Malaria Prevalence'].median()
sns.boxplot(x='Median',y='Rainfall',data=subset)
plt.title("Cote d'Ivoire Rainfall in 2020")
plt.xticks([0, 1], ['Low Malaria', 'High Malaria'])
plt.show()


# ## Ethiopia


plt.figure(figsize=(9, 7))
subset = ethiopia[['Land Surface Temperature 2015','Mean Temperature 2015','ITN Coverage 2015','Malaria Incidence 2015','Malaria Prevalence 2015','Annual Precipitation 2015','Aridity 2015','Wet Days 2015','Enhanced Vegetation Index 2015','Rainfall 2015','UN Population Count 2015','U5 Population 2015','UN Population Density 2015']]
subset.columns = subset.columns.str.replace(' 2015', '', regex=False)
sns.heatmap(subset.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Ethiopia Feature Correlations in 2015")
plt.show()


plt.figure(figsize=(6, 4))
subset = ethiopia[['Land Surface Temperature 2015','Mean Temperature 2015','ITN Coverage 2015','Malaria Incidence 2015','Malaria Prevalence 2015','Annual Precipitation 2015','Aridity 2015','Wet Days 2015','Enhanced Vegetation Index 2015','Rainfall 2015','UN Population Count 2015','U5 Population 2015','UN Population Density 2015']]
subset.columns = subset.columns.str.replace(' 2015', '', regex=False)
sns.heatmap(subset.corr(), annot=False, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5, linecolor='black')
plt.title("Ethiopia Feature Correlations in 2015")
plt.show()


sns.scatterplot(data=subset,x='ITN Coverage',y='Malaria Prevalence',hue='Mean Temperature',palette='Oranges')
plt.title("Ethiopia ITN Coverage and Malaria in 2015")
plt.xlabel("ITN Coverage")
plt.ylabel("Malaria Prevalence")
plt.show()


plt.figure(figsize=(4,3))
subset['Median'] = subset['Malaria Prevalence'] > subset['Malaria Prevalence'].median()
sns.boxplot(x='Median',y='Mean Temperature',data=subset)
plt.title("Ethiopia Temperature in 2015")
plt.xticks([0, 1], ['Low Malaria', 'High Malaria'])
plt.show()


# ## Malawi


plt.figure(figsize=(9, 7))
subset = malawi[['Malaria Incidence 2015','Malaria Prevalence 2015','Mean Temperature 2015','ITN Coverage 2015','Land Surface Temperature 2015','UN Population Count 2015','U5 Population 2015','UN Population Density 2015','Wet Days 2015','Annual Precipitation 2015','Aridity 2015','Enhanced Vegetation Index 2015','Rainfall 2015']]
subset.columns = subset.columns.str.replace(' 2015', '', regex=False)
sns.heatmap(subset.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Malawi Feature Correlations in 2015")
plt.show()


plt.figure(figsize=(6, 4))
subset = malawi[['Malaria Incidence 2015','Malaria Prevalence 2015','Mean Temperature 2015','ITN Coverage 2015','Land Surface Temperature 2015','UN Population Count 2015','U5 Population 2015','UN Population Density 2015','Wet Days 2015','Annual Precipitation 2015','Aridity 2015','Enhanced Vegetation Index 2015','Rainfall 2015']]
subset.columns = subset.columns.str.replace(' 2015', '', regex=False)
sns.heatmap(subset.corr(), annot=False, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5, linecolor='black')
plt.title("Malawi Feature Correlations in 2015")
plt.show()


# Bad data
sns.scatterplot(data=subset,x='ITN Coverage',y='Malaria Prevalence',hue='Wet Days',palette='Oranges')
plt.title("Malawi ITN Coverage and Malaria in 2015")
plt.xlabel("ITN Coverage")
plt.ylabel("Malaria Prevalence")
plt.show()


plt.figure(figsize=(4,3))
subset['Median'] = subset['Malaria Prevalence'] > subset['Malaria Prevalence'].median()
sns.boxplot(x='Median',y='Wet Days',data=subset)
plt.title("Malawi Wet Days in 2015")
plt.xticks([0, 1], ['Low Malaria', 'High Malaria'])
plt.show()


# ## Nigeria


lst = ['Enhanced Vegetation Index 2020','Malaria Incidence 2020', 'Malaria Prevalence 2020','ITN Coverage 2020','Rainfall 2020','Wet Days 2020','Aridity 2020','Precipitation 2020','U5 Population 2020','UN Population Count 2020', 'UN Population Density 2020','Land Surface Temperature 2020','Mean Temperature 2020']
lst.sort()
print(lst)


plt.figure(figsize=(9, 7))
subset = nigeria[['Rainfall 2020','Precipitation 2020','Aridity 2020','Wet Days 2020','Enhanced Vegetation Index 2020','Malaria Incidence 2020','Malaria Prevalence 2020','U5 Population 2020','UN Population Count 2020','UN Population Density 2020','ITN Coverage 2020','Land Surface Temperature 2020','Mean Temperature 2020']]
subset.columns = subset.columns.str.replace(' 2020', '', regex=False)
sns.heatmap(subset.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Nigeria Feature Correlations in 2020")
plt.show()


plt.figure(figsize=(6, 4))
subset = nigeria[['Rainfall 2020','Precipitation 2020','Aridity 2020','Wet Days 2020','Enhanced Vegetation Index 2020','Malaria Incidence 2020','Malaria Prevalence 2020','U5 Population 2020','UN Population Count 2020','UN Population Density 2020','ITN Coverage 2020','Land Surface Temperature 2020','Mean Temperature 2020']]
subset.columns = subset.columns.str.replace(' 2020', '', regex=False)
sns.heatmap(subset.corr(), annot=False, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5, linecolor='black')
plt.title("Nigeria Feature Correlations in 2020")
plt.show()


sns.scatterplot(data=subset,x='ITN Coverage',y='Malaria Prevalence',hue='Enhanced Vegetation Index',palette='Oranges')
plt.title("Nigeria ITN Coverage and Malaria in 2020")
plt.xlabel("ITN Coverage")
plt.ylabel("Malaria Prevalence")
plt.show()


plt.figure(figsize=(4,3))
subset['Median'] = subset['Malaria Prevalence'] > subset['Malaria Prevalence'].median()
sns.boxplot(x='Median',y='Enhanced Vegetation Index',data=subset)
plt.title("Nigeria EVI in 2020")
plt.xticks([0, 1], ['Low Malaria', 'High Malaria'])
plt.show()
