# ## Preprocessing


import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


colmap4 = {
    "HH1": "Cluster number",
    "HH2": "Household number",
    "TNLN": "Net number",
    "TN4": "Mosquito net observed",
    "TN5": "Brand/type of observed net",
    "TN6": "Months ago net obtained",
    "TN8": "Net treated with an insecticide when obtained",
    "TN9": "Net soaked or dipped since obtained",
    "TN10": "Months ago net soaked or dipped",
    "TN11": "Persons slept under mosquito net last night",
    "TN12_1": "Person 1 who slept under net",
    "TN12_2": "Person 2 who slept under net",
    "TN12_3": "Person 3 who slept under net",
    "TN12_4": "Person 4 who slept under net",
    "TN2": "Number of mosquito nets",
    "HH6": "Area",
    "HH7": "Region",
    "helevel": "Education of household head",
    "wscore": "Wealth index score",
    "windex5": "Wealth index quintiles",
    "PSU": "Primary sampling unit",
    "stratum": "Stratification group",
    "hhweight": "Household sample weight"
}


k_colmap4 = {
    "HH1": "Cluster number",
    "HH2": "Household number",
    "HH6": "Area",
    "HH7": "County",
    "TNLN": "Net number",
    "TN3": "Mosquito net observed",
    "TN4": "Months ago net obtained",
    "TN5": "Brand/type of observed net",
    "TN5A": "Source of mosquito net",
    "TN5B": "Amount paid for mosquito net",
    "TN7": "Net treated with insecticide when obtained",
    "TN8": "Net soaked or dipped since obtained",
    "TN9": "Months ago net soaked or dipped",
    "TN10": "Anyone slept under mosquito net last night",
    "TN11_1": "Person 1 who slept under net",
    "TN11_2": "Person 2 who slept under net",
    "TN11_3": "Person 3 who slept under net",
    "TN11_4": "Person 4 who slept under net",
    "TN2": "Number of mosquito nets",
    "helevel": "Education of household head",
    "religion": "Religion of household head",
    "hhweight": "Household sample weight",
    "wscore": "Wealth index score",
    "windex5": "Wealth index quintiles"
}


colmap5 = {
    "HH1": "Cluster number",
    "HH2": "Household number",
    "TNLN": "Net number",
    "TN4": "Mosquito net observed",
    "TN5": "Brand/type of observed net",
    "TN6": "Months ago net obtained",
    "TN8": "Net treated with an insecticide when obtained",
    "TN9": "Net soaked or dipped since obtained",
    "TN10": "Months ago net soaked or dipped",
    "TN11": "Persons slept under mosquito net last night",
    "TN12_1": "Person 1 who slept under net",
    "TN12_2": "Person 2 who slept under net",
    "TN12_3": "Person 3 who slept under net",
    "TN12_4": "Person 4 who slept under net",
    "TN12_5": "Person 5 who slept under net",
    "TN12_6": "Person 6 who slept under net",
    "TN2": "Number of mosquito nets",
    "HH6": "Area",
    "HH7": "District",
    "helevel": "Education of household head",
    "ethnicity": "Ethnicity",
    "religion": "Religion",
    "region": "Region",
    "hhweight": "Household sample weight",
    "wscore": "Combined wealth score",
    "windex5": "Wealth index quintile",
    "wscoreu": "Urban wealth score",
    "windex5u": "Urban wealth index quintile",
    "wscorer": "Rural wealth score",
    "windex5r": "Rural wealth index quintile"
}


colmap6 = {
    "HH1": "Cluster number",
    "HH2": "Household number",
    "TNLN": "Net number",
    "TN3": "Mosquito net observed",
    "TN4": "Months ago net obtained",
    "TN12": "Place where net was obtained",
    "TN13": "Persons slept under mosquito net last night",
    "TN15_1": "Person 1 who slept under net",
    "TN15_2": "Person 2 who slept under net",
    "TN15_3": "Person 3 who slept under net",
    "TN15_4": "Person 4 who slept under net",
    "TN15_5": "Person 5 who slept under net",
    "TN15_6": "Person 6 who slept under net",
    "TN15_7": "Person 7 who slept under net",
    "TN15_8": "Person 8 who slept under net",
    "TN15_9": "Person 9 who slept under net",
    "TN15_10": "Person 10 who slept under net",
    "TN1": "Household has mosquito nets",
    "TN2": "Number of mosquito nets",
    "TN5": "Brand/type of observed net",
    "TN10": "Way net was obtained",
    "HH3": "Interviewer Number",
    "HH4": "Supervisor number",
    "HH6": "Area",
    "HH7": "District/Island Group",
    "hhweight": "Household sample weight",
    "religion": "Religion",
    "wscore": "Combined wealth score",
    "windex5": "Wealth index quintile",
    "windex10": "Wealth index decile",
    "wscoreu": "Urban wealth score",
    "windex5u": "Urban wealth index quintile",
    "windex10u": "Urban wealth index decile",
    "wscorer": "Rural wealth score",
    "windex5r": "Rural wealth index quintile",
    "windex10r": "Rural wealth index decile",
    "helevel": "Education of household head",
    "ethnicity": "Ethnicity of household head",
    "PSU": "Primary sampling unit",
    "psu": "Primary sampling unit",
    "stratum": "Stratification group"
}


def urban_rural(data):
    data['Area'] = data['Area'].replace({1: 'Urban', 2: 'Rural'})


def urban_rural_graph(country_data,country_name):
    urban_rural(country_data)
    wiq_col = [col for col in country_data.columns if 'quintile' in col][0]
    columns_needed = ['Area', wiq_col, 'Number of mosquito nets']
    country_hm = country_data[columns_needed].dropna().pivot_table(index='Area',columns=wiq_col,values='Number of mosquito nets',aggfunc='mean')
    plt.figure(figsize=(6, 3))
    sns.heatmap(country_hm, annot=True, cmap='Oranges', fmt=".2f")
    plt.title(f"Mean ITNs per Protected Household in {country_name}")
    plt.ylabel('Area')
    plt.xlabel('Wealth Index Quintile (5 is richest)')
    plt.tight_layout()
    plt.show()


def itn_age(country_data,country_name):
    # Prepare and clean the relevant column
    country_ad = country_data[['Months ago net obtained']].copy().dropna()
    country_ad = country_ad[pd.to_numeric(country_ad['Months ago net obtained'], errors='coerce') < 95]
    country_ad['Months ago net obtained'] = pd.to_numeric(country_ad['Months ago net obtained'], errors='coerce').dropna()
    # KDE plot of overall net age distribution
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=country_ad, x='Months ago net obtained', fill=True, color='steelblue')
    plt.title(f'{country_name} Net Age Distribution (≤ 3 years)')
    plt.xlabel('Months Ago Net Was Obtained')
    plt.ylabel('Probability')
    plt.show()


def visual(country_data,country_name):
    urban_rural_graph(country_data,country_name)
    itn_age(country_data,country_name)


# Round 4
eswatini4 = pd.read_csv("malaria-data-for-modeling-dynamics/Multiple Indicator Cluster Surveys/4 Eswatini.csv")
kenya_nyanza4 = pd.read_csv("malaria-data-for-modeling-dynamics/Multiple Indicator Cluster Surveys/4 Kenya (Nyanza Province).csv")
madagascar4 = pd.read_csv("malaria-data-for-modeling-dynamics/Multiple Indicator Cluster Surveys/4 Madagascar.csv")
somalia_northeast4 = pd.read_csv("malaria-data-for-modeling-dynamics/Multiple Indicator Cluster Surveys/4 Somalia (Northeast Zone).csv")
somalia_somaliland4 = pd.read_csv("malaria-data-for-modeling-dynamics/Multiple Indicator Cluster Surveys/4 Somalia (Somaliland).csv")
eswatini4 = eswatini4.rename(columns=colmap4)
kenya_nyanza4 = kenya_nyanza4.rename(columns=k_colmap4)
madagascar4 = madagascar4.rename(columns=colmap4)
somalia_northeast4 = somalia_northeast4.rename(columns=colmap4)
somalia_somaliland4 = somalia_somaliland4.rename(columns=colmap4)


# Round 5
kenya_bungoma5 = pd.read_csv("malaria-data-for-modeling-dynamics/Multiple Indicator Cluster Surveys/5 Kenya (Bungoma County).csv")
kenya_kakamega5 = pd.read_csv("malaria-data-for-modeling-dynamics/Multiple Indicator Cluster Surveys/5 Kenya (Kakamega County).csv")
kenya_turkana5 = pd.read_csv("malaria-data-for-modeling-dynamics/Multiple Indicator Cluster Surveys/5 Kenya (Turkana County).csv")
malawi5 = pd.read_csv("malaria-data-for-modeling-dynamics/Multiple Indicator Cluster Surveys/5 Malawi.csv")
zimbabwe5 = pd.read_csv("malaria-data-for-modeling-dynamics/Multiple Indicator Cluster Surveys/5 Zimbabwe.csv")
kenya_bungoma5 = kenya_bungoma5.rename(columns=colmap5)
kenya_kakamega5 = kenya_kakamega5.rename(columns=colmap5)
kenya_turkana5 = kenya_turkana5.rename(columns=colmap5)
malawi5 = malawi5.rename(columns=colmap5)
zimbabwe5 = zimbabwe5.rename(columns=colmap5)


# Round 6
comoros6 = pd.read_csv("malaria-data-for-modeling-dynamics/Multiple Indicator Cluster Surveys/6 Comoros.csv")
comoros6 = comoros6.drop(["HH7a", "HH7Aux", "HH7Aux1", "IncProb_hh_mean", "stratum_SE", "PSU_SE"],axis=1)
madagascar6 = pd.read_csv("malaria-data-for-modeling-dynamics/Multiple Indicator Cluster Surveys/6 Madagascar.csv")
malawi6 = pd.read_csv("malaria-data-for-modeling-dynamics/Multiple Indicator Cluster Surveys/6 Malawi.csv")
zimbabwe6 = pd.read_csv("malaria-data-for-modeling-dynamics/Multiple Indicator Cluster Surveys/6 Zimbabwe.csv")
comoros6 = comoros6.rename(columns=colmap6)
madagascar6 = madagascar6.rename(columns=colmap6)
malawi6 = malawi6.rename(columns=colmap6)
zimbabwe6 = zimbabwe6.rename(columns=colmap6)


# ## Visualizations


visual(eswatini4,'Eswatini')


visual(kenya_nyanza4,'Kenya Nyanza')


visual(madagascar4,'Madagascar')


visual(somalia_northeast4,'Somalia Northeast')


visual(somalia_somaliland4,'Somalia Somaliland')


visual(kenya_bungoma5,'Kenya Bungoma')


visual(kenya_kakamega5,'Kenya Kakamega')


visual(kenya_turkana5,'Kenya Turkana')


visual(malawi5,'Malawi')


visual(zimbabwe5,'Zimbabwe')


visual(comoros6,'Comoros')


visual(madagascar6,'Madagascar')


visual(malawi6,'Malawi')


visual(zimbabwe6,'Zimbabwe')
