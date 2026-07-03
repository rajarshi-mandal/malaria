import pandas as pd
import plotly.express as px
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pycountry
import warnings
warnings.filterwarnings("ignore")


def assign_year(row):
    if row['Start year'] == row['End year']:
        return row['Start year']
    elif row['End year'] - row['Start year'] == 1:
        return random.choice([row['Start year'], row['End year']])
    else:
        print('ERROR')


def get_iso3(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except:
        if country_name[-6:]=="Ivoire":
            return "CIV"
        elif country_name[-9:]=='the Congo':
            return "COD"
        else:
            print("ERROR: " + country_name)


ir = pd.read_csv("malaria-data-for-modeling-dynamics/IR Mapper (Anopheles)/1_standard-WHO-susc-test_complex-subgroup.csv", encoding='ISO-8859-1')
ir = ir[["Start year", "End year", "Insecticide tested", "Insecticide class", "Concentration (%)", "No. mosquitoes tested", "Percent mortality", "Country", "Latitude", "Longitude"]]
ir['Start year'] = pd.to_numeric(ir['Start year'], errors='coerce')
ir['End year'] = pd.to_numeric(ir['End year'], errors='coerce')
ir = ir[(ir['Start year'] >= 2010) & (ir['End year'] >= 2010)]
ir['Start year'] = ir['Start year'].astype(int)
ir['End year'] = ir['End year'].astype(int)
ir = ir.replace("NR", np.nan)
ir = ir.dropna(subset=["No. mosquitoes tested", "Percent mortality", "Latitude", "Longitude", "Concentration (%)"])
ir['Year'] = ir.apply(assign_year, axis=1)
ir = ir.drop(columns=['Start year', 'End year'])
ir = ir.sort_values(by="Year")
ir = ir[ir['Year'] != 2018]
year_order = list(range(2010, 2018))
ir['Year'] = pd.Categorical(ir['Year'], categories=year_order, ordered=True)
pyrethroid_ir = ir[ir['Insecticide class'] == 'pyrethroid']
ir['Concentration (%)'] = pd.to_numeric(ir['Concentration (%)'])
ir['No. mosquitoes tested'] = ir['No. mosquitoes tested'].astype(int)
ir_country = ir.groupby('Country', as_index=False)['Percent mortality'].mean()
ir_country['ISO3'] = ir_country['Country'].apply(get_iso3)
### DO NOT DELETE
### ir.to_csv('insecticide_resistance.csv',index=False)


plt.figure(figsize=(8, 6))
sns.stripplot(data=ir, x='Insecticide class', y='Percent mortality', hue='Year', dodge=True, size=3, alpha=0.5)
plt.title("Anopheles Mortality Distribution by Insecticide Class")
plt.xlabel("Insecticide Class")
plt.ylabel("Percent Mortality")
plt.xticks(rotation=0, ha='center')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


plt.figure(figsize=(6, 4.5))
sns.boxplot(data=pyrethroid_ir, x='Year', y='Percent mortality')
plt.title("Anopheles Mortality for Pyrethroid Insecticides by Year")
plt.xlabel("Year")
plt.ylabel("Percent Mortality")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


pyrethroid_ir.columns


plt.figure(figsize=(6, 4.5))
sns.boxplot(data=pyrethroid_ir, x='Insecticide tested', y='Percent mortality')
plt.title("Anopheles Mortality for various Pyrethroid Insecticides")
plt.xlabel("Year")
plt.ylabel("Percent Mortality")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(rotation=45)
plt.show()


pivot = ir.pivot_table(index='Insecticide class', columns='Year', values='Percent mortality', aggfunc='mean')
plt.figure(figsize=(7, 3.5))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Oranges", cbar_kws={'label': 'Mean Mortality Percent'})
plt.title("Insecticide Mortality Over Time")
plt.ylabel("Insecticide Class")
plt.xlabel("Year")
plt.show()


plt.figure(figsize=(5, 4))
sns.lineplot(data=ir, x="Year", y="Percent mortality", estimator='median', errorbar=('ci', 95), label="Median", marker='s')
sns.lineplot(data=ir, x="Year", y="Percent mortality", estimator='mean', errorbar=('ci', 95), label="Mean", marker='o')
plt.title("Anopheles Mortality Over Time")
plt.ylabel("Percent Mortality")
plt.tight_layout()
plt.show()


fig = px.choropleth(ir_country, locations="ISO3", color="Percent mortality", hover_name="Country", color_continuous_scale="Oranges", range_color=[ir_country["Percent mortality"].min(), ir_country["Percent mortality"].max()])
fig.update_geos(showcountries=True, landcolor="#c0c0c0", scope="africa")
fig.update_layout(
    title={'text': f"Average Anopheles Mortality Percent", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
    coloraxis_colorbar={'title': {'text': 'Mortality Percent', 'font': {'size': 16}}, 'tickfont': {'size': 16}, 'ticklabelposition': 'outside', 'x': 0.95},
    height=400, width=600, margin={"r": 0, "t": 40, "l": 0, "b": 0}
)
fig.show()


# Size based on number of mosquitoes in the study
for plt_year in [2010,2017]:
    fig = px.scatter_geo(data_frame=ir[ir["Year"]==plt_year], lat='Latitude', lon='Longitude', color='Percent mortality', size_max=7, size='No. mosquitoes tested', hover_name='Country', color_continuous_scale="Oranges", range_color=[ir["Percent mortality"].min(), ir["Percent mortality"].max()])
    fig.update_geos(showcountries=True, landcolor="#c0c0c0", scope="africa")
    fig.update_layout(
        title={'text': f"Anopheles Mortality Percent in {plt_year}", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        coloraxis_colorbar={'title': {'text': 'Mortality Percent', 'font': {'size': 16}}, 'tickfont': {'size': 16}, 'ticklabelposition': 'outside', 'x': 0.95,},
        height=400, width=600, margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.show()
    print("\n")


fig = px.scatter_geo(data_frame=ir, lat='Latitude', lon='Longitude', color='Percent mortality', animation_frame='Year', size_max=10, size='No. mosquitoes tested', hover_name='Country', color_continuous_scale="Oranges", range_color=[ir["Percent mortality"].min(), ir["Percent mortality"].max()])
fig.update_geos(showcountries=True, landcolor="#c0c0c0", scope="africa")
fig.update_layout(
    title={'text': f"Anopheles Mortality Percent Over Time", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 21}},
    coloraxis_colorbar={'title': {'text': 'Mortality Percent', 'font': {'size': 17}}, 'tickfont': {'size': 17}, 'ticklabelposition': 'outside', 'x': 0.95},
    height=600, width=800, margin={"r": 0, "t": 40, "l": 0, "b": 0}
)
fig.update_traces(marker=dict(line=dict(width=0)))
fig.layout.updatemenus[0].x = 0.23
fig.layout.updatemenus[0].y = 0.13
fig.layout.sliders[0].x = 0.23
fig.layout.sliders[0].y = 0.13
fig.show()
