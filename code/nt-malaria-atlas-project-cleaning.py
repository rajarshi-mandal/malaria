import pandas as pd
import plotly.express as px


# Per unit of population
Pf_nat = pd.read_csv("malaria-data-for-modeling-dynamics/Malaria Atlas Project/Pf National.csv")
Pf_nat = Pf_nat.drop('Admin Level',axis=1)
Pf_nat_ir = Pf_nat[Pf_nat['Metric']=='Incidence Rate'].drop(['Metric','Units'],axis=1)
Pf_nat_ip = Pf_nat[Pf_nat['Metric']=='Infection Prevalence'].drop(['Metric','Units'],axis=1)
Pf_nat_mr = Pf_nat[Pf_nat['Metric']=='Mortality Rate'].drop(['Metric','Units'],axis=1)
# In total
Pf_nat_cnts = pd.read_csv("malaria-data-for-modeling-dynamics/Malaria Atlas Project/Pf National Counts.csv")
Pf_nat_cnts = Pf_nat_cnts.drop('Admin Level',axis=1)
Pf_nat_cnts_c = Pf_nat_cnts[Pf_nat_cnts['Metric']=='Clinical Cases'].drop(['Metric','Units'],axis=1)
Pf_nat_cnts_d = Pf_nat_cnts[Pf_nat_cnts['Metric']=='Deaths'].drop(['Metric','Units'],axis=1)
# Subnational data cannot be visualized with Plotly
Pf_subnat = pd.read_csv("malaria-data-for-modeling-dynamics/Malaria Atlas Project/Pf National.csv")
Pf_subnat = Pf_subnat.drop('Admin Level',axis=1)
Pf_subnat_cnts = pd.read_csv("malaria-data-for-modeling-dynamics/Malaria Atlas Project/Pf Subnational Counts.csv")
Pf_subnat_cnts = Pf_subnat_cnts.drop('Admin Level',axis=1)


#Pf_subnat
#Pf_subnat[Pf_subnat['Name']=='Mongala']['Value']
#Pf_subnat_cnts.describe()


# ## Country-Level Visualizations


for plt_year in [2010,2022]:
    fig = px.choropleth(Pf_nat_ir[Pf_nat_ir["Year"]==plt_year], locations="ISO3", color="Value", hover_name="Name", color_continuous_scale="Oranges", range_color=[Pf_nat_ir["Value"].min(), Pf_nat_ir["Value"].max()])
    fig.update_geos(showcountries=True, landcolor="#c0c0c0", scope="africa")
    fig.update_layout(
        title={'text': f"Incidence Rate (Cases per Thousand) in {plt_year}", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        coloraxis_colorbar={'title': {'text': 'Incidence Rate', 'font': {'size': 16}}, 'tickfont': {'size': 16}, 'ticklabelposition': 'outside', 'x': 0.95,},
        height=400, width=600, margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    fig.show()
    print("\n")


# 0.98 instead of 0.95
for plt_year in [2010,2022]:
    fig = px.choropleth(Pf_nat_ip[Pf_nat_ip["Year"]==plt_year], locations="ISO3", color="Value", hover_name="Name", color_continuous_scale="Oranges", range_color=[Pf_nat_ip["Value"].min(), Pf_nat_ip["Value"].max()])
    fig.update_geos(showcountries=True, landcolor="#c0c0c0", scope="africa")
    fig.update_layout(
        title={'text': f"Infection Prevalence (per 100 Children) in {plt_year}", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        coloraxis_colorbar={'title': {'text': 'Infection Prevalence', 'font': {'size': 16}}, 'tickfont': {'size': 16}, 'ticklabelposition': 'outside', 'x': 0.98},
        height=400, width=600, margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    fig.show()
    print("\n")


for plt_year in [2010,2022]:
    fig = px.choropleth(Pf_nat_mr[Pf_nat_mr["Year"]==plt_year], locations="ISO3", color="Value", hover_name="Name", color_continuous_scale="Oranges", range_color=[Pf_nat_mr["Value"].min(), Pf_nat_mr["Value"].max()])
    fig.update_geos(showcountries=True, landcolor="#c0c0c0", scope="africa")
    fig.update_layout(
        title={'text': f"Mortality Rate (Deaths per 100 Thousand) in {plt_year}", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        coloraxis_colorbar={'title': {'text': 'Mortality Rate', 'font': {'size': 16}}, 'tickfont': {'size': 16}, 'ticklabelposition': 'outside', 'x': 0.95,},
        height=400, width=600, margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    fig.show()
    print("\n")


for plt_year in [2010,2022]:
    fig = px.choropleth(Pf_nat_cnts_c[Pf_nat_cnts_c["Year"]==plt_year], locations="ISO3", color="Value", hover_name="Name", color_continuous_scale="Oranges", range_color=[Pf_nat_cnts_c["Value"].min(), Pf_nat_cnts_c["Value"].max()])
    fig.update_geos(showcountries=True, landcolor="#c0c0c0", scope="africa")
    fig.update_layout(
        title={'text': f"Total Clinical Cases in {plt_year}", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        coloraxis_colorbar={'title': {'text': 'Clinical Cases', 'font': {'size': 16}}, 'tickfont': {'size': 16}, 'ticklabelposition': 'outside', 'x': 0.95,},
        height=400, width=600, margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    fig.show()
    print("\n")


for plt_year in [2010,2022]:
    fig = px.choropleth(Pf_nat_cnts_d[Pf_nat_cnts_d["Year"]==plt_year], locations="ISO3", color="Value", hover_name="Name", color_continuous_scale="Oranges", range_color=[Pf_nat_cnts_d["Value"].min(), Pf_nat_cnts_d["Value"].max()])
    fig.update_geos(showcountries=True, landcolor="#c0c0c0", scope="africa")
    fig.update_layout(
        title={'text': f"Total Deaths in {plt_year}", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        coloraxis_colorbar={'title': {'text': 'Deaths', 'font': {'size': 16}}, 'tickfont': {'size': 16}, 'ticklabelposition': 'outside', 'x': 0.95,},
        height=400, width=600, margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    fig.show()
    print("\n")


# 0.25
fig = px.choropleth(Pf_nat_ir, locations="ISO3", color="Value", hover_name="Name", animation_frame="Year", color_continuous_scale="Oranges", range_color=[Pf_nat_ir["Value"].min(), Pf_nat_ir["Value"].max()])
fig.update_geos(showcountries=True, landcolor="#c0c0c0", scope="africa")
fig.update_layout(
    title={'text': "Incidence Rate (Cases per Thousand) Over Time", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 21}},
    coloraxis_colorbar={'title': {'text': 'Incidence Rate', 'font': {'size': 17}}, 'tickfont': {'size': 17}, 'ticklabelposition': 'outside', 'x': 0.95},
    height=600, width=800, margin={"r": 0, "t": 40, "l": 0, "b": 0}
)
fig.layout.updatemenus[0].x = 0.25
fig.layout.updatemenus[0].y = 0.13
fig.layout.sliders[0].x = 0.25
fig.layout.sliders[0].y = 0.13
fig.show()


fig = px.choropleth(Pf_nat_ip, locations="ISO3", color="Value", hover_name="Name", animation_frame="Year", color_continuous_scale="Oranges", range_color=[Pf_nat_ip["Value"].min(), Pf_nat_ip["Value"].max()])
fig.update_geos(showcountries=True, landcolor="#c0c0c0", scope="africa")
fig.update_layout(
    title={'text': f"Infection Prevalence (per 100 Children) Over Time", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 21}},
    coloraxis_colorbar={'title': {'text': 'Infection Prevalence', 'font': {'size': 17}}, 'tickfont': {'size': 17}, 'ticklabelposition': 'outside', 'x': 0.95,},
    height=600, width=800, margin={"r": 0, "t": 40, "l": 0, "b": 0}
)
fig.layout.updatemenus[0].x = 0.24
fig.layout.updatemenus[0].y = 0.13
fig.layout.sliders[0].x = 0.24
fig.layout.sliders[0].y = 0.13
fig.show()


fig = px.choropleth(Pf_nat_mr, locations="ISO3", color="Value", hover_name="Name", animation_frame="Year", color_continuous_scale="Oranges", range_color=[Pf_nat_mr["Value"].min(), Pf_nat_mr["Value"].max()])
fig.update_geos(showcountries=True, landcolor="#c0c0c0", scope="africa")
fig.update_layout(
    title={'text': f"Mortality Rate (Deaths per 100 Thousand) Over Time", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 21}},
    coloraxis_colorbar={'title': {'text': 'Mortality Rate', 'font': {'size': 17}}, 'tickfont': {'size': 17}, 'ticklabelposition': 'outside', 'x': 0.95,},
    height=600, width=800, margin={"r": 0, "t": 40, "l": 0, "b": 0}
)
fig.layout.updatemenus[0].x = 0.24
fig.layout.updatemenus[0].y = 0.13
fig.layout.sliders[0].x = 0.24
fig.layout.sliders[0].y = 0.13
fig.show()


fig = px.choropleth(Pf_nat_cnts_c, locations="ISO3", color="Value", hover_name="Name", animation_frame="Year", color_continuous_scale="Oranges", range_color=[Pf_nat_cnts_c["Value"].min(), Pf_nat_cnts_c["Value"].max()])
fig.update_geos(showcountries=True, landcolor="#c0c0c0", scope="africa")
fig.update_layout(
    title={'text': f"Clinical Cases Over Time", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 21}},
    coloraxis_colorbar={'title': {'text': 'Clinical Cases', 'font': {'size': 17}}, 'tickfont': {'size': 17}, 'ticklabelposition': 'outside', 'x': 0.95,},
    height=600, width=800, margin={"r": 0, "t": 40, "l": 0, "b": 0}
)
fig.layout.updatemenus[0].x = 0.23
fig.layout.updatemenus[0].y = 0.13
fig.layout.sliders[0].x = 0.23
fig.layout.sliders[0].y = 0.13
fig.show()


# 200, 0.47
fig = px.choropleth(Pf_nat_cnts_d, locations="ISO3", color="Value", hover_name="Name", animation_frame="Year", color_continuous_scale="Oranges", range_color=[Pf_nat_cnts_d["Value"].min(), Pf_nat_cnts_d["Value"].max()])
fig.update_geos(showcountries=True, landcolor="#c0c0c0", scope="africa")
fig.update_layout(
    title={'text': f"Deaths Over Time", 'x': 0.47, 'xanchor': 'center', 'font': {'size': 21}},
    coloraxis_colorbar={'title': {'text': 'Deaths', 'font': {'size': 17}}, 'tickfont': {'size': 17}, 'ticklabelposition': 'outside', 'x': 0.95},
    height=600, width=850, margin={"r": 200, "t": 40, "l": 0, "b": 0}
)
fig.layout.updatemenus[0].x = 0.23
fig.layout.updatemenus[0].y = 0.13
fig.layout.sliders[0].x = 0.23
fig.layout.sliders[0].y = 0.13
fig.show()
