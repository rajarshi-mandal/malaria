import pandas as pd
import plotly.express as px


itn_access = pd.read_csv("malaria-data-for-modeling-dynamics/The Global Health Observatory/ITN Access.csv")
itn_access = itn_access.sort_values(by="Period", ascending=True)
sus_malaria = pd.read_csv("malaria-data-for-modeling-dynamics/The Global Health Observatory/Malaria cases, suspected, number.csv")
sus_malaria['Value'] = sus_malaria['Value'].str.replace(' ', '').astype(int)
sus_malaria = sus_malaria[sus_malaria['Location'] != "India"]
sus_malaria = sus_malaria.sort_values(by='Period', ascending=True)


for plt_year in [2015,2023]:
    fig = px.choropleth(itn_access[itn_access["Period"]==plt_year], locations="SpatialDimValueCode", color="Value", hover_name="Location", color_continuous_scale="Oranges", range_color=[itn_access["Value"].min(), itn_access["Value"].max()])
    fig.update_geos(showcountries=True, landcolor="#c0c0c0", scope="africa")
    fig.update_layout(
        title={'text': f"ITN Access Percentage in {plt_year}", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        coloraxis_colorbar={'title': {'text': 'Percent', 'font': {'size': 16}}, 'tickfont': {'size': 16}, 'ticklabelposition': 'outside', 'x': 0.95},
        height=400, width=600, margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    fig.show()
    print("\n")


for plt_year in [2015,2023]:
    fig = px.choropleth(sus_malaria[sus_malaria["Period"]==plt_year], locations="SpatialDimValueCode", color="Value", hover_name="Location", color_continuous_scale="Oranges", range_color=[sus_malaria["Value"].min(), sus_malaria["Value"].max()])
    fig.update_geos(showcountries=True, landcolor="#c0c0c0", scope="africa")
    fig.update_layout(
        title={'text': f"Suspected Malaria Cases in {plt_year}", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        coloraxis_colorbar={'title': {'text': 'Number of Cases', 'font': {'size': 16}}, 'tickfont': {'size': 16}, 'ticklabelposition': 'outside', 'x': 0.95},
        height=400, width=600, margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    fig.show()
    print("\n")


fig = px.choropleth(itn_access, locations="SpatialDimValueCode", color="Value", hover_name="Location", animation_frame="Period", color_continuous_scale="Oranges", range_color=[itn_access["Value"].min(), itn_access["Value"].max()])
fig.update_geos(showcountries=True, landcolor="#c0c0c0", scope="africa")
fig.update_layout(
    title={'text': "ITN Access Percentage Over Time", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 21}},
    coloraxis_colorbar={'title': {'text': 'Percent', 'font': {'size': 17}}, 'tickfont': {'size': 17}, 'ticklabelposition': 'outside', 'x': 1},
    height=600, width=700, margin={"r": 0, "t": 40, "l": 0, "b": 0}
)
fig.layout.updatemenus[0].x = 0.23
fig.layout.updatemenus[0].y = 0.13
fig.layout.sliders[0].x = 0.23
fig.layout.sliders[0].y = 0.13
fig.show()


# 0.25 120
fig = px.choropleth(sus_malaria, locations="SpatialDimValueCode", color="Value", hover_name="Location", animation_frame="Period", color_continuous_scale="Oranges", range_color=[sus_malaria["Value"].min(), sus_malaria["Value"].max()])
fig.update_geos(showcountries=True, landcolor="#c0c0c0", scope="africa")
fig.update_layout(
    title={'text': "Suspected Malaria Cases Over Time", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 21}},
    coloraxis_colorbar={'title': {'text': 'Number of Cases', 'font': {'size': 17}}, 'tickfont': {'size': 17}, 'ticklabelposition': 'outside', 'x': 0.95},
    height=600, width=800, margin={"r": 120, "t": 40, "l": 0, "b": 0}
)
fig.layout.updatemenus[0].x = 0.25
fig.layout.updatemenus[0].y = 0.13
fig.layout.sliders[0].x = 0.25
fig.layout.sliders[0].y = 0.13
fig.show()
