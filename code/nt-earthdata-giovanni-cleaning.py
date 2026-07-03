import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


ast = pd.read_csv("malaria-data-for-modeling-dynamics/EarthData Giovanni/raw/Average Surface Temperature.csv")
ast.columns.values[-1] = "Average Surface Temperature"
csw = pd.read_csv("malaria-data-for-modeling-dynamics/EarthData Giovanni/raw/Canopy Surface Water.csv")
csw.columns.values[-1] = "Canopy Surface Water"
et = pd.read_csv("malaria-data-for-modeling-dynamics/EarthData Giovanni/raw/Evapotranspiration.csv")
et.columns.values[-1] = "Evapotranspiration"
nsat = pd.read_csv("malaria-data-for-modeling-dynamics/EarthData Giovanni/raw/Near Surface Air Temperature.csv")
nsat.columns.values[-1] = "Near Surface Air Temperature"
pp = pd.read_csv("malaria-data-for-modeling-dynamics/EarthData Giovanni/raw/Precipitation.csv")
pp.columns.values[-1] = "Precipitation"
sm = pd.read_csv("malaria-data-for-modeling-dynamics/EarthData Giovanni/raw/Soil Moisture (10 to 40 cm underground).csv")
sm.columns.values[-1] = "Soil Moisture (10 to 40 cm underground)"
srhd = pd.read_csv("malaria-data-for-modeling-dynamics/EarthData Giovanni/raw/Surface Relative Humidity (Daytime).csv")
srhd.columns.values[-1] = "Surface Relative Humidity (Daytime)"
srhn = pd.read_csv("malaria-data-for-modeling-dynamics/EarthData Giovanni/raw/Surface Relative Humidity (Nighttime).csv")
srhn.columns.values[-1] = "Surface Relative Humidity (Nighttime)"


merged = ast
datasets = [csw, et, nsat, pp, sm, srhd, srhn]
for df in datasets:
    merged = pd.merge(merged, df, on=["lat", "lon"], how="outer")
### DO NOT DELETE
### merged.to_csv('combined_data.csv', index=False)


#for variable in ['Average Surface Temperature', 'Canopy Surface Water', 'Evapotranspiration']:
#for variable in ['Near Surface Air Temperature', 'Precipitation', 'Soil Moisture (10 to 40 cm underground)']:
#for variable in ['Surface Relative Humidity (Daytime)', 'Surface Relative Humidity (Nighttime)']:
# Temporary
for variable in ['Average Surface Temperature']:
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=merged['lat'],lon=merged['lon'],mode='markers',marker=dict(size=8,color=merged[variable],colorscale='Oranges',
        colorbar=dict(title=dict(text=variable,side='right',font=dict(size=16)),tickfont=dict(size=14)),cmin=merged[variable].min(),cmax=merged[variable].max(),opacity=0.8,)))
    fig.update_geos(scope='africa',showland=True,landcolor='#c0c0c0',showcountries=True,countrycolor='black',countrywidth=3.5,)
    fig.update_layout(title=dict(text=variable,x=0.5,xanchor='center',font=dict(size=20)),height=600,width=800,margin=dict(r=0, t=40, l=0, b=0))
    fig.show()
