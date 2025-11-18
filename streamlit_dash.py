import streamlit as st
import xarray as xr
import rioxarray  # for CRS handling
import numpy as np
from geopy.geocoders import Nominatim
import plotly.express as px
import plotly.graph_objects as go

# set up page
st.set_page_config(page_title="Durham AT & RH Dashboard")#, layout="wide")

st.title("Durham Air Temperature & Relative Humidity Explorer")
st.write("Explore air temperature and relative humidity data for July–August 2024.")

netcdf_filepath =  r"D:\DDL\Durham_ML\Absolute\Clipped\Daily_Rasters\Multibands\Durham_AT_RH_JulAug2024.nc"
# load data
@st.cache_data
def load_data():
    ds = xr.open_dataset(netcdf_filepath)
    if not ds.rio.crs:
        ds = ds.rio.write_crs("EPSG:4326")  # ensure lat/lon CRS
    return ds

ds = load_data()

# --- USER INPUTS ---
var_choice = st.radio("Select variable to visualize:", ["Air Temperature", "Relative Humidity"])
var_key = "air_temperature" if "Temperature" in var_choice else "relative_humidity"
colorbar_label = "Air Temperature (°C)" if "Temperature" in var_choice else "Relative Humidity (%)"

time_options = ds.time.values
time_selected = st.select_slider("Select time:", options=time_options)

# --- GEOCODE LOCATION ---
st.subheader("🔍 Find value at a specific location")
user_location = st.text_input("Enter coordinates or address")

geolocator = Nominatim(user_agent="durham_dashboard", timeout = 10)
location = None
lat = lon = None
if user_location:
    try:
        location = geolocator.geocode(user_location)
        if location:
            lat, lon = location.latitude, location.longitude
            st.success(f"Found location: {location.address}")
            st.write(f"Latitude: {lat:.4f}, Longitude: {lon:.4f}")
        else:
            st.warning("Location not found.")
    except Exception as e:
        st.error(f"Geocoding failed: {e}")

# select data
da = ds[var_key].sel(time=time_selected)
# downsample raster to speed up plotting
downsample_factor = st.sidebar.number_input(
    "Downsample factor (higher = faster map, lower = more detail)",
    min_value=1, max_value=20, value=4, step=1
)
if downsample_factor > 1:
    #da_small = da.isel(x=slice(0, None, downsample_factor), y=slice(0, None, downsample_factor))
    da_small = da.coarsen(x=downsample_factor, y=downsample_factor, boundary='trim').mean()

else:
    da_small = da
# plot map
time_str = time_selected.strftime("%Y-%m-%d %H:%M")
st.subheader(f"{var_choice} at {time_str}")
fig = px.imshow(
    da_small,
    origin="lower",
    #labels={'x': 'Longitude', 'y': 'Latitude', 'color': var_key},
    labels={'x': '', 'y': '', 'color': colorbar_label},
    #title=f"{var_choice} for {time_str}",
    color_continuous_scale="Reds" if var_key == "air_temperature" else "Blues"
)

fig.update_layout(
    coloraxis_colorbar_len=0.4)


# stop map from becoming distorted
fig.update_yaxes(scaleanchor="x", scaleratio=1, visible = False,showgrid=False)
fig.update_xaxes(visible=False, showgrid=False)
fig.update_layout(title_text = None,
    height=500,
    margin=dict(l=0, r=0, t=0, b=0), autosize =True
)


# point for location
if lat and lon:
    fig.add_scatter(
        x=[lon], y=[lat],
        mode="markers+text",
        text=["Selected Location"],
        textposition="top right",
        marker=dict(color="black", size=10, symbol="x"),
         textfont=dict(color="black", size=14),
        name="Selected Location"
    )
    
    # --- GET VALUE AT POINT ---
    try:
        value = da.sel(y=lat, x=lon, method="nearest").values.item()
        st.metric(f"{var_choice} at {user_location}", f"{value:.2f}")
    except Exception:
        st.warning("Unable to extract value at this location (check if within map bounds).")

st.plotly_chart(fig,width="stretch")

# --- FOOTER ---
st.markdown("---")
st.caption("Data Driven Envirolab| https://datadrivenlab.org/")
