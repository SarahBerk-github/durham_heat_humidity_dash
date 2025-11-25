import streamlit as st
import xarray as xr
import rioxarray  # for CRS handling
import numpy as np
from geopy.geocoders import Nominatim
import plotly.express as px
import plotly.graph_objects as go

# set up page
st.set_page_config(page_title="Durham AT & RH Dashboard")#, layout="wide")

st.title("Durham Heat Stress Explorer")
st.write("Explore air temperature, relative humidity and heat index data for July–August 2024.")

netcdf_filepath =  r"D:\DDL\Durham_ML\Absolute\Clipped\Daily_Rasters\Multibands\Durham_AT_RH_JulAug2024.nc"
# load data
@st.cache_data
def load_data():
    ds = xr.open_dataset(netcdf_filepath)
    if not ds.rio.crs:
        ds = ds.rio.write_crs("EPSG:4326")  # ensure lat/lon CRS
    return ds

ds = load_data()

# function for calculating heat index 
def compute_heat_index(T, R):
    HI = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + R * 0.094) # steadman
    mask = HI >= 80 #use roth for heat index over 80
    HI_roth = (-42.379 + 2.04901523 * T + 10.14333127 * R
               - 0.22475541 * T * R - 6.83783e-3 * T**2
               - 5.481717e-2 * R**2 + 1.22874e-3 * T**2 * R
               + 8.5282e-4 * T * R**2 - 1.99e-6 * T**2 * R**2)
    HI = xr.where(mask, HI_roth, HI)
    # adjustments, in line with NOAA
    adj1_mask = (R < 13) & (T >= 80) & (T <= 112)
    sqrt_arg = (17 - np.abs(T - 95.0)) / 17.0
    sqrt_arg = xr.where(sqrt_arg < 0, 0, sqrt_arg)
    adj1 = ((13 - R) / 4.0) * np.sqrt(sqrt_arg)
    HI = xr.where(adj1_mask, HI - adj1, HI)

    adj2_mask = (R > 85) & (T >= 80) & (T <= 87)
    adj2 = ((R - 85) / 10.0) * ((87 - T) / 5.0)
    HI = xr.where(adj2_mask, HI + adj2, HI)
    return HI

# function for catagoriing the heat index 
def categorise_heat_index(hi):
    hi_cat = xr.full_like(hi, "", dtype="object")
    hi_cat = xr.where(hi < 79.5, "No Warning", hi_cat)
    hi_cat = xr.where((hi >= 79.5) & (hi <= 89.5), "Caution", hi_cat)
    hi_cat = xr.where((hi > 89.5) & (hi <= 104.5), "Extreme Caution", hi_cat)
    hi_cat = xr.where((hi > 104.5) & (hi <= 129.5), "Danger", hi_cat)  
    hi_cat = xr.where(hi > 129.5, "Extreme Danger", hi_cat)
    return hi_cat

# colors for the categories
hi_colors = {
    "Extreme Danger": "darkred",
    "Danger": "red",
    "Extreme Caution": "orange",
    "Caution": "yellow",
    "No Warning": "lightgreen"
}

hi_levels = {
    "Extreme Danger": 1.00,
    "Danger": 0.75,
    "Extreme Caution": 0.50,
    "Caution": 0.25,
    "No Warning": 0.00,
}

def convert_to_numeric(hi_cat):
    return xr.apply_ufunc(
        lambda x: hi_levels.get(x, np.nan),
        hi_cat,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float] )

hi_colorscale = [
    [0.00, "lightgreen"],   # No warning
    [0.25, "yellow"],       # Caution
    [0.50, "orange"],       # Extreme Caution
    [0.75, "red"],          # Danger
    [1.00, "darkred"],      # Extreme Danger
]

# user inputs
var_choice = st.radio("Select variable to visualize:", ["Air Temperature", "Relative Humidity","Heat Index"])

if var_choice == "Air Temperature":
    var_key = "air_temperature"
    colorbar_label = "Air Temperature (°F)"
    cmap_color = 'Reds'
elif var_choice == "Relative Humidity":
    var_key = "relative_humidity"
    colorbar_label = "Relative Humidity (%)"
    cmap_color = 'Blues'
else:  
    var_key = "heat_index"
    colorbar_label = "Heat Index (°F)"
    cmap_color = 'Oranges'

units = "%" if "Humidity" in var_choice else "°F"
time_options = ds.time.values
time_selected = st.select_slider("Select time:", options=time_options)

# selecting location
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
            st.warning("Location not found")
    except Exception as e:
        st.error(f"Geocoding failed: {e}")

if var_choice in ["Air Temperature", "Relative Humidity"]:
    da = ds[var_key].sel(time=time_selected)

    # Convert temperature to Fahrenheit
    if var_choice == "Air Temperature":
        da = da * 9/5 + 32

else:  # Heat Index
    T = ds["air_temperature"].sel(time=time_selected)
    R = ds["relative_humidity"].sel(time=time_selected)

    # Convert temperature to Fahrenheit
    T_f = T * 9/5 + 32

    da = compute_heat_index(T_f, R)
    #hi_cat = xr.apply_ufunc(
    #    np.vectorize(categorise_heat_index_scalar),
    #    da,
     #   vectorize=True,
     #   dask="parallelized",
     #   output_dtypes=[str]
     #   )

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

if var_choice == "Heat Index":

    hi_cat = categorise_heat_index(da)
    hi_numeric = convert_to_numeric(hi_cat)

    #fig = px.imshow(
    #    hi_numeric,
    #    origin="lower",
        #labels={"x": "", "y": "", "color": "Heat Index Category"},
    #    color_continuous_scale=hi_colorscale,
    #    labels={'color': 'Heat Index Category'})

    # update the hover with the warning category 
   # fig.update_traces(
   #     hovertemplate="Warning Level: %{customdata}",
   #     customdata=hi_cat.values)
    # remove the continuous colorbar to replace with categorical key
  #  fig.update_coloraxes(showscale=False)

    # Add dummy scatter points for legend/key
   # for level, color in hi_colors.items():
  #      fig.add_trace(
  #          go.Scatter(
  #              x=[None], y=[None],
  #              mode="markers",
   #             marker=dict(size=10, color=color),
   #             name=level
   #         )
   #     )

    fig = px.imshow(
        da,
        origin="lower",
        #labels={'x': 'Longitude', 'y': 'Latitude', 'color': var_key},
        labels={'x': '', 'y': '', 'color': colorbar_label},
        #title=f"{var_choice} for {time_str}",
        color_continuous_scale="Reds"  )

else:

    fig = px.imshow(
        da_small,
        origin="lower",
        #labels={'x': 'Longitude', 'y': 'Latitude', 'color': var_key},
        labels={'x': '', 'y': '', 'color': colorbar_label},
        #title=f"{var_choice} for {time_str}",
        color_continuous_scale="Reds" if var_key == "air_temperature" else "Blues" )

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
    
    #get a value at each point 
    try:
        value = da.sel(y=lat, x=lon, method="nearest").values.item()
        if var_choice in ["Air Temperature", "Relative Humidity"]:
            st.metric(f"{var_choice} at {user_location}", f"{value:.1f} {units}")
        else: 
            hi_cat_single = categorise_heat_index(xr.DataArray(value)).item()
            st.metric(f"{var_choice} at {user_location}", f"{value:.1f} {units} - Warning Level: {hi_cat_single}")
    except Exception:
        st.warning("Unable to extract value at this location, check if within city bounds")

st.plotly_chart(fig,width="stretch")

#footer
st.markdown("---")
st.caption("Data Driven Envirolab| https://datadrivenlab.org/")
