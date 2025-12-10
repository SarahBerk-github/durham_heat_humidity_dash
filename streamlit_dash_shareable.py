## code for the steamlit dash to be made sharable ##

import gdown
import os
import streamlit as st
import xarray as xr
import rioxarray
import numpy as np
#from geopy.geocoders import Nominatim   # not using this anymore as streamlit doesn't allow
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64
import pandas as pd
import requests


file_links = ["1R-lSasJ6PI2GZZqbM30-PRUXT6ZF6itx",
"1naKjt4s8-16x0uJvz1UxiBnMYsuHDWNH",
"1jE-T4AhVPZVVY6df5Wvi8DT090PGQuOI",
"12ANvCDt0WGg_Jvs_DGzfWP1zvOfrs27r",
"1br7k_bGpy8-yJBxiHPwy6AZ_xulHsQkJ",
"1MG3EGNlEZAZr4ucxsaUYtsHC3q9sFH0D",
"1ho84zLzT939gDlktdy2uLKtuvG_WiFA-",
"1mR3_tqmYvxfU9I5aWa_7j5-eQfxhFbU6",
"1ufrgKKkVbYMGb4m0R6IKtO7xZko_473B",
"1ThWhE1Q4UVyWXXD73eW1Sf4NhwFISX_C",
"1TOUZN3zxfhB38y571_hQ-WNzYZFpnbxU",
"1bXtJDoUZHvjYLQByFLAzZN9xd0eT9MYd",
"1OT2Is4mLH_Cpr4ztvMnyaVmVCDPrEwjj",
"1EMJBUejsCbkjJqAxtla1fTQk4IHst0xz",
"1fvYgvWuQsyL20ftf18vTFr0K2P7dDbTG",
"1W09VKMyqKhJLLstL1oijTnr7kszgfh5a",
"1y0PiZ9u7-lBC_rDviJ6aAfw7qrWXHCc3",
"1XYs_26trNHRLW2S91vgc1PelOE0MLDsc",
"1bWipSavKiXEL-YVCWUBM90tHC8FAXdmo",
"10Kwi9XxYgtTuM_vxCVSMsS2-u5aRyFhY",
"1ZylSEldHUW3e-Akesrf_Vre3RHiudY1f",
"1KjWcwBvB2X-w1pH4u8vGXFLugo6U2QVG",
"1KBcRNwFGsSI1WrbB1-9RVHGEJI71jL2j",
"1q_fXNFfuE-Pz-iLGMmoQ-ZHaTOovg8n1",
"1XxGeXxNN8KnTnIEnHFLodqYL5wv15aH-",
"17yoX4m4zwRdIMk__VPpu8STvbQtE6AEB",
"1Y-3wUMwxKcV-q5Oj0OQIvRSPKupSL1WQ",
"1g53v9pWztFJdF02wp8h2bX_WY7OIRElR",
"1oT3csOl6iwcGr3hCIXNNoeIDkDn0N6VA",
"1pUzemvx_tx6dqoSdOMdrCxTJIhDcdnC_",
"1lsnM8n0VNLnT_dy53gOYgVjKSz50cwe-",
"1vzvRQkfj3E_id6oZo3SpBvFXXqDmmZD3",
"1Oaweujm-RJT_ajxJKQh-6-yT3GtNHC-5",
"1H-NHswuOvwXuNWi2HStZFVvTd7dmHZcP",
"1SQDayaWFe6DnRyZ8RHpJzay_DaumcpLr",
"1O-kIxe3jY3soirROze7x6oIu9Yhorpzy",
"1ExtZ__dHyNP3gWmxRnHLgJr3yw8RpTDg",
"1nIcxSlpP6l7UzR0pvr0lUpryY7IbPFmz",
"1B6q9qXf04sMzvUKdcsuoMxbSpNgRFivr",
"1w9Hu4vHHg_S0EW3G8vuS3THiIbzTRNbi",
"1C99MibKPvnK3LKSlAOP-9TCGWzgQxM-e",
"15x0Zc2Y8GC9uV9bYAOkO5JpXhqvLIg3I",
"14N9kXlX9HrMVUG08sG9aJqi7-h7DsgBO",
"1krJRxTGxAqQqy34DLkU_sycml3ZTKBI6",
"1To5KU8cnA4LwcShSbfc3VqC3JEtSMz_Y",
"1y4xL4g0S4UDA-jFFNR1eXJOebcIUsD9w",
"17SDktuaOxgrr_eYPV_KYN_e8vee8bh2i",
"1tr9P78BFvcIZcd3opNk4YETTJkeZM7fe",
"1Ym2WBLH-AiE418TffYOD3FZHdREkP1VN",
"1H0A6R5DaPtanQYTxdC00sA1rF7z5Px2a",
"1HKnXK0Jjgc-XUoav_apYHvOh_hr_HH04",
"1PQRkLjFG6zC_Vkk8JFTOLuaGrr-BB702",
"1qvlrGGTDlau_NVi-rL4fAuEb8qmsVVIr",
"1kATOYGcUVqcrryqVPpzk6j8Vn1JO7J65",
"1VP9kUKBtWCv9121RQnBUe-v5fGo5TDOr",
"1xzMb1m9zOxZEybYD04P4C8EADABKJoP9",
"1QVw6WLUVEnXGrcCZKLghlOpFxIKXK_br",
"1w3E0CCoJZLUoRzuONCYyAX3SCEgk_olU",
"16KCWMZziCtYTnaPFtflKP17A7XM6Te9b",
"1abZaM_i88_KxyUau0yMvjl51OqvmXDQC",
"1a3JNtGNUc1EGkxN1r8939EkOKBzrZOBy",
"1asoRQmCIiIE-NPYqPpwPXnSafL4VCc0H"]

filenames = ["Durham_AT_RH_20240701_20240701.nc",
"Durham_AT_RH_20240831_20240831.nc",
"Durham_AT_RH_20240830_20240830.nc",
"Durham_AT_RH_20240829_20240829.nc",
"Durham_AT_RH_20240828_20240828.nc",
"Durham_AT_RH_20240827_20240827.nc",
"Durham_AT_RH_20240826_20240826.nc",
"Durham_AT_RH_20240825_20240825.nc",
"Durham_AT_RH_20240824_20240824.nc",
"Durham_AT_RH_20240823_20240823.nc",
"Durham_AT_RH_20240822_20240822.nc",
"Durham_AT_RH_20240821_20240821.nc",
"Durham_AT_RH_20240820_20240820.nc",
"Durham_AT_RH_20240819_20240819.nc",
"Durham_AT_RH_20240818_20240818.nc",
"Durham_AT_RH_20240817_20240817.nc",
"Durham_AT_RH_20240816_20240816.nc",
"Durham_AT_RH_20240815_20240815.nc",
"Durham_AT_RH_20240814_20240814.nc",
"Durham_AT_RH_20240813_20240813.nc",
"Durham_AT_RH_20240812_20240812.nc",
"Durham_AT_RH_20240811_20240811.nc",
"Durham_AT_RH_20240810_20240810.nc",
"Durham_AT_RH_20240809_20240809.nc",
"Durham_AT_RH_20240808_20240808.nc",
"Durham_AT_RH_20240807_20240807.nc",
"Durham_AT_RH_20240806_20240806.nc",
"Durham_AT_RH_20240805_20240805.nc",
"Durham_AT_RH_20240804_20240804.nc",
"Durham_AT_RH_20240803_20240803.nc",
"Durham_AT_RH_20240802_20240802.nc",
"Durham_AT_RH_20240801_20240801.nc",
"Durham_AT_RH_20240731_20240731.nc",
"Durham_AT_RH_20240730_20240730.nc",
"Durham_AT_RH_20240729_20240729.nc",
"Durham_AT_RH_20240728_20240728.nc",
"Durham_AT_RH_20240727_20240727.nc",
"Durham_AT_RH_20240726_20240726.nc",
"Durham_AT_RH_20240725_20240725.nc",
"Durham_AT_RH_20240724_20240724.nc",
"Durham_AT_RH_20240723_20240723.nc",
"Durham_AT_RH_20240722_20240722.nc",
"Durham_AT_RH_20240721_20240721.nc",
"Durham_AT_RH_20240720_20240720.nc",
"Durham_AT_RH_20240719_20240719.nc",
"Durham_AT_RH_20240718_20240718.nc",
"Durham_AT_RH_20240717_20240717.nc",
"Durham_AT_RH_20240716_20240716.nc",
"Durham_AT_RH_20240715_20240715.nc",
"Durham_AT_RH_20240714_20240714.nc",
"Durham_AT_RH_20240713_20240713.nc",
"Durham_AT_RH_20240712_20240712.nc",
"Durham_AT_RH_20240711_20240711.nc",
"Durham_AT_RH_20240710_20240710.nc",
"Durham_AT_RH_20240709_20240709.nc",
"Durham_AT_RH_20240708_20240708.nc",
"Durham_AT_RH_20240707_20240707.nc",
"Durham_AT_RH_20240706_20240706.nc",
"Durham_AT_RH_20240705_20240705.nc",
"Durham_AT_RH_20240704_20240704.nc",
"Durham_AT_RH_20240703_20240703.nc",
"Durham_AT_RH_20240702_20240702.nc"]

FILE_MAP = dict(zip(filenames, file_links))

# Durham bounding box 
lon_min = -79.00747291257709
lon_max = -78.75454745095128
lat_min = 35.86640498625444
lat_max = 36.13696655321311


#CITY_MEAN_FILE_ID = "1fIpNj2z8Krhhx2IPkht1zBWrXRzisroZ"
#CITY_MEAN_PATH = os.path.join("data_cache", "city_mean_hourly.csv")
#if not os.path.exists(CITY_MEAN_PATH):
#    gdown.download(f"https://drive.google.com/uc?id={CITY_MEAN_FILE_ID}", CITY_MEAN_PATH, quiet=False)

city_mean_df = pd.read_csv("city_means_hourly.csv", parse_dates=["time"])
city_mean_df.set_index("time", inplace=True)

# function for downloading the files
def download_file(filename):
    file_id = FILE_MAP[filename]
    output_path = os.path.join("data_cache", filename)
    os.makedirs("data_cache", exist_ok=True)

    if not os.path.exists(output_path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

    return output_path

   
st.set_page_config(page_title="Durham AT & RH Dashboard")
st.title("Durham Heat Stress Explorer")
st.write("Explore air temperature, relative humidity and heat index data for July–August 2024. Search for a location to see its timeseries and compare it to the city mean.")

#selected_date = st.date_input("Select Date", value=pd.to_datetime("2024-07-01"))
# restrict the dates the calendar allows users to select 
min_date = pd.to_datetime("2024-07-01")
max_date = pd.to_datetime("2024-08-31")

selected_date = st.date_input(
    "Select Date",
    value=min_date,       # default selection
    min_value=min_date,   
    max_value=max_date    
)

date_str = selected_date.strftime("%Y%m%d")

# clear the cache if the date has changed, stops the system from crashing 
if "last_selected_date" not in st.session_state:
    st.session_state.last_selected_date = selected_date

if selected_date != st.session_state.last_selected_date:
    st.cache_data.clear()
    st.session_state.last_selected_date = selected_date


# Find filename containing the date
matching = [f for f in filenames if date_str in f]
if not matching:
    st.error("No file found for this date.")
    st.stop()

selected_filename = matching[0]
netcdf_filepath = download_file(selected_filename)

# Load data
@st.cache_data
def load_data(path):
    ds = xr.open_dataset(path)#, chunks = 'auto') # having issues with dask import 
    if not ds.rio.crs:
        ds = ds.rio.write_crs("EPSG:4326")
    return ds

#st.write("NetCDF exists?", os.path.exists(netcdf_filepath))
#st.write("File size:", os.path.getsize(netcdf_filepath) if os.path.exists(netcdf_filepath) else "N/A")
# clear the old net cdf from the cache, but not the csv of city means
#if "last_selected_file" not in st.session_state:
#    st.session_state.last_selected_file = None

#if st.session_state.last_selected_file != selected_filename:
    # remove cached NetCDF for previous file
#    load_data.clear()
#    st.session_state.last_selected_file = selected_filename


ds = load_data(netcdf_filepath)

# Heat Index calculation
def compute_heat_index(T, R):
    HI = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + R * 0.094)
    mask = HI >= 80
    HI_roth = (-42.379 + 2.04901523*T + 10.14333127*R - 0.22475541*T*R
               - 6.83783e-3*T**2 - 5.481717e-2*R**2 + 1.22874e-3*T**2*R
               + 8.5282e-4*T*R**2 - 1.99e-6*T**2*R**2)
    HI = xr.where(mask, HI_roth, HI)
    adj1_mask = (R<13) & (T>=80) & (T<=112)
    sqrt_arg = np.maximum((17 - np.abs(T-95))/17, 0)
    adj1 = ((13-R)/4)*np.sqrt(sqrt_arg)
    HI = xr.where(adj1_mask, HI-adj1, HI)
    adj2_mask = (R>85) & (T>=80) & (T<=87)
    adj2 = ((R-85)/10)*((87-T)/5)
    HI = xr.where(adj2_mask, HI+adj2, HI)
    return HI

# Heat Index categories
hi_colors = {
    "No Warning": "lightgreen",        # transparent (so satellite shows through)
    "Caution": "yellow",
    "Extreme Caution": "orange",
    "Danger": "red",
    "Extreme Danger": "darkred"
}

category_order = ["No Warning", "Caution", "Extreme Caution", "Danger", "Extreme Danger"]

def categorise_heat_index(hi):
    hi_cat = xr.full_like(hi, "", dtype="object")
    hi_cat = xr.where(hi < 79.5, "No Warning", hi_cat)
    hi_cat = xr.where((hi >= 79.5) & (hi <= 89.5), "Caution", hi_cat)
    hi_cat = xr.where((hi > 89.5) & (hi <= 104.5), "Extreme Caution", hi_cat)
    hi_cat = xr.where((hi > 104.5) & (hi <= 129.5), "Danger", hi_cat)
    hi_cat = xr.where(hi > 129.5, "Extreme Danger", hi_cat)
    return hi_cat

# user controls
var_choice = st.radio("Select variable:", ["Air Temperature", "Relative Humidity", "Heat Index"])
time_selected = st.select_slider("Select time:", options=ds.time.values)
basemap = st.sidebar.radio("Map style:", ["Satellite", "Street Map"])
opacity = st.sidebar.slider("Transparency", 0.0, 1.0, 0.5, 0.05)
downsample_factor = st.sidebar.number_input("Downsample factor", 1, 20, 4, 1)


# get array
if var_choice in ["Air Temperature", "Relative Humidity"]:
    da = ds[var_choice.lower().replace(" ", "_")].sel(time=time_selected)
    if var_choice == "Air Temperature":
        da = da * 9/5 + 32  # convert to F
else:
    T = ds["air_temperature"].sel(time=time_selected) * 9/5 + 32
    R = ds["relative_humidity"].sel(time=time_selected)
    da = compute_heat_index(T, R)

# downsample to speed things up
if downsample_factor > 1:
    da_small = da.coarsen(x=downsample_factor, y=downsample_factor, boundary='trim').mean()
else:
    da_small = da

# coordinates for overlay (extent) - instead of this, used durham bounding box, its at the top
#lon_min, lon_max = float(da_small.x.min()), float(da_small.x.max())
#lat_min, lat_max = float(da_small.y.min()), float(da_small.y.max())


# Heat Index 
if var_choice == "Heat Index":
    # categorize
    hi_cat = categorise_heat_index(da_small)          # DataArray of strings ("" for undefined)
    # prepare numeric grid
    hi_numeric = np.full(hi_cat.shape, np.nan, dtype=float)
    for idx, cat in enumerate(category_order):
        mask = (hi_cat.values == cat)
        hi_numeric[mask] = idx

    # Build plotly colorscale for 5  cats
    eps = 1e-6
    colorscale = [
        [0.0, hi_colors["No Warning"]],
        [ (0 + eps) / 4.0, hi_colors["No Warning"] ],

        [ (1 - eps) / 4.0, hi_colors["Caution"] ],
        [ 1.0 / 4.0, hi_colors["Caution"] ],

        [ (2 - eps) / 4.0, hi_colors["Extreme Caution"] ],
        [ 2.0 / 4.0, hi_colors["Extreme Caution"] ],

        [ (3 - eps) / 4.0, hi_colors["Danger"] ],
        [ 3.0 / 4.0, hi_colors["Danger"] ],

        [ (4 - eps) / 4.0, hi_colors["Extreme Danger"] ],
        [ 1.0, hi_colors["Extreme Danger"] ],
    ]

    # create heatmap, Use x/y coordinates so extents line up
    fig = go.Figure(go.Heatmap(
        z=hi_numeric,
        x=da_small.x.values,
        y=da_small.y.values,
        zmin=0,
        zmax=4,
        colorscale=colorscale,
        showscale=False, # dont show the colorbar, going to add it in the dummy points as categories 
        colorbar=dict(
            tickvals=list(range(len(category_order))),
            ticktext=category_order,
            title="Heat Index warning"
        ),
        hoverinfo="text",
        text=hi_cat.values,  # hover shows category strings
        opacity=opacity
    ))

    # styling
    fig.update_yaxes(scaleanchor="x", scaleratio=1, visible=False)
    fig.update_xaxes(visible=False)
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0))

    # Add categorical legend using dummy points (points in legend)
    for level in category_order[::-1]:  # reverse
        # use a single invisible point with same color for legend
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=hi_colors[level]),
            name=level,
            showlegend=True
        ))
    # add static image underneath 
    try:
        bg_path = "durham_satellite_bbox.png" if basemap == "Satellite" else "durham_streets_bbox.png"
        bg_img = Image.open(bg_path)
        buffered = io.BytesIO()
        bg_img.save(buffered, format="PNG")
        bg_base64 = base64.b64encode(buffered.getvalue()).decode()

        fig.add_layout_image(
            dict(
                source="data:image/png;base64," + bg_base64,
                xref="x", yref="y",
                x=lon_min, y=lat_max,
                sizex=lon_max - lon_min,
                sizey=lat_max - lat_min,
                sizing="stretch",
                opacity=1,        # use sidebar transparency
                layer="below"           # puts satellite below HI layer
            )
        )

    except FileNotFoundError:
        # image missing — continue without it
        pass
#AT and Rh
else:
    fig = px.imshow(
        da_small,
        origin="lower",
        labels={'x': '', 'y': '', 'color': var_choice},
        color_continuous_scale="Reds" if var_choice == "Air Temperature" else "Blues"
    )
    fig.data[0].opacity = opacity
    fig.update_yaxes(scaleanchor="x", scaleratio=1, visible=False, showgrid=False)
    fig.update_xaxes(visible=False, showgrid=False)
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0))

    # add static  image underneath 
    try:
        bg_path = "durham_satellite_bbox.png" if basemap == "Satellite" else "durham_streets_bbox.png"
        bg_img = Image.open(bg_path)
        buffered = io.BytesIO()
        bg_img.save(buffered, format="PNG")
        bg_base64 = base64.b64encode(buffered.getvalue()).decode()

        fig.add_layout_image(
            dict(
                source="data:image/png;base64," + bg_base64,
                xref="x", yref="y",
                x=lon_min, y=lat_max,
                sizex=lon_max - lon_min,
                sizey=lat_max - lat_min,
                sizing="stretch",
                opacity=1,        # use sidebar transparency
                layer="below"           # puts satellite below HI layer
            )
        )

    except FileNotFoundError:
        # satellite image missing — continue without it
        pass

#geolocation
st.subheader("🔍 Find value at a specific location")
user_location = st.text_input("Enter coordinates or address")
lat = lon = None

MAPBOX_TOKEN = st.secrets["MAPBOX_TOKEN"]

def geocode_mapbox(query):
    """Return (lat, lon, place_name) from Mapbox Geocoding API."""
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{query}.json"
    params = {
        "access_token": MAPBOX_TOKEN,
        "limit": 1,
        "types": "address,place,locality,neighborhood,poi",
         "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}" # Durham bounding box - want to limit the search to here
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        st.error("Mapbox geocoding error.")
        return None

    data = response.json()
    
    if len(data["features"]) == 0:
        st.error("No matching location found.")
        return None

    feature = data["features"][0]
    lon, lat = feature["center"]
    place_name = feature["place_name"]

    return lat, lon, place_name

if user_location:
    result = geocode_mapbox(user_location)
    if result:
        lat, lon, place_name = result
        st.success(f"Found location: {place_name}")
        st.write(f"Lat: {lat:.4f}, Lon: {lon:.4f}")


# streamlit blocks Nominatim :'(
#if user_location:
#    geolocator = Nominatim(user_agent="durham_dashboard", timeout=10)
#    try:
#        location = geolocator.geocode(user_location)
#        if location:
#            lat, lon = location.latitude, location.longitude
#            st.success(f"Found location: {location.address}")
#            st.write(f"Lat: {lat:.4f}, Lon: {lon:.4f}")
#        else:
#            st.warning("Location not found")
#    except Exception as e:
#        st.error(f"Geocoding failed: {e}")

if lat and lon:
    # add marker to the existing figure
    fig.add_scatter(
        x=[lon], y=[lat], mode="markers+text", text=["Selected Location"],
        textposition="top right",
        marker=dict(color="black", size=10, symbol="x"),
        textfont=dict(color="black", size=14),
        name="Selected Location"
    )
    try:
        value = da.sel(y=lat, x=lon, method="nearest").values.item()
        units = "°F" if var_choice == "Air Temperature" or var_choice == "Heat Index" else "%"
        if var_choice == "Heat Index":
            hi_cat_single = categorise_heat_index(xr.DataArray(value)).item()
            st.metric(f"{var_choice} at {user_location}", f"{value:.1f} {units} - {hi_cat_single}")
        else:
            st.metric(f"{var_choice} at {user_location}", f"{value:.1f} {units}")
    except Exception:
        st.warning("Unable to extract value at this location")

st.plotly_chart(fig)#, width='stretch')

# timeseries for selected location
if lat is not None and lon is not None:
    st.subheader("📈 Timeseries for Selected Day")

    selected_date = pd.to_datetime(time_selected).normalize()
    start_time = selected_date
    end_time = selected_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    # Slice dataset by time first
    ds_slice = ds.sel(time=slice(start_time, end_time))

    # City mean and location series
    if var_choice == "Air Temperature":
        city_series = city_mean_df["AT_F"]
        city_series = city_series[city_series.index.normalize() == selected_date]
        loc_series = ds_slice["air_temperature"].sel(x=lon, y=lat, method="nearest") * 9/5 + 32
        units = "°F"

    elif var_choice == "Relative Humidity":
        city_series = city_mean_df["RH"]
        city_series = city_series[city_series.index.normalize() == selected_date]
        loc_series = ds_slice["relative_humidity"].sel(x=lon, y=lat, method="nearest")
        units = "%"

    else:  # Heat Index
        city_series = city_mean_df["HI_F"]
        city_series = city_series[city_series.index.normalize() == selected_date]
        T_loc = ds_slice["air_temperature"].sel(x=lon, y=lat, method="nearest") * 9/5 + 32
        R_loc = ds_slice["relative_humidity"].sel(x=lon, y=lat, method="nearest")
        loc_series = compute_heat_index(T_loc, R_loc)
        units = "°F"

    # Make df for plotly
    df_ts = pd.DataFrame({
        "time": loc_series.time.values,
        "City Mean": city_series.values,
        "Selected Location": loc_series.values
    })
    df_ts.set_index("time", inplace=True)

    # Plot timeseries
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df_ts.index, y=df_ts["City Mean"],
        mode="lines+markers", name="City Mean",
        line=dict(color="#807660")
    ))
    fig_ts.add_trace(go.Scatter(
        x=df_ts.index, y=df_ts["Selected Location"],
        mode="lines+markers", name="Selected Location",
        line=dict(color="#4e7d4e")
    ))

    # Vertical line at selected time
    #fig_ts.add_vline(
     #   x=pd.to_datetime(time_selected),
     #   line=dict(color="black", dash="dash")
    #)

    fig_ts.update_layout(
        title=f"{var_choice} on {selected_date.date()}",
        xaxis_title="Time",
        yaxis_title=f"{var_choice} ({units})",
        height=400
    )
    st.plotly_chart(fig_ts)#, width='stretch')

    # csv download
    csv = df_ts.to_csv().encode("utf-8")
    st.download_button(
        label="Download timeseries CSV",
        data=csv,
        file_name=f"{var_choice}_timeseries_{selected_date.date()}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.caption("Data Driven Envirolab | https://datadrivenlab.org/")