## code for the steamlit dash to be made sharable ##

# link to where the netcdf is stored on DDL google drive 
#https://drive.google.com/file/d/1PzwNpggeCcUy9ODf2wB4tJHrY2nonNbf/view?usp=sharing
#file_id = '1PzwNpggeCcUy9ODf2wB4tJHrY2nonNbf'

import gdown
import os
import streamlit as st
import xarray as xr
import rioxarray
import numpy as np
from geopy.geocoders import Nominatim
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64
import pandas as pd

netcdf_filepath = "Durham_AT_RH_JulAug2024.nc"

if not os.path.exists(netcdf_filepath):
    url = "https://drive.google.com/uc?id=1PzwNpggeCcUy9ODf2wB4tJHrY2nonNbf" 
    gdown.download(url, netcdf_filepath, quiet=False)

st.set_page_config(page_title="Durham AT & RH Dashboard")
st.title("Durham Heat Stress Explorer")
st.write("Explore air temperature, relative humidity and heat index data for July–August 2024. Search for a location to see a timeseries 12 hours either side.")

# Load data
@st.cache_data
def load_data():
    ds = xr.open_dataset(netcdf_filepath)
    if not ds.rio.crs:
        ds = ds.rio.write_crs("EPSG:4326")
    return ds

st.write("NetCDF exists?", os.path.exists(netcdf_filepath))
st.write("File size:", os.path.getsize(netcdf_filepath) if os.path.exists(netcdf_filepath) else "N/A")

ds = load_data()

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
opacity = st.sidebar.slider("Transparency", 0.0, 1.0, 0.9, 0.05)
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

# coordinates for overlay (extent)
lon_min, lon_max = float(da_small.x.min()), float(da_small.x.max())
lat_min, lat_max = float(da_small.y.min()), float(da_small.y.max())


# Heat Index 
if var_choice == "Heat Index":
    # categorize
    hi_cat = categorise_heat_index(da_small)          # DataArray of strings ("" for undefined)
    # prepare numeric grid: 0..4 for categories, keep np.nan where empty
    hi_numeric = np.full(hi_cat.shape, np.nan, dtype=float)
    for idx, cat in enumerate(category_order):
        mask = (hi_cat.values == cat)
        hi_numeric[mask] = idx

    # Build a valid Plotly colorscale for 5 discrete categories.
    # must be normalized to [0,1] — we'll map integers 0..4.
    # Use small eps to make sharp boundaries.
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

    # create heatmap. Use x/y coordinates so extents line up
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
    for level in category_order[::-1]:  # reverse so legend order looks natural
        # use a single invisible point with same color for legend
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=hi_colors[level]),
            name=level,
            showlegend=True
        ))
    # --- Add static satellite image underneath ---
    sat_img = Image.open(r"durham_satellite_bbox.png")
    buffered = io.BytesIO()
    sat_img.save(buffered, format="PNG")
    sat_base64 = base64.b64encode(buffered.getvalue()).decode()

    fig.add_layout_image(
        dict(
            source="data:image/png;base64," + sat_base64,
            xref="x", yref="y",
            x=lon_min, y=lat_max,
            sizex=lon_max - lon_min,
            sizey=lat_max - lat_min,
            sizing="stretch",
            opacity=1,        # use sidebar transparency
            layer="below"           # puts satellite below HI layer
        )
    )

# -------------------------
# Air Temp / RH branch (unchanged)
# -------------------------
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

    # Add static satellite image (if you have it)
    try:
        sat_img = Image.open(r"durham_satellite_bbox.png")
        buffered = io.BytesIO()
        sat_img.save(buffered, format="PNG")
        sat_base64 = base64.b64encode(buffered.getvalue()).decode()

        fig.add_layout_image(
            dict(
                source="data:image/png;base64," + sat_base64,
                xref="x", yref="y",
                x=lon_min, y=lat_max,
                sizex=lon_max - lon_min, sizey=lat_max - lat_min,
                sizing="stretch",
                opacity=1,
                layer="below"
            )
        )
    except FileNotFoundError:
        # satellite image missing — continue without it
        pass

# -------------------------
# Geolocation / value readout
# -------------------------
st.subheader("🔍 Find value at a specific location")
user_location = st.text_input("Enter coordinates or address")
lat = lon = None
if user_location:
    geolocator = Nominatim(user_agent="durham_dashboard", timeout=10)
    try:
        location = geolocator.geocode(user_location)
        if location:
            lat, lon = location.latitude, location.longitude
            st.success(f"Found location: {location.address}")
            st.write(f"Lat: {lat:.4f}, Lon: {lon:.4f}")
        else:
            st.warning("Location not found")
    except Exception as e:
        st.error(f"Geocoding failed: {e}")

if lat and lon:
    # add marker to the existing figure (works for both heatmap and px.imshow figs)
    # for Plotly heatmap (x/y), fig.add_scatter works; for mapbox-based you'll need different code.
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

st.plotly_chart(fig, width='stretch')

# -------------------------
# Timeseries ±12 hours for selected location
# -------------------------
if lat is not None and lon is not None:
    st.subheader("📈 Timeseries around selected time")

    # Convert to pandas datetime
    selected_time_pd = pd.to_datetime(time_selected)
    start_time = selected_time_pd - pd.Timedelta(hours=12)
    end_time = selected_time_pd + pd.Timedelta(hours=12)

    # Slice dataset by time first
    ds_slice = ds.sel(time=slice(start_time, end_time))

    # City mean and location series
    if var_choice == "Air Temperature":
        city_series = ds_slice["air_temperature"].mean(dim=["x", "y"]) * 9/5 + 32
        loc_series = ds_slice["air_temperature"].sel(
            x=lon, y=lat, method="nearest"
        ) * 9/5 + 32
        units = "°F"

    elif var_choice == "Relative Humidity":
        city_series = ds_slice["relative_humidity"].mean(dim=["x", "y"])
        loc_series = ds_slice["relative_humidity"].sel(
            x=lon, y=lat, method="nearest"
        )
        units = "%"

    else:  # Heat Index
        T = ds_slice["air_temperature"] * 9/5 + 32
        R = ds_slice["relative_humidity"]
        HI = compute_heat_index(T, R)
        city_series = HI.mean(dim=["x", "y"])
        loc_series = HI.sel(x=lon, y=lat, method="nearest")
        units = "°F"

    # Make DataFrame for Plotly
    df_ts = pd.DataFrame({
        "time": city_series.time.values,
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
    fig_ts.add_vline(
        x=selected_time_pd,
        line=dict(color="black", dash="dash")
    )

    fig_ts.update_layout(
        title=f"{var_choice} from {start_time} to {end_time}",
        xaxis_title="Time",
        yaxis_title=f"{var_choice} ({units})",
        height=400
    )
    st.plotly_chart(fig_ts, width='stretch')

    # CSV download
    csv = df_ts.to_csv().encode("utf-8")
    st.download_button(
        label="Download timeseries CSV",
        data=csv,
        file_name=f"{var_choice}_timeseries.csv",
        mime="text/csv"
    )


# Footer
st.markdown("---")
st.caption("Data Driven Envirolab | https://datadrivenlab.org/")