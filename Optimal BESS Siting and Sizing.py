#%% Library
import pandas as pd
import numpy as np
import pandapower as pp
import pandapower.plotting as plot
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pyswarms.single import GlobalBestPSO
import traceback
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import random
import pickle

#%% Data Input
pv_data = pd.read_csv('pv_plant_data.csv')
bus_data = pd.read_csv('ieee33_data.csv')  
#field_data = pd.read_csv('Solar_Irradiance.csv')
#field_data['Datetime'] = pd.date_range(start="2023-01-01 00:00:00", periods=len(field_data), freq="H")
#field_data = field_data.iloc[:-7]

bus_numbers = [4, 5, 8, 13, 14, 15, 18, 20, 22, 30, 31, 7, 27, 29]
solar_irradiance_data = {}
for bus in bus_numbers:
    file_name = f'Solar_Irradiance_bus_{bus}.csv'
    solar_irradiance_data[bus] = pd.read_csv(file_name)
    solar_irradiance_data[bus]['Datetime'] = pd.date_range(start="2023-01-01 00:00:00", periods=len(solar_irradiance_data[bus]), freq="H")
    solar_irradiance_data[bus] = solar_irradiance_data[bus].iloc[:-7]  # Remove last 7 rows if necessary

#Load Load Data
try:
    load_data = pd.read_csv('Hourly_Load.csv', encoding='latin1')
    if 'Start date' in load_data.columns and 'End date' in load_data.columns:
        load_data['Start date'] = pd.to_datetime(load_data['Start date'], errors='coerce')
        #load_data['Start date'] = pd.to_datetime(load_data['Start date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        load_data['End date'] = pd.to_datetime(load_data['End date'], errors='coerce')
        #load_data['End date'] = pd.to_datetime(load_data['End date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        load_data.rename(columns={'Total (grid load) [MWh] Calculated resolutions': 'load'}, inplace=True)
    else:
        raise ValueError("Hourly Load.csv file must contain 'Start_DateTime' and 'End_DateTime' columns.")
    load_data.fillna(method='ffill', inplace=True)  
    load_data.fillna(method='bfill', inplace=True)
except Exception as e:
    print(f"Error loading Hourly Load: {e}")
    raise

#Load Market Data
try:
    market_prices = pd.read_csv('market_prices.csv', encoding='latin1')  
    if 'Start_DateTime' in market_prices.columns and 'End_DateTime' in market_prices.columns:
        market_prices['Start_DateTime'] = pd.to_datetime(market_prices['Start_DateTime'], errors='coerce')
        market_prices['End_DateTime'] = pd.to_datetime(market_prices['End_DateTime'], errors='coerce')
    else:
        raise ValueError("The market_prices.csv file must contain 'Start_DateTime' and 'End_DateTime' columns.")
    market_prices.fillna(method='ffill', inplace=True)  
    market_prices.fillna(method='bfill', inplace=True) 
except Exception as e:
    print(f"Error loading market prices: {e}")
    raise  

#Load Balancing Market Data
try:
    reg_prices = pd.read_csv('reg_prices.csv', encoding='latin1')  
    if 'Start_DateTime' in reg_prices.columns and 'End_DateTime' in reg_prices.columns:
        reg_prices['Start_DateTime'] = pd.to_datetime(reg_prices['Start_DateTime'], errors='coerce')
        reg_prices['End_DateTime'] = pd.to_datetime(reg_prices['End_DateTime'], errors='coerce')
        reg_prices = reg_prices.rename(columns={"Price_Up": "Reg_up", "Price_Down": "Reg_down"})
        
        
        reg_prices['Reg_up'] = pd.to_numeric(reg_prices['Reg_up'], errors='coerce')
        reg_prices['Reg_down'] = pd.to_numeric(reg_prices['Reg_down'], errors='coerce')

    else:
        raise ValueError("The reg_prices.csv file must contain 'Start_DateTime' and 'End_DateTime' columns.")
    reg_prices.fillna(method='ffill', inplace=True)  # Forward fill missing values
    reg_prices.fillna(method='bfill', inplace=True)  # Backward fill any remaining missing values
except Exception as e:
    print(f"Error loading regulation prices: {e}")
    raise 

# Load Forecast Data
forecast_prices = pd.read_csv('forecast_prices.csv', encoding='latin1')
forecast_prices['Datetime'] = pd.to_datetime(forecast_prices['Datetime'], errors='coerce')



#Appending to Hourly_Data Dataframe
hourly_data = pd.merge(load_data, market_prices[['Start_DateTime', 'Price_Euro_per_MWh']], left_on='Start date', right_on='Start_DateTime', how='left')
hourly_data = pd.merge(hourly_data, reg_prices[['Start_DateTime', 'Reg_up', 'Reg_down']],  left_on='Start date', right_on='Start_DateTime', how='left')
hourly_data.drop(['Start_DateTime_x','Start_DateTime_y','End date', 'Residual load [MWh] Calculated resolutions', 'Hydro pumped storage [MWh] Calculated resolutions'], axis=1, inplace=True)
hourly_data['load'] = pd.to_numeric(hourly_data['load'].astype(str).str.replace(',', ''), errors='coerce')
hourly_data['Price_Euro_per_MWh'] = pd.to_numeric(hourly_data['Price_Euro_per_MWh'].astype(str).str.replace(',', ''), errors='coerce')



#%% PV Generation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from cycler import cycler

# Create mapping of bus numbers to plant names
bus_to_name = {
    4: 'Oberseiffersdorf',
    5: 'Ronneburg',
    7: 'Görlitz',
    8: 'Vahldorf',
    13: 'Pölzig',
    14: 'Hillersleben',
    15: 'Ramelow',
    18: 'Gahro 2',
    20: 'Kleinwusterwitz',
    22: 'Prenzlau',
    27: 'Bitterfeld',
    29: 'Bresewitz',
    30: 'Serbitz',
    31: 'Danstedt 1'
}

# Define a custom distinct color palette
def get_distinct_colors(n):
    """Generate n visually distinct colors"""
    # Start with these distinct colors
    base_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
        '#aec7e8',  # Light blue
        '#ffbb78',  # Light orange
        '#98df8a',  # Light green
        '#ff9896',  # Light red
        '#c5b0d5',  # Light purple
        '#c49c94',  # Light brown
        '#f7b6d2',  # Light pink
        '#dbdb8d',  # Light olive
        '#9edae5'   # Light cyan
    ]
    
    # If we need more colors than available, we'll create additional ones
    if n <= len(base_colors):
        return base_colors[:n]
    else:
        # Create additional colors using HSV color space
        additional_colors = []
        for i in range(n - len(base_colors)):
            hue = i / (n - len(base_colors))
            additional_colors.append(mcolors.hsv_to_rgb([hue, 0.8, 0.9]))
        
        return base_colors + additional_colors

# PV Generation calculation function (unchanged)
def calculate_pv_generation_for_all_projects(pv_data, solar_irradiance_data, derating_factor=0.8, alpha_p=-0.005, t_stc=25, g_stc=1000, inv_efficiency=0.96, soiling_loss=0.03, shading_loss=0.02, degradation_rate=0.005, age=0):
    # Initialize DataFrame to store PV generation
    pv_gen_data = pd.DataFrame(columns=['timestep'] + [f'pv_gen_{int(bus)}' for bus in pv_data['bus']])
    pv_gen_data['timestep'] = range(len(solar_irradiance_data[next(iter(solar_irradiance_data))]))  # Use the length of the first solar irradiance file
    
    # Iterate over all projects
    for i, row in pv_data.iterrows():
        bus = int(row['bus'])  # Bus number
        y_pv = row['P_MWp']    # Rated power of the PV system (MWp)
        
        # Get the solar irradiance data for this bus
        if bus in solar_irradiance_data:
            g_t = solar_irradiance_data[bus]['solar_irradiance']  # Solar irradiance (W/m²)
            t_c = solar_irradiance_data[bus]['temp']              # Cell temperature (°C)
            
            # Calculate PV generation for all timesteps
            pv_gen = y_pv * derating_factor * (g_t / g_stc) * (1 + alpha_p * (t_c - t_stc)) * inv_efficiency * (1 - soiling_loss) * (1 - shading_loss) * (1 - degradation_rate) ** age
            pv_gen_data[f'pv_gen_{bus}'] = pv_gen
        else:
            print(f"No solar irradiance data found for bus {bus}")
    
    return pv_gen_data

# Plotting with plant names sorted by total production with distinct colors
def plot_pv_generation_profiles(pv_gen_data, pv_data, bus_to_name, summer_start=196*24, winter_start=15*24):
    # Calculate total production for sorting
    total_production = {}
    for bus in pv_data['bus']:
        bus_int = int(bus)
        column_name = f'pv_gen_{bus_int}'
        if column_name in pv_gen_data.columns:
            total_production[bus_int] = pv_gen_data[column_name].sum()
    
    # Sort buses by total production (highest to lowest)
    sorted_buses = sorted(total_production.keys(), key=lambda b: total_production[b], reverse=True)
    
    # Generation Profile for Summer and Winter
    summer_day = pv_gen_data.iloc[summer_start:summer_start+24]
    winter_day = pv_gen_data.iloc[winter_start:winter_start+24]
    
    # Create figure with more width for the single-column legend
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Get distinct colors
    distinct_colors = get_distinct_colors(len(sorted_buses))
    
    # Add line styles for even more distinction
    line_styles = ['-', '--', '-.', ':']
    
    # Summer Generation
    for i, bus in enumerate(sorted_buses):
        column_name = f'pv_gen_{bus}'
        plant_name = bus_to_name.get(bus, f'Bus {bus}')
        # Add the total production in the legend for reference
        label = f"{plant_name} ({total_production[bus] / 1000:.1f} GWh)"
        
        # Use different line styles for groups of 4 to add more distinction
        line_style = line_styles[i % len(line_styles)]
        
        ax1.plot(range(24), summer_day[column_name], label=label, 
                linewidth=2.5, color=distinct_colors[i], 
                linestyle=line_style)
    
    ax1.set_title('Summer Day (July 15) PV Generation Profile', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Hour of Day', fontsize=12)
    ax1.set_ylabel('PV Generation (MW)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    # Create a legend with a single column
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, 
              frameon=True, fancybox=True, ncol=1, shadow=True)
    
    # Winter Day Generation
    for i, bus in enumerate(sorted_buses):
        column_name = f'pv_gen_{bus}'
        plant_name = bus_to_name.get(bus, f'Bus {bus}')
        # Add the total production in the legend for reference
        label = f"{plant_name} ({total_production[bus] / 1000:.1f} GWh)"
        
        # Use different line styles for groups of 4 to add more distinction
        line_style = line_styles[i % len(line_styles)]
        
        ax2.plot(range(24), winter_day[column_name], label=label, 
                linewidth=2.5, color=distinct_colors[i], 
                linestyle=line_style)
    
    ax2.set_title('Winter Day (January 15) PV Generation Profile', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Hour of Day', fontsize=12)
    ax2.set_ylabel('PV Generation (MW)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    # Create a legend with a single column
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, 
              frameon=True, fancybox=True, ncol=1, shadow=True)
    
    # Adjust layout for the single-column legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.80)  # Make more room for the legend
    plt.show()
    
    return sorted_buses, distinct_colors  # Return the sorted order and colors for reference

# Monthly Average PV Generation with plant names sorted by production
def plot_monthly_pv_generation(pv_gen_data, pv_data, bus_to_name, sorted_buses=None, colors=None):
    hours_per_month = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760]
    monthly_data = []
    
    for i in range(len(hours_per_month)-1):
        start_hour = hours_per_month[i]
        end_hour = hours_per_month[i+1]
        monthly_avg = pv_gen_data.iloc[start_hour:end_hour].mean()
        monthly_data.append(monthly_avg)
    
    monthly_df = pd.DataFrame(monthly_data)
    
    # If sorted_buses not provided, calculate total production for sorting
    if sorted_buses is None:
        total_production = {}
        for bus in pv_data['bus']:
            bus_int = int(bus)
            column_name = f'pv_gen_{bus_int}'
            if column_name in pv_gen_data.columns:
                total_production[bus_int] = pv_gen_data[column_name].sum()
        
        # Sort buses by total production (highest to lowest)
        sorted_buses = sorted(total_production.keys(), key=lambda b: total_production[b], reverse=True)
    
    # If colors not provided, generate them
    if colors is None:
        colors = get_distinct_colors(len(sorted_buses))
    
    # Create figure with more width for the single-column legend
    plt.figure(figsize=(16, 8))
    
    # Line styles for additional distinction
    line_styles = ['-', '--', '-.', ':']
    
    # Calculate annual production for each plant to include in legend
    annual_production = {}
    for bus in sorted_buses:
        column_name = f'pv_gen_{bus}'
        annual_production[bus] = pv_gen_data[column_name].sum()
    
    for i, bus in enumerate(sorted_buses):
        column_name = f'pv_gen_{bus}'
        plant_name = bus_to_name.get(bus, f'Bus {bus}')
        # Add the annual production to the legend
        label = f"{plant_name} ({annual_production[bus] / 1000:.1f} GWh)"
        
        # Use different line styles for groups of 4
        line_style = line_styles[i % len(line_styles)]
        
        plt.plot(range(1, 13), monthly_df[column_name], label=label, 
                linewidth=2.5, color=colors[i], linestyle=line_style)
    
    plt.title('Monthly Average PV Generation', fontsize=14, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average PV Generation (MWh)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, 
              frameon=True, fancybox=True, ncol=1, shadow=True)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    # Adjust layout for the single-column legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.80)  # Make more room for the legend
    plt.show()

# Example of how to use these functions in sequence:
# 1. First calculate the PV generation data
pv_gen_data = calculate_pv_generation_for_all_projects(pv_data, solar_irradiance_data)
# 
# 2. Plot daily profiles and get sorted order of buses and colors
sorted_buses, colors = plot_pv_generation_profiles(pv_gen_data, pv_data, bus_to_name)
# 
# 3. Use the same sorted order and colors for monthly plot
plot_monthly_pv_generation(pv_gen_data, pv_data, bus_to_name, sorted_buses, colors)

def calculate_pv_generation_for_all_projects(pv_data, solar_irradiance_data, derating_factor=0.8, alpha_p=-0.005, t_stc=25, g_stc=1000, inv_efficiency=0.96, soiling_loss=0.03, shading_loss=0.02, degradation_rate=0.005, age=0):
    # Initialize DataFrame to store PV generation
    pv_gen_data = pd.DataFrame(columns=['timestep'] + [f'pv_gen_{int(bus)}' for bus in pv_data['bus']])
    pv_gen_data['timestep'] = range(len(solar_irradiance_data[next(iter(solar_irradiance_data))]))  # Use the length of the first solar irradiance file
    
    # Iterate over all projects
    for i, row in pv_data.iterrows():
        bus = int(row['bus'])  # Bus number
        y_pv = row['P_MWp']    # Rated power of the PV system (MWp)
        
        # Get the solar irradiance data for this bus
        if bus in solar_irradiance_data:
            g_t = solar_irradiance_data[bus]['solar_irradiance']  # Solar irradiance (W/m²)
            t_c = solar_irradiance_data[bus]['temp']              # Cell temperature (°C)
            
            # Calculate PV generation for all timesteps
            pv_gen = y_pv * derating_factor * (g_t / g_stc) * (1 + alpha_p * (t_c - t_stc)) * inv_efficiency * (1 - soiling_loss) * (1 - shading_loss) * (1 - degradation_rate) ** age
            pv_gen_data[f'pv_gen_{bus}'] = pv_gen
        else:
            print(f"No solar irradiance data found for bus {bus}")
    
    return pv_gen_data

# Example usage
pv_gen_data = calculate_pv_generation_for_all_projects(pv_data, solar_irradiance_data)

# Merge PV generation data with hourly_data
hourly_data = pd.merge(hourly_data, pv_gen_data, left_index=True, right_on='timestep')
hourly_data.drop('timestep', axis=1, inplace=True)

colors = ['b', 'g', 'r', 'c', 'm', 'y']
# Generation Profile for Summer and Winter
summer_day_start = 196 * 24
summer_day = pv_gen_data.iloc[summer_day_start:summer_day_start+24]
winter_day_start = 15 * 24
winter_day = pv_gen_data.iloc[winter_day_start:winter_day_start+24]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Summer Generation
for bus in pv_data['bus']:
    column_name = f'pv_gen_{int(bus)}'
    ax1.plot(range(24), summer_day[column_name], label=f'PV Plant {int(bus)}', linewidth=2)
ax1.set_title('Summer Day (July 15) PV Generation Profile', fontsize=12)
ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('PV Generation (MW)')
ax1.grid(True, alpha=0.3)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Winter Day Generation
for bus in pv_data['bus']:
    column_name = f'pv_gen_{int(bus)}'
    ax2.plot(range(24), winter_day[column_name], label=f'PV Plant {int(bus)}', linewidth=2)
ax2.set_title('Winter Day (January 15) PV Generation Profile', fontsize=12)
ax2.set_xlabel('Hour of Day')
ax2.set_ylabel('PV Generation (MW)')
ax2.grid(True, alpha=0.3)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# Monthly Average PV Generation
hours_per_month = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760]
monthly_data = []
for i in range(len(hours_per_month)-1):
    start_hour = hours_per_month[i]
    end_hour = hours_per_month[i+1]
    monthly_avg = pv_gen_data.iloc[start_hour:end_hour].mean()
    monthly_data.append(monthly_avg)

monthly_df = pd.DataFrame(monthly_data)

# Plot monthly averages
plt.figure(figsize=(12, 6))
for bus in pv_data['bus']:
    column_name = f'pv_gen_{int(bus)}'
    plt.plot(range(1, 13), monthly_df[column_name], label=f'PV Plant {int(bus)}', linewidth=2)
    
plt.title('Monthly Average PV Generation', fontsize=12)
plt.xlabel('Month')
plt.ylabel('Average PV Generation (MWh)')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


#%% Project Parameters
class MarketParams:
    # Basic BESS Parameters
    BESS_COST_E = 450  # $/kWh
    BESS_COST_P = 250# $/kW
    BESS_EFF_CHARGE = 0.91   
    BESS_EFF_DISCHARGE = 0.91
    SOC_MIN = 0.02   
    SOC_MAX = 0.99
    BUDGET = 15000000   
    C_VAR_OM = BESS_COST_E*0.02  
    MAX_PROJECT_BUDGET = 3000000  # Maximum 5M € per project

    # Degradation Parameters
    ANNUAL_CALENDAR_DEG = 0.02
    CYCLE_DEG_RATE = 0.00004
    
    # Market Operation Limits
    MAX_DAILY_GRID_CHARGES = 5
    MAX_DAILY_CYCLES = 5
    MAX_DAILY_BALANCING = 8
    MIN_PRICE_SPREAD = 9
    MIN_PROFIT_MARGIN = 2
    
    # BESS Sizing Constraints
    MAX_POWER_RATIO = 0.5
    MAX_DURATION = 2
    MIN_DURATION = 1.5
    MIN_PRICE_FOR_CHARGE = 0.7  # Relative to FiP reference price
    MAX_PRICE_FOR_CHARGE = 1.0  # Relative to FiP reference price
    MIN_DISCHARGE_PREMIUM = 0.1  # Minimum 15% price increase for discharge
    
    # Market Participation Parameters
    DA_GATE_CLOSURE = 12  # Day-ahead market closes at 12:00
    DA_MIN_BID = 1.0     # Minimum bid size in MW
    DA_PRICE_PREMIUM = 1.05  # Required premium for discharge
    DA_PRICE_DISCOUNT = 0.9 # Required discount for charging
    
    # Regulation Market Parameters
    REG_BLOCK_HOURS = 4      # Hours per block for regulation
    REG_MAX_POWER = 0.5      # Maximum power for regulation (50% of capacity)
    REG_MIN_SOC = 0.2        # Minimum SOC for regulation up
    REG_MAX_SOC = 0.95        # Maximum SOC for regulation down
    REG_PRICE_THRESHOLD = 0.85
    REG_ACTIVATION_PROB = 0.45 # Probability of regulation activation
    MAX_DAILY_REG_HOURS = 10
    
    # Safety and Buffer Parameters
    SAFETY_BUFFER = 0.1      # 10% safety buffer for commitments
    SOC_BUFFER = 0.05        # Additional SOC buffer
    STRATEGIC_DISCHARGE = 1.1 # Price threshold for strategic discharge
    
    LIFETIME_YEARS = 20
    END_OF_LIFE_CAPACITY = 0.62  # 62% remaining capacity after 20 years
    
    @staticmethod
    def get_oversized_capacity(required_capacity):
        return required_capacity / MarketParams.END_OF_LIFE_CAPACITY

# BESS Sizing Constraints
max_power_ratio = 2.5  # Maximum P/E ratio
max_duration = 2   # Maximum hours of storage
min_duration = 1.5   # Minimum hours of storage
price_threshold_method = 'dynamic'  # Options: 'dynamic', 'percentile', 'time_based', 'ma_crossover'
rolling_window = 336  # Hours to look back (1 week)
std_dev_multiplier = 0.65  # For dynamic threshold
percentile_threshold = 25  # For percentile-based approach
short_ma_window = 24  # For MA crossover
long_ma_window = 168  # For MA crossover
peak_hours_start = 9
peak_hours_end = 20
peak_multiplier = 1.1
off_peak_multiplier = 0.9
evening_discharge_start_hour = 18  # 6 PM
evening_discharge_end_hour = 22    # 10 PM
force_discharge_soc = 0.7         # Force discharge if SOC above this
# Add near your other parameters
soc_discharge_tiers = {
    0.9: 0.95,  # If SOC > 90%, discharge if price is >95% of predicted
    0.8: 1.05,  # If SOC > 80%, discharge if price is >105% of predicted
    0.7: 1.10,  # If SOC > 70%, discharge if price is >110% of predicted
    0.6: 1.15   # If SOC > 60%, discharge if price is >115% of predicted
}

#%% Helper Functions
def calculate_price_thresholds(current_hour: int, market_prices: np.array, timestamps, window_size=168):

 
    historical_window = market_prices[max(0, current_hour-window_size):current_hour]
    
    if len(historical_window) > 0:
        mean_price = np.mean(historical_window)
        std_price = np.std(historical_window)
        
        # Get time-based factors
        hour = pd.to_datetime(timestamps[current_hour]).hour
        is_peak = 9 <= hour <= 20
        
        # Calculate base thresholds
        if is_peak:
            high_threshold = mean_price + 0.8 * std_price
            low_threshold = mean_price - 0.5 * std_price
        else:
            high_threshold = mean_price + 0.6 * std_price
            low_threshold = mean_price - 0.7 * std_price
            
        # Adjust for recent trend
        if len(historical_window) >= 24:
            recent_trend = np.mean(historical_window[-24:]) - mean_price
            high_threshold += 0.3 * recent_trend
            low_threshold += 0.3 * recent_trend
        
        return high_threshold, low_threshold
    else:
        return market_prices[current_hour] * 1.1, market_prices[current_hour] * 0.9
####################################################################################################################
def evaluate_charging_opportunity(current_price: float, predicted_prices: list, 
                               hour: int, soc: float) -> bool:
  
    # Get relevant future prices (until end of next day)
    future_prices = predicted_prices[hour:]
    
    # Calculate potential profit thresholds
    min_profitable_price = current_price * 1.15  # Need at least 15% price increase
    
    # Check if enough higher prices exist
    higher_prices = [p for p in future_prices if p > min_profitable_price]
    
    # Decision factors
    price_opportunity = len(higher_prices) >= 3  # At least 3 hours of higher prices
    soc_headroom = soc < 0.8  # Have enough room to charge
    
    return price_opportunity and soc_headroom
####################################################################################################################
def should_discharge_now(current_price: float, predicted_prices: list, 
                        hour: int, soc: float, high_threshold: float) -> bool:

    # Immediate price check
    price_is_high = current_price > high_threshold
    
    # SOC-based decision
    soc_is_high = soc > 0.9
    
    # Future price check
    future_prices = predicted_prices[hour:]
    better_prices_coming = any(p > current_price * 1.1 for p in future_prices[:6])  # Next 6 hours
    
    # Decision logic
    if soc_is_high:
        # If SOC is very high, be more willing to discharge
        return price_is_high or not better_prices_coming
    else:
        # If SOC is normal, be more selective
        return price_is_high and not better_prices_coming
####################################################################################################################
def calculate_fip_price(market_price: float, fip_reference: float) -> float:

    if market_price < fip_reference:
        premium = fip_reference - market_price
        return market_price + premium
    return market_price
####################################################################################################################
def calculate_dynamic_threshold(prices, t, window=168, std_multiplier=0.5):
    historical_window = prices[max(0, t-window):t]
    if len(historical_window) > 0:
        mean_price = np.mean(historical_window)
        std_price = np.std(historical_window)
        return mean_price - std_multiplier * std_price
    return np.mean(prices[:t+1]) if t > 0 else prices[0]
####################################################################################################################
def calculate_percentile_threshold(prices, t, window=168, percentile=25):
    historical_window = prices[max(0, t-window):t]
    if len(historical_window) > 0:
        return np.percentile(historical_window, percentile)
    return np.percentile(prices[:t+1], percentile) if t > 0 else prices[0]
####################################################################################################################
def calculate_time_based_threshold(prices, t, timestamps, peak_mult=1.1, off_peak_mult=0.9):
    hour = pd.to_datetime(timestamps[t]).hour
    is_peak = peak_hours_start <= hour <= peak_hours_end
    historical_window = prices[max(0, t-168):t]
    base_threshold = np.mean(historical_window) if len(historical_window) > 0 else prices[t]
    return base_threshold * (peak_mult if is_peak else off_peak_mult)
####################################################################################################################
def calculate_ma_crossover_threshold(prices, t, short_window=24, long_window=168):
    short_ma = np.mean(prices[max(0, t-short_window):t]) if t >= short_window else np.mean(prices[:t+1])
    long_ma = np.mean(prices[max(0, t-long_window):t]) if t >= long_window else np.mean(prices[:t+1])
    return min(short_ma, long_ma)
####################################################################################################################
def get_price_threshold(prices, t, timestamps, method='dynamic', **kwargs):
    if method == 'dynamic':
        return calculate_dynamic_threshold(prices, t, kwargs.get('window', rolling_window), 
                                        kwargs.get('std_multiplier', std_dev_multiplier))
    elif method == 'percentile':
        return calculate_percentile_threshold(prices, t, kwargs.get('window', rolling_window),
                                           kwargs.get('percentile', percentile_threshold))
    elif method == 'time_based':
        return calculate_time_based_threshold(prices, t, timestamps, 
                                           kwargs.get('peak_mult', peak_multiplier),
                                           kwargs.get('off_peak_mult', off_peak_multiplier))
    elif method == 'ma_crossover':
        return calculate_ma_crossover_threshold(prices, t, 
                                             kwargs.get('short_window', short_ma_window),
                                             kwargs.get('long_window', long_ma_window))
    else:
        raise ValueError(f"Unknown price threshold method: {method}")
#################################################################################################################### 
def calculate_dynamic_price_thresholds(prices, timestamps, current_idx, window_size=168):  # 1 week window
    current_dt = pd.to_datetime(timestamps[current_idx])
    historical_window = prices[max(0, current_idx-window_size):current_idx]
    
    # 1. Calculate rolling statistics
    rolling_mean = np.mean(historical_window)
    rolling_std = np.std(historical_window)
    
    # 2. Time-of-day pattern
    hour_mask = [pd.to_datetime(t).hour == current_dt.hour 
                for t in timestamps[max(0, current_idx-window_size):current_idx]]
    hourly_prices = historical_window[hour_mask]
    hour_mean = np.mean(hourly_prices) if len(hourly_prices) > 0 else rolling_mean
    hour_std = np.std(hourly_prices) if len(hourly_prices) > 0 else rolling_std

    # 3. Day-of-week pattern
    day_mask = [pd.to_datetime(t).dayofweek == current_dt.dayofweek 
                for t in timestamps[max(0, current_idx-window_size):current_idx]]
    daily_prices = historical_window[day_mask]
    day_mean = np.mean(daily_prices) if len(daily_prices) > 0 else rolling_mean
    
    # 4. Calculate short and long term moving averages
    short_window = 24  # 24 hours
    long_window = 168  # 1 week
    
    short_ma = np.mean(prices[max(0, current_idx-short_window):current_idx])
    long_ma = np.mean(prices[max(0, current_idx-long_window):current_idx])
    
    # Calculate moving average convergence/divergence
    macd = short_ma - long_ma
    
    # Combine all factors with weights
    weights = {
        'hour_pattern': 0.3,
        'day_pattern': 0.2,
        'rolling_stats': 0.3,
        'macd': 0.2
    }
    
    # Calculate base thresholds
    base_high = (
        weights['hour_pattern'] * (hour_mean + hour_std) +
        weights['day_pattern'] * day_mean +
        weights['rolling_stats'] * (rolling_mean + rolling_std) +
        weights['macd'] * (rolling_mean + macd)
    )
    
    base_low = (
        weights['hour_pattern'] * (hour_mean - hour_std) +
        weights['day_pattern'] * day_mean +
        weights['rolling_stats'] * (rolling_mean - rolling_std) +
        weights['macd'] * (rolling_mean - macd)
    )
    
    # Add seasonal adjustments
    month = current_dt.month
    hour = current_dt.hour
    is_peak_season = 6 <= month <= 8 or 12 <= month <= 2  # Summer and Winter
    is_peak_hour = 9 <= hour <= 20
    
    if is_peak_season and is_peak_hour:
        high_threshold = base_high * 1.15  # More aggressive in peak seasons/hours
        low_threshold = base_low * 0.95
    else:
        high_threshold = base_high * 1.05
        low_threshold = base_low * 0.98
    
    return high_threshold, low_threshold
####################################################################################################################
def calculate_profit(decision, spot_price, future_price_estimate, max_energy, efficiency):
    if decision == 'grid_charge':
        return max_energy * (future_price_estimate - spot_price) * efficiency
    elif decision == 'pv_charge':
        return max_energy * future_price_estimate * efficiency
    elif decision == 'discharge':
        return max_energy * (future_price_estimate - spot_price) / efficiency
    elif decision == 'curtail':
        return 0  # Curtailment yields no profit
    return -float('inf')  # Invalid decision
####################################################################################################################
def calculate_reg_participation_score(soc, reg_up_premium, reg_down_premium, 
                                   current_reg_hours, max_reg_hours, weights):
    # SOC score - prefer middle range for maximum flexibility
    soc_score = 1 - 2 * abs(soc - 0.5)  # Highest score at SOC = 0.5
    
    # Price premium score
    price_score = max(reg_up_premium, reg_down_premium)
    
    # Available hours score
    hours_score = 1 - (current_reg_hours / max_reg_hours)
    
    # Weighted combination
    total_score = (
        weights['price_impact'] * price_score +
        weights['soc_impact'] * soc_score +
        weights['reg_revenue_impact'] * hours_score
    )
    
    return total_score
####################################################################################################################
def calculate_dynamic_reg_thresholds(prices, reg_prices, window_size=168):
    price_std = np.std(prices[-window_size:]) if len(prices) >= window_size else np.std(prices)
    reg_std = np.std(reg_prices[-window_size:]) if len(reg_prices) >= window_size else np.std(reg_prices)
    
    up_threshold = np.mean(prices[-window_size:]) + 0.5 * price_std
    down_threshold = np.mean(prices[-window_size:]) - 0.5 * price_std
    
    return up_threshold, down_threshold
####################################################################################################################
def calculate_available_capacity(
    bess_state: Dict[str, float],
    market_type: str = 'da'
) -> Dict[str, float]:
    
    # Ensure capacity is a scalar value
    capacity = float(bess_state['capacity'][0] if isinstance(bess_state['capacity'], np.ndarray) else bess_state['capacity'])
    degradation = float(bess_state['degradation'][0] if isinstance(bess_state['degradation'], np.ndarray) else bess_state['degradation'])
    power = float(bess_state['power'][0] if isinstance(bess_state['power'], np.ndarray) else bess_state['power'])
    
    # Calculate degraded capacities
    degraded_capacity = capacity * (1 - degradation)
    degraded_power = power * (1 - degradation)
    
    # Apply safety buffer
    safe_capacity = degraded_capacity * (1 - MarketParams.SAFETY_BUFFER)
    safe_power = degraded_power * (1 - MarketParams.SAFETY_BUFFER)
    
    available = {
        'charge': 0.0,
        'discharge': 0.0
    }
    
    if market_type == 'da':
        # Consider regulation commitments
        reg_up = float(bess_state['commitments'].get('reg_up', 0))
        reg_down = float(bess_state['commitments'].get('reg_down', 0))
        available['charge'] = float(max(0, safe_power - reg_down))
        available['discharge'] = max(0, safe_power - reg_up)
        
    elif market_type == 'reg_up':
        da_discharge = bess_state['commitments'].get('da_discharge', 0)
        available['discharge'] = min(
            safe_power * MarketParams.REG_MAX_POWER - da_discharge,
            (bess_state['soc'] - MarketParams.REG_MIN_SOC) * 
            safe_capacity * MarketParams.BESS_EFF_DISCHARGE / MarketParams.REG_BLOCK_HOURS
        )
        
    elif market_type == 'reg_down':
        da_charge = bess_state['commitments'].get('da_charge', 0)
        available['charge'] = min(
            safe_power * MarketParams.REG_MAX_POWER - da_charge,
            (MarketParams.REG_MAX_SOC - bess_state['soc']) * 
            safe_capacity / MarketParams.BESS_EFF_CHARGE / MarketParams.REG_BLOCK_HOURS
        )
    
    return available
    
####################################################################################################################
def calculate_regulation_capacity(
    bess_state: Dict[str, float],
    hour: int
) -> Dict[str, float]:

    # Get base available capacity
    available = calculate_available_capacity(bess_state, 'reg')
    
    # Calculate energy constraints for sustained regulation
    energy_for_up = (bess_state['soc'] - MarketParams.REG_MIN_SOC) * \
                    bess_state['capacity'] * MarketParams.BESS_EFF_DISCHARGE
                    
    energy_for_down = (MarketParams.REG_MAX_SOC - bess_state['soc']) * \
                      bess_state['capacity'] / MarketParams.BESS_EFF_CHARGE
    
    # Convert to power considering block duration
    reg_up_power = min(
        available['discharge'],
        energy_for_up / MarketParams.REG_BLOCK_HOURS,
        bess_state['power'] * MarketParams.REG_MAX_POWER
    )
    
    reg_down_power = min(
        available['charge'],
        energy_for_down / MarketParams.REG_BLOCK_HOURS,
        bess_state['power'] * MarketParams.REG_MAX_POWER
    )
    
    return {
        'reg_up': reg_up_power,
        'reg_down': reg_down_power
    }
####################################################################################################################


def simulate_reg_activation(hour, reg_bid, market_price):
    """
    Simulate the activation of aFRR bids based on probability and market conditions.
    """
    # Base activation probability from MarketParams
    base_prob = MarketParams.REG_ACTIVATION_PROB
    
    # Modify probability based on time of day
    if 7 <= hour <= 10 or 17 <= hour <= 20:  # Morning and evening peaks
        base_prob *= 1.5
    elif 0 <= hour <= 5:  # Night hours
        base_prob *= 0.5
        
    # Additional probability based on bid sizes
    up_prob = base_prob * (reg_bid.get('bid_up', 0) > 0)
    down_prob = base_prob * (reg_bid.get('bid_down', 0) > 0)
    
    # Simulate activations
    up_activated = np.random.random() < up_prob
    down_activated = np.random.random() < down_prob
    
    return up_activated, down_activated

def process_regulation_service(bess_state, reg_bid, hour, market_price, ops, results_tracker, scenario_type='pv_only'):
    """
    Process the activation of aFRR bids and update BESS state and revenue.
    Handles both PV-only (aFRR Up only) and PV+Grid (aFRR Up and Down) scenarios.
    """
    if reg_bid['bid_up'] > 0 or reg_bid['bid_down'] > 0:
        # Simulate market activation
        up_activated, down_activated = simulate_reg_activation(
            hour, reg_bid, market_price
        )
        
        # Calculate available capacity for regulation
        reg_capacity = calculate_regulation_capacity(bess_state, hour)
        
        # Handle regulation up activation
        if up_activated and reg_bid['bid_up'] > 0:
            reg_up_power = min(
                reg_bid['bid_up'],
                reg_capacity['reg_up']
            )
            
            if reg_up_power >= MarketParams.DA_MIN_BID:
                results_tracker.p_reg_up[hour] = reg_up_power
                bess_state = update_bess_state(
                    bess_state, reg_up_power, 1, 'discharge'
                )
                ops.reg_up_revenue += reg_up_power * reg_bid['prices']['reg_up']
                ops.reg_up_activations += 1
                ops.reg_up_energy += reg_up_power
                ops.daily_reg_hours += 1
                
        # Handle regulation down activation (only for PV+Grid scenario)
        if scenario_type == 'pv_grid' and down_activated and reg_bid['bid_down'] > 0:
            reg_down_power = min(
                reg_bid['bid_down'],
                reg_capacity['reg_down']
            )
            
            if reg_down_power >= MarketParams.DA_MIN_BID:
                results_tracker.p_reg_down[hour] = reg_down_power
                bess_state = update_bess_state(
                    bess_state, reg_down_power, 1, 'charge'
                )
                ops.reg_down_revenue += reg_down_power * reg_bid['prices']['reg_down']
                ops.reg_down_activations += 1
                ops.reg_down_energy += reg_down_power
                ops.daily_reg_hours += 1
    
    return bess_state

def update_bess_state(bess_state, power, duration, operation_type):
    """
    Update the BESS state after a charge or discharge operation.
    """
    if operation_type == 'charge':
        energy = power * duration * MarketParams.BESS_EFF_CHARGE
        new_soc = min(
            MarketParams.SOC_MAX,
            bess_state['soc'] + energy / bess_state['capacity']
        )
    else:  # discharge
        energy = power * duration / MarketParams.BESS_EFF_DISCHARGE
        new_soc = max(
            MarketParams.SOC_MIN,
            bess_state['soc'] - energy / bess_state['capacity']
        )
    
    # Update cycle count and degradation
    cycle_fraction = energy / (2 * bess_state['capacity'])
    calendar_deg = MarketParams.ANNUAL_CALENDAR_DEG * (duration / 8760)
    cycle_deg = MarketParams.CYCLE_DEG_RATE * cycle_fraction
    
    # Update state
    bess_state.update({
        'soc': new_soc,
        'degradation': bess_state['degradation'] + calendar_deg + cycle_deg,
        'cycle_count': bess_state['cycle_count'] + cycle_fraction,
        'total_throughput': bess_state['total_throughput'] + energy
    })
    
    return bess_state

####################################################################################################################

def prepare_forecast_data(hourly_data: pd.DataFrame, window_size: int = 168) -> pd.DataFrame:
   
    forecast_data = pd.DataFrame()
    
    # Price forecast using rolling mean
    forecast_data['price_forecast'] = hourly_data['Price_Euro_per_MWh'].rolling(
        window=window_size, min_periods=1).mean()
    
    # PV forecast (assume perfect forecast for now)
    pv_columns = [col for col in hourly_data.columns if 'pv_gen_' in col]
    for col in pv_columns:
        forecast_data[f'{col}_forecast'] = hourly_data[col]
    
    # Regulation price forecasts
    forecast_data['reg_up_forecast'] = hourly_data['Reg_up'].rolling(
        window=window_size, min_periods=1).mean()
    forecast_data['reg_down_forecast'] = hourly_data['Reg_down'].rolling(
        window=window_size, min_periods=1).mean()
    
    return forecast_data

forecast_data = prepare_forecast_data(hourly_data)

####################################################################################################################
def check_project_budget(bess_power: float, bess_capacity: float) -> bool:
    oversized_capacity = MarketParams.get_oversized_capacity(bess_capacity)
    
    # Calculate total cost with oversized capacity
    total_cost = (bess_power * MarketParams.BESS_COST_P * 1000 + 
                 oversized_capacity * MarketParams.BESS_COST_E * 1000)

    # Check budget constraint
    if total_cost > MarketParams.MAX_PROJECT_BUDGET:
        return False
    
    # Check duration constraints
    if bess_power > 0:  # Avoid division by zero
        duration = bess_capacity / bess_power
        if not (MarketParams.MIN_DURATION <= duration <= MarketParams.MAX_DURATION):
            return False
    return True

def enforce_budget_optimization(x: np.ndarray, num_buses: int) -> float:
    """
    Return infinite cost if any selected project exceeds budget
    """
    delta = np.round(x[:num_buses]).astype(int)
    selected_buses = np.where(delta == 1)[0]
    bess_power = x[num_buses:num_buses*2]
    bess_capacity = x[num_buses*2:num_buses*3]
    
    for i in selected_buses:
        power_cost = bess_power[i] * MarketParams.BESS_COST_P * 1000
        energy_cost = bess_capacity[i] * MarketParams.BESS_COST_E * 1000
        total_capex = power_cost + energy_cost
        
        if total_capex > MarketParams.MAX_PROJECT_BUDGET:
            return float('inf')
    
    return 0.0

#%% Price Prediction Model
def calculate_moving_averages(prices, current_idx):    
    if current_idx < 24:
        ma_24h = prices[:current_idx + 1].mean() if current_idx > 0 else prices[0]
    else:
        ma_24h = prices[current_idx - 24:current_idx].mean()
    
    if current_idx < 168:
        ma_168h = prices[:current_idx + 1].mean() if current_idx > 0 else prices[0]
    else:
        ma_168h = prices[current_idx - 168:current_idx].mean()
    return ma_24h, ma_168h
####################################################################################################################
def get_hourly_price_patterns(prices, timestamps):
    hours = np.array([i % 24 for i in range(len(timestamps))])
    hourly_averages = {}
    overall_mean = np.mean(prices)
    
    for hour in range(24):
        hour_prices = prices[hours == hour]
        if len(hour_prices) > 0:
            hourly_averages[hour] = np.mean(hour_prices)
        else:
            hourly_averages[hour] = overall_mean
    return hourly_averages
####################################################################################################################
def get_daily_price_patterns(prices, timestamps):
    days = np.array([i // 24 % 7 for i in range(len(timestamps))])
    daily_averages = {}
    overall_mean = np.mean(prices)
    
    for day in range(7):
        day_prices = prices[days == day]
        if len(day_prices) > 0:
            daily_averages[day] = np.mean(day_prices)
        else:
            daily_averages[day] = overall_mean
    return daily_averages
####################################################################################################################
def predict_price(current_idx, prices, timestamps, hourly_patterns, daily_patterns):
    current_hour = current_idx % 24
    current_day = current_idx // 24 % 7
    
    # Calculate volatility
    volatility = np.std(prices[max(0, current_idx-168):current_idx]) if current_idx >= 168 else np.std(prices[:current_idx+1])
    
    # Calculate moving averages
    ma_24h, ma_168h = calculate_moving_averages(prices, current_idx)
    
    # Calculate short-term trend (last 6 hours)
    if current_idx >= 6:
        short_term_trend = np.mean(prices[current_idx-6:current_idx])
    else:
        short_term_trend = prices[current_idx]
    
    # Calculate recent price change (last 1 hour)
    if current_idx >= 1:
        recent_change = prices[current_idx] - prices[current_idx-1]
    else:
        recent_change = 0
    
    # Adjust weights based on hour
    if 8 <= current_hour <= 20:  # Peak hours
        hourly_weight = 0.7  # Increased weight for hourly patterns
        daily_weight = 0.1
        ma24_weight = 0.05  # Reduced weight for moving averages
        ma168_weight = 0.05
        trend_weight = 0.05  # Increased weight for short-term trend
        recent_weight = 0.05  # Added weight for recent price change
    else:  # Off-peak
        hourly_weight = 0.6
        daily_weight = 0.1
        ma24_weight = 0.1
        ma168_weight = 0.05
        trend_weight = 0.1
        recent_weight = 0.05
    
    # Predict next 24 hours
    predicted_prices = []
    upper_bounds = []
    lower_bounds = []
    
    for hour_ahead in range(24):
        # Calculate the hour and day for the prediction
        pred_hour = (current_hour + hour_ahead) % 24
        pred_day = (current_day + (current_hour + hour_ahead) // 24) % 7
        
        # Combine predictions using weighted average
        predicted = (
            hourly_patterns[pred_hour] * hourly_weight +
            daily_patterns[pred_day] * daily_weight +
            ma_24h * ma24_weight +
            ma_168h * ma168_weight +
            short_term_trend * trend_weight +
            recent_change * recent_weight
        )
        
        # Add confidence bands based on volatility
        upper_band = predicted + volatility * 1
        lower_band = predicted - volatility * 1
        
        predicted_prices.append(predicted)
        upper_bounds.append(upper_band)
        lower_bounds.append(lower_band)
    
    return predicted_prices, upper_bounds, lower_bounds
####################################################################################################################
def run_price_prediction(hourly_data):
    prices = hourly_data['Price_Euro_per_MWh'].values
    timestamps = np.arange(len(prices))  # Using numeric indices instead of timestamps
    
    # Calculate patterns
    hourly_patterns = get_hourly_price_patterns(prices, timestamps)
    daily_patterns = get_daily_price_patterns(prices, timestamps)
    
    # Make predictions for a specific time
    current_idx = 100  # Example index
    predicted_prices, upper_bounds, lower_bounds = predict_price(
        current_idx, prices, timestamps, hourly_patterns, daily_patterns
    )
    
    return predicted_prices, upper_bounds, lower_bounds
####################################################################################################################
data = {
    'Start_DateTime': pd.date_range(start='2023-01-01', periods=8760, freq='H'),
    'Reg_up': np.random.rand(8760) * 100,  # Replace with actual Reg_up prices
    'Reg_down': np.random.rand(8760) * 100  # Replace with actual Reg_down prices
}
reg_prices = pd.DataFrame(data)

# Feature Engineering
# 1. Calculate moving averages
reg_prices['Reg_up_MA_24'] = reg_prices['Reg_up'].rolling(window=24).mean()
reg_prices['Reg_down_MA_24'] = reg_prices['Reg_down'].rolling(window=24).mean()

# 2. Create lagged features
lags = 24  # Number of lagged hours to include
for lag in range(1, lags + 1):
    reg_prices[f'Reg_up_lag_{lag}'] = reg_prices['Reg_up'].shift(lag)
    reg_prices[f'Reg_down_lag_{lag}'] = reg_prices['Reg_down'].shift(lag)

# Drop rows with NaN values (due to rolling window and lagged features)
reg_prices.dropna(inplace=True)

# Prepare Features and Target
# Features for Reg_up
X_up = reg_prices[['Reg_up_MA_24'] + [f'Reg_up_lag_{lag}' for lag in range(1, lags + 1)]]
y_up = reg_prices['Reg_up']

# Features for Reg_down
X_down = reg_prices[['Reg_down_MA_24'] + [f'Reg_down_lag_{lag}' for lag in range(1, lags + 1)]]
y_down = reg_prices['Reg_down']

# Train-Test Split
# Split data for Reg_up
X_train_up, X_test_up, y_train_up, y_test_up = train_test_split(X_up, y_up, test_size=0.2, shuffle=False)

# Split data for Reg_down
X_train_down, X_test_down, y_train_down, y_test_down = train_test_split(X_down, y_down, test_size=0.2, shuffle=False)

# Train Random Forest Models
model_up = RandomForestRegressor(n_estimators=100, random_state=42)
model_up.fit(X_train_up, y_train_up)

model_down = RandomForestRegressor(n_estimators=100, random_state=42)
model_down.fit(X_train_down, y_train_down)

# Generate Forecasts
# Forecast for the next 24 hours
forecast_up = model_up.predict(X_test_up.tail(24))
forecast_down = model_down.predict(X_test_down.tail(24))

# Combine actual and forecasted data
afrr_prices_df = pd.DataFrame({
    'Actual_Reg_Up': y_test_up.tail(24).values,
    'Forecast_Reg_Up': forecast_up,
    'Actual_Reg_Down': y_test_down.tail(24).values,
    'Forecast_Reg_Down': forecast_down
})

# Set the index to the datetime column
afrr_prices_df.index = pd.to_datetime(reg_prices.tail(24)['Start_DateTime'])

#%% Market Bidding Model
def prepare_pv_only_bids(price_forecast, pv_forecast, bess_state, grid_limits, fit_price):
    bids = []
    current_soc = bess_state['soc']
    capacity = bess_state['capacity']
    power = bess_state['power']
    
    for hour in range(24):
        price = price_forecast.iloc[hour]  # Predicted market price for this hour
        pv_gen = pv_forecast.iloc[hour]    # PV generation forecast for this hour
        
        # Calculate available capacity for discharging
        available = calculate_available_capacity(bess_state, 'da')
        
        # Initialize bid
        bid = {'hour': hour, 'type': None, 'price': 0.0, 'amount': 0.0}
        
        # Case 1: Market price > FiP/FiT -> Sell energy (PV + BESS)
        if price > fit_price:
            # Step 1: Sell PV power directly to the grid (up to grid_limits)
            pv_direct_sale = min(pv_gen, grid_limits)
            if pv_direct_sale > 0:
                bid.update({
                    'type': 'sell_pv',
                    'price': price,
                    'amount': pv_direct_sale
                })
            
            # Step 2: If there is remaining grid capacity, discharge BESS
            remaining_grid_capacity = grid_limits - pv_direct_sale
            if remaining_grid_capacity > 0:
                max_discharge = min(
                    available['discharge'],
                    power,
                    remaining_grid_capacity,
                    (current_soc - MarketParams.SOC_MIN) * capacity * MarketParams.BESS_EFF_DISCHARGE
                )
                if max_discharge >= MarketParams.DA_MIN_BID:
                    bid.update({
                        'type': 'discharge',
                        'price': price,
                        'amount': max_discharge
                    })
        
        # Case 2: Market price <= FiP/FiT -> Sell PV generation at FiP price if no future high price or SOC is full
        elif price <= fit_price:
            # Check if there are future high-price hours
            future_prices = price_forecast.iloc[hour+1:hour+6]  # Look ahead 5 hours
            if not any(future_price > fit_price for future_price in future_prices) or current_soc >= MarketParams.SOC_MAX:
                # Sell PV generation directly to the grid at FiP price
                pv_direct_sale = min(pv_gen, grid_limits)
                if pv_direct_sale > 0:
                    bid.update({
                        'type': 'sell_pv',
                        'price': fit_price,  # Sell at FiP price
                        'amount': pv_direct_sale
                    })
        
        # Add bid if valid
        if bid['type'] is not None:
            bids.append(bid)
            
        # Update expected SOC for next hour
        if bid['type'] == 'charge':
            current_soc += (bid['amount'] * MarketParams.BESS_EFF_CHARGE) / capacity
        elif bid['type'] == 'discharge':
            current_soc -= (bid['amount'] / MarketParams.BESS_EFF_DISCHARGE) / capacity
    
    return bids

####################################################################################################################
def prepare_pv_grid_bids(price_forecast, pv_forecast, bess_state, grid_limits):
    bids = []
    current_soc = bess_state['soc']
    capacity = bess_state['capacity']
    power = bess_state['power']
    
    # Calculate price thresholds
    avg_price = price_forecast.mean()
    high_price = avg_price * MarketParams.DA_PRICE_PREMIUM  # Threshold for discharge
    low_price = avg_price * MarketParams.DA_PRICE_DISCOUNT  # Threshold for charge
    
    for hour in range(24):
        price = price_forecast.iloc[hour]  # Predicted market price for this hour
        pv_gen = pv_forecast.iloc[hour]    # PV generation forecast for this hour
        
        # Calculate available capacity for charging/discharging
        available = calculate_available_capacity(bess_state, 'da')
        
        # Initialize bid
        bid = {'hour': hour, 'type': None, 'price': 0.0, 'amount': 0.0}
        
        # Case 1: Market price is high -> Sell energy (PV + BESS)
        if price > high_price:
            # Step 1: Sell PV power directly to the grid (up to grid_limits)
            pv_direct_sale = min(pv_gen, grid_limits)
            if pv_direct_sale > 0:
                bid.update({
                    'type': 'sell_pv',
                    'price': price,
                    'amount': pv_direct_sale
                })
            
            # Step 2: If there is remaining grid capacity, discharge BESS
            remaining_grid_capacity = grid_limits - pv_direct_sale
            if remaining_grid_capacity > 0:
                max_discharge = min(
                    available['discharge'],
                    power,
                    remaining_grid_capacity,
                    (current_soc - MarketParams.SOC_MIN) * capacity * MarketParams.BESS_EFF_DISCHARGE
                )
                if max_discharge >= MarketParams.DA_MIN_BID:
                    bid.update({
                        'type': 'discharge',
                        'price': price,
                        'amount': max_discharge
                    })
        
        # Case 2: Market price is low -> Buy energy (charge BESS from grid only)
        elif price < low_price:
            # Check if there are future high-price hours to justify charging
            future_prices = price_forecast.iloc[hour+1:hour+6]  # Look ahead 5 hours
            if any(future_price > high_price for future_price in future_prices):
                # Charge from the grid (grid-to-BESS only)
                max_charge_grid = min(
                    available['charge'],
                    power,
                    grid_limits,
                    (MarketParams.SOC_MAX - current_soc) * capacity / MarketParams.BESS_EFF_CHARGE
                )
                if max_charge_grid >= MarketParams.DA_MIN_BID:
                    bid.update({
                        'type': 'charge_grid',
                        'price': price,
                        'amount': max_charge_grid
                    })
        
        # Add bid if valid
        if bid['type'] is not None:
            bids.append(bid)
            
        # Update expected SOC for next hour
        if bid['type'] == 'charge_grid':
            current_soc += (bid['amount'] * MarketParams.BESS_EFF_CHARGE) / capacity
        elif bid['type'] == 'discharge':
            current_soc -= (bid['amount'] / MarketParams.BESS_EFF_DISCHARGE) / capacity
        
        # Add expected PV contribution (PV-to-BESS charging is not part of the bid)
        current_soc = min(MarketParams.SOC_MAX, 
                         current_soc + (pv_gen * MarketParams.BESS_EFF_CHARGE) / capacity)
    
    return bids

####################################################################################################################
def group_bids_by_type(bids, scenario_type):
    grouped_bids = {
        'sell': [],
        'buy': []
    }
    
    for bid in bids:
        if scenario_type == 'pv_only':
            # PV-only scenario: Only sell bids (discharge or sell_pv)
            if bid['type'] in ['discharge', 'sell_pv']:
                grouped_bids['sell'].append(bid)
        elif scenario_type == 'pv_grid':
            # PV+Grid scenario: Sell bids (discharge or sell_pv) and buy bids (charge or charge_grid)
            if bid['type'] in ['discharge', 'sell_pv']:
                grouped_bids['sell'].append(bid)
            elif bid['type'] in ['charge', 'charge_grid']:
                grouped_bids['buy'].append(bid)
    
    return grouped_bids
####################################################################################################################
def run_bidding_process(pv_data, hourly_data, bess_state, project_index, scenario_type):
    # Get FiT and grid limits for the specific project from pv_data
    fit_price = pv_data.iloc[project_index]['FIT']  # FiT price from the FIT column
    grid_limits = pv_data.iloc[project_index]['max_grid']  # Grid connection capacity from max_grid column

    # Get hourly patterns for price prediction
    hourly_patterns = get_hourly_price_patterns(hourly_data['Price_Euro_per_MWh'].values, hourly_data.index)
    daily_patterns = get_daily_price_patterns(hourly_data['Price_Euro_per_MWh'].values, hourly_data.index)
    
    # Predict next day prices using the price prediction model
    current_idx = len(hourly_data) - 1  # Assuming we're at the end of the dataset
    predicted_prices, _, _ = predict_price(
        current_idx, hourly_data['Price_Euro_per_MWh'].values, hourly_data.index, hourly_patterns, daily_patterns
    )
    
    # Prepare bids for the next day using predicted prices
    if scenario_type == 'pv_only':
        # PV-only scenario
        bids = prepare_pv_only_bids(
            pd.Series(predicted_prices),  # Predicted prices
            hourly_data[f'pv_gen_{int(pv_data.iloc[project_index]["bus"])}'],  # PV generation from hourly_data
            bess_state,
            grid_limits,  # Grid connection capacity (scalar value)
            fit_price  # FiP/FiT price
        )
    elif scenario_type == 'pv_grid':
        # PV+Grid scenario
        bids = prepare_pv_grid_bids(
            pd.Series(predicted_prices),  # Predicted prices
            hourly_data[f'pv_gen_{int(pv_data.iloc[project_index]["bus"])}'],  # PV generation from hourly_data
            bess_state,
            grid_limits  # Grid connection capacity (scalar value)
        )
    
    # Group bids by type (sell or buy)
    grouped_bids = group_bids_by_type(bids, scenario_type)
    
    return grouped_bids

# Define initial BESS state
bess_state = {
    'power': 5.0,  # MW
    'capacity': 20.0,  # MWh
    'soc': 0.5,  # State of Charge (50%)
    'degradation': 0,  # No degradation
    'cycle_count': 0,  # No cycles yet
    'total_throughput': 0,  # No energy throughput yet
    'commitments': {}  # No commitments yet
}

# Run bidding process for PV-only scenario
project_index = 0  # Change this to the index of the project you want to analyze
pv_only_bids = run_bidding_process(pv_data, hourly_data, bess_state, project_index, scenario_type='pv_only')

# Run bidding process for PV+Grid scenario
pv_grid_bids = run_bidding_process(pv_data, hourly_data, bess_state, project_index, scenario_type='pv_grid')

# Print grouped bids for PV-only scenario
print("\nPV-Only Scenario Bids:")
print("Sell Bids:")
for bid in pv_only_bids['sell']:
    print(f"Hour: {bid['hour']}, Type: {bid['type']}, Price: {bid['price']:.2f} €/MWh, Amount: {bid['amount']:.2f} MW")

# Print grouped bids for PV+Grid scenario
print("\nPV+Grid Scenario Bids:")
print("Sell Bids:")
for bid in pv_grid_bids['sell']:
    print(f"Hour: {bid['hour']}, Type: {bid['type']}, Price: {bid['price']:.2f} €/MWh, Amount: {bid['amount']:.2f} MW")
print("Buy Bids:")
for bid in pv_grid_bids['buy']:
    print(f"Hour: {bid['hour']}, Type: {bid['type']}, Price: {bid['price']:.2f} €/MWh, Amount: {bid['amount']:.2f} MW")

####################################################################################################################
# AFRR Market Parameters
class AFRRParams:
    # Market timing parameters
    BLOCK_HOURS = 4        # Hours per block
    BLOCKS_PER_DAY = 6     # Number of blocks per day
    
    # Capacity limits
    MAX_REG_POWER = 0.5    # Maximum power for regulation (50% of capacity)
    MIN_REG_POWER = 0.1    # Minimum power for regulation (MW)
    
    # SOC limits for regulation
    MIN_SOC_REG_UP = 0.2   # Minimum SOC for regulation up
    MAX_SOC_REG_DOWN = 0.8 # Maximum SOC for regulation down
    
    # Price thresholds
    UP_PRICE_THRESHOLD = 1.1   # Multiplier above average price
    DOWN_PRICE_THRESHOLD = 0.9  # Multiplier below average price
    
    # Operational limits
    MAX_DAILY_REG_HOURS = 12
    ACTIVATION_PROB = 0.15     # Probability of activation

class AFRRTracker:
    """Tracks AFRR commitments and operations"""
    def __init__(self, num_timesteps):
        self.reg_up = np.zeros(num_timesteps)
        self.reg_down = np.zeros(num_timesteps)
        self.revenue_up = np.zeros(num_timesteps)
        self.revenue_down = np.zeros(num_timesteps)
        self.activations_up = 0
        self.activations_down = 0
        self.total_energy_up = 0
        self.total_energy_down = 0
####################################################################################################################
def calculate_available_reg_capacity(
    bess_state: dict,
    da_commitments: dict,
    block_hour: int
) -> dict:
    """
    Calculate available capacity for regulation considering DA commitments
    """
    # Get base BESS parameters
    max_power = bess_state['power']
    current_soc = bess_state['soc']
    capacity = bess_state['capacity']
    
    # Get DA commitments for this hour
    da_charge = da_commitments.get(block_hour, {}).get('charge', 0)
    da_discharge = da_commitments.get(block_hour, {}).get('discharge', 0)
    
    # Calculate remaining power capacity
    power_up = max(0, max_power * AFRRParams.MAX_REG_POWER - da_discharge)
    power_down = max(0, max_power * AFRRParams.MAX_REG_POWER - da_charge)
    
    # Calculate energy constraints
    energy_for_up = (current_soc - AFRRParams.MIN_SOC_REG_UP) * capacity
    energy_for_down = (AFRRParams.MAX_SOC_REG_DOWN - current_soc) * capacity
    
    # Return available capacities
    return {
        'reg_up': min(power_up, energy_for_up / AFRRParams.BLOCK_HOURS),
        'reg_down': min(power_down, energy_for_down / AFRRParams.BLOCK_HOURS)
    }
####################################################################################################################
def prepare_afrr_bids(
    prices: pd.DataFrame,
    bess_state: dict,
    da_commitments: dict,
    scenario_type: str = 'pv_only'
) -> List[dict]:
    """
    Prepare AFRR bids considering DA commitments
    """
    bids = []
    num_blocks = AFRRParams.BLOCKS_PER_DAY
    
    for block in range(num_blocks):
        start_hour = block * AFRRParams.BLOCK_HOURS
        end_hour = start_hour + AFRRParams.BLOCK_HOURS
        
        # Get average prices for the block
        block_prices = prices.iloc[start_hour:end_hour]
        avg_market_price = block_prices['Price_Euro_per_MWh'].mean()
        avg_reg_up = block_prices['Reg_up'].mean()
        avg_reg_down = block_prices['Reg_down'].mean()
        
        # Calculate available capacity
        available = calculate_available_reg_capacity(
            bess_state,
            da_commitments,
            start_hour
        )
        
        bid = {
            'block': block,
            'start_hour': start_hour,
            'end_hour': end_hour,
            'reg_up': 0,
            'reg_down': 0,
            'prices': {
                'market': avg_market_price,
                'reg_up': avg_reg_up,
                'reg_down': avg_reg_down
            }
        }
        
        # Check price conditions for reg up
        if avg_reg_up > avg_market_price * AFRRParams.UP_PRICE_THRESHOLD:
            bid['reg_up'] = min(
                available['reg_up'],
                bess_state['power'] * AFRRParams.MAX_REG_POWER
            )
        
        # For PV+Grid scenario, also check reg down
        if scenario_type == 'pv_grid':
            if avg_reg_down < avg_market_price * AFRRParams.DOWN_PRICE_THRESHOLD:
                bid['reg_down'] = min(
                    available['reg_down'],
                    bess_state['power'] * AFRRParams.MAX_REG_POWER
                )
        
        if bid['reg_up'] > 0 or bid['reg_down'] > 0:
            bids.append(bid)
    
    return bids
####################################################################################################################
def submit_afrr_bids(prices, reg_prices, soc, bess_power, bess_capacity, scenario_type='pv_only'):
    """
    Prepare and submit aFRR bids based on available capacity and market conditions.
    Uses calculate_regulation_capacity to ensure energy constraints are respected.
    """
    num_blocks = len(prices) // MarketParams.REG_BLOCK_HOURS
    bids = []

    for block in range(num_blocks):
        start_idx = block * MarketParams.REG_BLOCK_HOURS
        end_idx = start_idx + MarketParams.REG_BLOCK_HOURS

        block_prices = prices[start_idx:end_idx]
        block_reg_up = reg_prices['Reg_up'][start_idx:end_idx]
        block_reg_down = reg_prices['Reg_down'][start_idx:end_idx]

        avg_price = np.mean(block_prices)
        avg_reg_up = np.mean(block_reg_up)  # Forecasted regulation up price
        avg_reg_down = np.mean(block_reg_down)  # Forecasted regulation down price

        # Calculate available capacity for regulation
        reg_capacity = calculate_regulation_capacity(
            {
                'power': bess_power,
                'capacity': bess_capacity,
                'soc': soc,
                'commitments': {}  # Add DA commitments here if available
            },
            hour=start_idx  # Current hour
        )

        # Submit regulation up bid if price is attractive
        if avg_reg_up > avg_price * MarketParams.REG_PRICE_THRESHOLD:
            bid_up = max(
                MarketParams.DA_MIN_BID,
                min(reg_capacity['reg_up'], bess_power * MarketParams.REG_MAX_POWER)
            )
        else:
            bid_up = 0

        # Submit regulation down bid if price is attractive (only for PV+Grid scenario)
        if scenario_type == 'pv_grid' and avg_reg_down < avg_price * MarketParams.REG_PRICE_THRESHOLD:
            bid_down = max(
                MarketParams.DA_MIN_BID,
                min(reg_capacity['reg_down'], bess_power * MarketParams.REG_MAX_POWER)
            )
        else:
            bid_down = 0

        bids.append({
            'block': block,
            'bid_up': bid_up,
            'bid_down': bid_down,
            'start_hour': start_idx,
            'end_hour': end_idx,
            'prices': {
                'market': avg_price,
                'reg_up': avg_reg_up,
                'reg_down': avg_reg_down
            }
        })
    
    return bids
####################################################################################################################
def simulate_afrr_activations(hour: int, bid: dict) -> Tuple[bool, bool]:
    """
    Simulate if regulation services are activated
    """
    # Base probability
    base_prob = AFRRParams.ACTIVATION_PROB
    
    # Modify probability based on time of day
    if 7 <= hour <= 10 or 17 <= hour <= 20:  # Peak hours
        base_prob *= 1.5
    elif 0 <= hour <= 5:  # Night hours
        base_prob *= 0.5
    
    # Calculate activation probabilities
    up_prob = base_prob if bid['reg_up'] > 0 else 0
    down_prob = base_prob if bid['reg_down'] > 0 else 0
    
    # Simulate activations
    up_activated = np.random.random() < up_prob
    down_activated = np.random.random() < down_prob
    
    return up_activated, down_activated

def run_afrr_operation(
    prices: pd.DataFrame,
    bess_state: dict,
    da_commitments: dict,
    scenario_type: str = 'pv_only'
) -> Tuple[Dict, AFRRTracker]:
    """
    Run AFRR market operation
    """
    # Initialize tracker
    tracker = AFRRTracker(len(prices))
    
    # Prepare bids for each day
    num_days = len(prices) // 24
    
    for day in range(num_days):
        # Get daily data slice
        start_idx = day * 24
        end_idx = start_idx + 24
        daily_prices = prices.iloc[start_idx:end_idx]
        
        # Prepare bids
        daily_bids = prepare_afrr_bids(
            daily_prices,
            bess_state,
            da_commitments,
            scenario_type
        )
        
        # Process each bid
        for bid in daily_bids:
            for hour in range(bid['start_hour'], bid['end_hour']):
                # Simulate activations
                up_activated, down_activated = simulate_afrr_activations(
                    hour % 24, 
                    bid
                )
                
                # Process activations
                if up_activated and bid['reg_up'] > 0:
                    tracker.reg_up[start_idx + hour] = bid['reg_up']
                    tracker.revenue_up[start_idx + hour] = (
                        bid['reg_up'] * bid['prices']['reg_up']
                    )
                    tracker.activations_up += 1
                    tracker.total_energy_up += bid['reg_up']
                    
                    # Update BESS state
                    bess_state['soc'] -= (
                        bid['reg_up'] / bess_state['capacity']
                    )
                
                if down_activated and bid['reg_down'] > 0:
                    tracker.reg_down[start_idx + hour] = bid['reg_down']
                    tracker.revenue_down[start_idx + hour] = (
                        bid['reg_down'] * bid['prices']['reg_down']
                    )
                    tracker.activations_down += 1
                    tracker.total_energy_down += bid['reg_down']
                    
                    # Update BESS state
                    bess_state['soc'] += (
                        bid['reg_down'] / bess_state['capacity']
                    )
                
                # Enforce SOC limits
                bess_state['soc'] = np.clip(
                    bess_state['soc'],
                    AFRRParams.MIN_SOC_REG_UP,
                    AFRRParams.MAX_SOC_REG_DOWN
                )
    
    return bess_state, tracker
####################################################################################################################
def calculate_afrr_revenue(tracker: AFRRTracker) -> dict:
    """
    Calculate total AFRR revenue and statistics
    """
    return {
        'revenue_up': np.sum(tracker.revenue_up),
        'revenue_down': np.sum(tracker.revenue_down),
        'total_revenue': np.sum(tracker.revenue_up) + np.sum(tracker.revenue_down),
        'activations_up': tracker.activations_up,
        'activations_down': tracker.activations_down,
        'energy_up': tracker.total_energy_up,
        'energy_down': tracker.total_energy_down
    }

#%% Base Classes
class MarketOperation:
    def __init__(self):
        # Market specific revenues
        self.is_pv_only = True
        self.da_market_revenue = 0  # Day-ahead market revenue
        self.reg_up_revenue = 0     # Regulation up revenue
        self.reg_down_revenue = 0   # Regulation down revenue
        self.pv_revenue = 0         # PV generation revenue
        self.total_costs = 0        # Total costs (charging, O&M, etc.)
        
        # Operation counters
        self.daily_cycles = 0
        self.daily_grid_charges = 0
        self.daily_reg_hours = 0
        
        # Detailed tracking
        self.reg_up_activations = 0
        self.reg_down_activations = 0
        self.reg_up_energy = 0      # MWh of regulation up provided
        self.reg_down_energy = 0    # MWh of regulation down provided
        
    def reset_daily_counters(self):
        self.daily_cycles = 0
        self.daily_grid_charges = 0
        self.daily_reg_hours = 0
        
    def get_revenue_breakdown(self):
        # For PV-only: FIT revenue + Regulation revenue
        if self.is_pv_only:
            return {
                'pv_fit': self.pv_revenue,          # Using FIT price
                'regulation_up': self.reg_up_revenue,  # Using aFRR price
                'total_costs': self.total_costs,
                'net_profit': (self.pv_revenue + self.reg_up_revenue - self.total_costs)
            }
        # For PV+Grid: All market-based revenue
        else:
            return {
                'day_ahead': self.da_market_revenue,    # Using market price
                'regulation_up': self.reg_up_revenue,   # Using aFRR prices
                'regulation_down': self.reg_down_revenue,
                'total_costs': self.total_costs,
                'net_profit': (self.da_market_revenue + self.reg_up_revenue + 
                              self.reg_down_revenue - self.total_costs)
            }
####################################################################################################################
class OperationResults:
    """Class to store operation results"""
    def __init__(self, num_timesteps: int):
        self.p_charge = np.zeros(num_timesteps)
        self.p_charge_pv = np.zeros(num_timesteps)
        self.p_charge_grid = np.zeros(num_timesteps)
        self.p_discharge = np.zeros(num_timesteps)
        self.p_reg_up = np.zeros(num_timesteps)
        self.p_reg_down = np.zeros(num_timesteps)
        self.p_export = np.zeros(num_timesteps)
        self.curtailment = np.zeros(num_timesteps)
        self.soc = np.zeros(num_timesteps)
        self.day_ahead_bids = np.zeros(num_timesteps)  # Store day-ahead bids
        self.reg_up_bids = np.zeros(num_timesteps)     # Store regulation up bids
        self.reg_down_bids = np.zeros(num_timesteps)   # Store regulation down bids
        self.bid_prices = np.zeros(num_timesteps)      # Store bid prices

#%% PV Only Scenario
def pv_only_scenario(x: np.ndarray, pv_data: pd.DataFrame, hourly_data: pd.DataFrame, forecast_data: pd.DataFrame, **params) -> float:
    try:
        # Your existing initialization code stays the same
        num_buses = len(pv_data)
        delta = np.round(x[:num_buses]).astype(int)
        selected_buses = np.where(delta == 1)[0]
        bess_power = x[num_buses:num_buses*2]
        bess_capacity = x[num_buses*2:num_buses*3]
        
        budget_penalty = enforce_budget_optimization(x, num_buses)
        if budget_penalty == float('inf'):
            return float('inf')
        
        total_profit = 0
        results = []
         
        
        # Get hourly and daily patterns for price prediction
        hourly_patterns = get_hourly_price_patterns(hourly_data['Price_Euro_per_MWh'].values, hourly_data.index)
        daily_patterns = get_daily_price_patterns(hourly_data['Price_Euro_per_MWh'].values, hourly_data.index)
        
        for i in selected_buses:
            bess_state = {
                'power': bess_power[i],
                'capacity': MarketParams.get_oversized_capacity(bess_capacity[i]),
                'soc': MarketParams.SOC_MIN,
                'degradation': 0,
                'cycle_count': 0,
                'total_throughput': 0,
                'commitments': {}
            }
            
            ops = MarketOperation()
            ops.is_pv_only = True
            results_tracker = OperationResults(len(hourly_data))
            current_day = -1
            
            # Process each timestep
            for t, row in hourly_data.iterrows():
                hour = t % 24
                day = t // 24
                
                if day != current_day:
                    ops.reset_daily_counters()
                    current_day = day
                    
                    # Prepare DA bids (your existing code)
                    predicted_prices, _, _ = predict_price(
                        t, hourly_data['Price_Euro_per_MWh'].values, hourly_data.index, 
                        hourly_patterns, daily_patterns
                    )
                    
                    bids = prepare_pv_only_bids(
                        pd.Series(predicted_prices),
                        forecast_data[f'pv_gen_{int(pv_data.iloc[i]["bus"])}_forecast'],
                        bess_state,
                        pv_data.iloc[i]['max_grid'],
                        pv_data.iloc[i]['FIT']
                    )
                    
                    bess_state['commitments']['day_ahead'] = {bid['hour']: bid for bid in bids}
                    
                    # Now prepare AFRR bids for the day
                    afrr_bids = prepare_afrr_bids(
                        hourly_data.iloc[t:t+24],
                        bess_state,
                        bess_state['commitments']['day_ahead'],
                        'pv_only'
                    )
                    bess_state['commitments']['afrr'] = afrr_bids

                # Get current conditions
                pv_gen = row[f'pv_gen_{int(pv_data.iloc[i]["bus"])}']
                fip_reference = pv_data.iloc[i]['FIT']
                market_price = row['Price_Euro_per_MWh']
                grid_limit = pv_data.iloc[i]['max_grid']
                
                # 1. First Priority: Fulfill Day-Ahead Bids
                remaining_pv = pv_gen
                bid = bess_state['commitments'].get('day_ahead', {}).get(hour, None)
                
                if bid:
                    if bid['type'] == 'discharge':
                        max_discharge = min(
                            bid['amount'],
                            (bess_state['soc'] - MarketParams.SOC_MIN) * bess_state['capacity'] * MarketParams.BESS_EFF_DISCHARGE,
                            bess_state['power'],
                            grid_limit - results_tracker.p_export[t]
                        )
                        if max_discharge >= MarketParams.DA_MIN_BID:
                            bess_state['soc'] -= (max_discharge / MarketParams.BESS_EFF_DISCHARGE) / bess_state['capacity']
                            results_tracker.p_discharge[t] = max_discharge
                            ops.pv_revenue += max_discharge * market_price

                    elif bid['type'] == 'sell_pv':
                        max_sell = min(bid['amount'], remaining_pv, grid_limit - results_tracker.p_export[t])
                        if max_sell > 0:
                            results_tracker.p_export[t] += max_sell
                            ops.pv_revenue += max_sell * bid['price']
                            remaining_pv -= max_sell

                # 2. Second Priority: Process AFRR Activations
                current_block = hour // AFRRParams.BLOCK_HOURS
                afrr_bid = next((bid for bid in bess_state['commitments'].get('afrr', []) 
                               if bid['block'] == current_block), None)
                
                if afrr_bid:
                    up_activated, down_activated = simulate_afrr_activations(hour, afrr_bid)
                    
                    if up_activated and afrr_bid['reg_up'] > 0:
                        max_reg_up = min(
                            afrr_bid['reg_up'],
                            (bess_state['soc'] - MarketParams.SOC_MIN) * bess_state['capacity'] * MarketParams.BESS_EFF_DISCHARGE,
                            bess_state['power'] - results_tracker.p_discharge[t]
                        )
                        if max_reg_up > 0:
                            results_tracker.p_reg_up[t] = max_reg_up
                            bess_state['soc'] -= (max_reg_up / MarketParams.BESS_EFF_DISCHARGE) / bess_state['capacity']
                            ops.reg_up_revenue += max_reg_up * row['Reg_up']
                
                # 3. Third Priority: Handle remaining PV and charging
                if remaining_pv > 0 and market_price < fip_reference:
                    available_power = bess_state['power'] - results_tracker.p_charge_pv[t]
                    if available_power > 0 and bess_state['soc'] < MarketParams.SOC_MAX:
                        max_charge = min(
                            remaining_pv,
                            (MarketParams.SOC_MAX - bess_state['soc']) * bess_state['capacity'] / MarketParams.BESS_EFF_CHARGE,
                            available_power
                        )
                        if max_charge > 0:
                            bess_state['soc'] += (max_charge * MarketParams.BESS_EFF_CHARGE) / bess_state['capacity']
                            results_tracker.p_charge_pv[t] = max_charge
                            remaining_pv -= max_charge
                
                # Export any remaining PV
                if remaining_pv > 0:
                    remaining_grid_capacity = grid_limit - results_tracker.p_export[t]
                    if remaining_grid_capacity > 0:
                        export_amount = min(remaining_pv, remaining_grid_capacity)
                        results_tracker.p_export[t] += export_amount
                        ops.pv_revenue += export_amount * fip_reference
                
                # Track SOC
                results_tracker.soc[t] = bess_state['soc']
                
                # Add O&M costs
                ops.total_costs += MarketParams.C_VAR_OM * (
                    results_tracker.p_charge_pv[t] + 
                    results_tracker.p_discharge[t] +
                    results_tracker.p_reg_up[t]
                )
            
            results.append({
                'operations': ops,
                'results_tracker': results_tracker,
                'final_state': bess_state
            })
            
            total_profit += ops.get_revenue_breakdown()['net_profit']
        
        global latest_results_pv_only
        latest_results_pv_only = {
            'profit': total_profit,
            'results': results,
            'bess_config': {
                'delta': delta,
                'power': bess_power,
                'capacity': bess_capacity
            }
        }
        
        return float(-total_profit)
        
    except Exception as e:
        print(f"Error in PV-only scenario: {str(e)}")
        traceback.print_exc()
        return float('inf')


#%% PV+Grid Scenario
def pv_grid_scenario(x: np.ndarray, pv_data: pd.DataFrame, hourly_data: pd.DataFrame, forecast_data: pd.DataFrame, **params) -> float:
    try:
        # Initialize as before
        num_buses = len(pv_data)
        delta = np.round(x[:num_buses]).astype(int)
        selected_buses = np.where(delta == 1)[0]
        bess_power = x[num_buses:num_buses*2]
        bess_capacity = x[num_buses*2:num_buses*3]
        
        budget_penalty = enforce_budget_optimization(x, num_buses)
        if budget_penalty == float('inf'):
            return float('inf')
        
        total_profit = 0
        results = []
        
        # Get price patterns for prediction
        hourly_patterns = get_hourly_price_patterns(hourly_data['Price_Euro_per_MWh'].values, hourly_data.index)
        daily_patterns = get_daily_price_patterns(hourly_data['Price_Euro_per_MWh'].values, hourly_data.index)
        
        for i in selected_buses:
            bess_state = {
                'power': bess_power[i],
                'capacity': MarketParams.get_oversized_capacity(bess_capacity[i]),
                'soc': MarketParams.SOC_MIN,
                'degradation': 0,
                'cycle_count': 0,
                'total_throughput': 0,
                'commitments': {}
            }
            
            ops = MarketOperation()
            ops.is_pv_only = False  # PV+Grid scenario
            results_tracker = OperationResults(len(hourly_data))
            current_day = -1
            
            for t, row in hourly_data.iterrows():
                hour = t % 24
                day = t // 24
                
                if day != current_day:
                    ops.reset_daily_counters()
                    current_day = day
                    
                    # Prepare DA bids
                    predicted_prices, _, _ = predict_price(
                        t, hourly_data['Price_Euro_per_MWh'].values, hourly_data.index, 
                        hourly_patterns, daily_patterns
                    )
                    
                    bids = prepare_pv_grid_bids(
                        pd.Series(predicted_prices),
                        forecast_data[f'pv_gen_{int(pv_data.iloc[i]["bus"])}_forecast'],
                        bess_state,
                        pv_data.iloc[i]['max_grid']
                    )
                    
                    bess_state['commitments']['day_ahead'] = {bid['hour']: bid for bid in bids}
                    
                    # Prepare AFRR bids - now includes both up and down regulation
                    afrr_bids = prepare_afrr_bids(
                        hourly_data.iloc[t:t+24],
                        bess_state,
                        bess_state['commitments']['day_ahead'],
                        'pv_grid'  # Allow both up and down regulation
                    )
                    bess_state['commitments']['afrr'] = afrr_bids
                
                # Get current conditions
                pv_gen = row[f'pv_gen_{int(pv_data.iloc[i]["bus"])}']
                market_price = row['Price_Euro_per_MWh']
                grid_limit = pv_data.iloc[i]['max_grid']
                
                # 1. First Priority: Fulfill Day-Ahead Bids
                remaining_pv = pv_gen
                bid = bess_state['commitments'].get('day_ahead', {}).get(hour, None)
                
                if bid:
                    if bid['type'] == 'discharge':
                        max_discharge = min(
                            bid['amount'],
                            (bess_state['soc'] - MarketParams.SOC_MIN) * bess_state['capacity'] * MarketParams.BESS_EFF_DISCHARGE,
                            bess_state['power'],
                            grid_limit - results_tracker.p_export[t]
                        )
                        if max_discharge >= MarketParams.DA_MIN_BID:
                            bess_state['soc'] -= (max_discharge / MarketParams.BESS_EFF_DISCHARGE) / bess_state['capacity']
                            results_tracker.p_discharge[t] = max_discharge
                            ops.da_market_revenue += max_discharge * market_price
                    
                    elif bid['type'] == 'charge_grid':
                        max_charge = min(
                            bid['amount'],
                            (MarketParams.SOC_MAX - bess_state['soc']) * bess_state['capacity'] / MarketParams.BESS_EFF_CHARGE,
                            bess_state['power'],
                            grid_limit
                        )
                        if max_charge >= MarketParams.DA_MIN_BID:
                            bess_state['soc'] += (max_charge * MarketParams.BESS_EFF_CHARGE) / bess_state['capacity']
                            results_tracker.p_charge_grid[t] = max_charge
                            ops.total_costs += max_charge * market_price
                            ops.daily_grid_charges += 1
                    
                    elif bid['type'] == 'sell_pv':
                        max_sell = min(bid['amount'], remaining_pv, grid_limit - results_tracker.p_export[t])
                        if max_sell > 0:
                            results_tracker.p_export[t] += max_sell
                            ops.da_market_revenue += max_sell * market_price
                            remaining_pv -= max_sell
                
                # 2. Second Priority: Process AFRR Activations
                current_block = hour // AFRRParams.BLOCK_HOURS
                afrr_bid = next((bid for bid in bess_state['commitments'].get('afrr', []) 
                               if bid['block'] == current_block), None)
                
                if afrr_bid and ops.daily_reg_hours < MarketParams.MAX_DAILY_REG_HOURS:
                    up_activated, down_activated = simulate_afrr_activations(hour, afrr_bid)
                    
                    # Handle regulation up
                    if up_activated and afrr_bid['reg_up'] > 0:
                        max_reg_up = min(
                            afrr_bid['reg_up'],
                            (bess_state['soc'] - MarketParams.SOC_MIN) * bess_state['capacity'] * MarketParams.BESS_EFF_DISCHARGE,
                            bess_state['power'] - results_tracker.p_discharge[t]
                        )
                        if max_reg_up > 0:
                            results_tracker.p_reg_up[t] = max_reg_up
                            bess_state['soc'] -= (max_reg_up / MarketParams.BESS_EFF_DISCHARGE) / bess_state['capacity']
                            ops.reg_up_revenue += max_reg_up * row['Reg_up']
                            ops.reg_up_activations += 1
                            ops.reg_up_energy += max_reg_up
                            ops.daily_reg_hours += 1
                    
                    # Handle regulation down
                    if down_activated and afrr_bid['reg_down'] > 0:
                        max_reg_down = min(
                            afrr_bid['reg_down'],
                            (MarketParams.SOC_MAX - bess_state['soc']) * bess_state['capacity'] / MarketParams.BESS_EFF_CHARGE,
                            bess_state['power'] - results_tracker.p_charge_grid[t]
                        )
                        if max_reg_down > 0:
                            results_tracker.p_reg_down[t] = max_reg_down
                            bess_state['soc'] += (max_reg_down * MarketParams.BESS_EFF_CHARGE) / bess_state['capacity']
                            ops.reg_down_revenue += max_reg_down * row['Reg_down']
                            ops.reg_down_activations += 1
                            ops.reg_down_energy += max_reg_down
                            ops.daily_reg_hours += 1
                
                # 3. Third Priority: Handle remaining PV
                if remaining_pv > 0:
                    # Try charging BESS with remaining PV
                    available_power = bess_state['power'] - (results_tracker.p_charge_grid[t] + results_tracker.p_reg_down[t])
                    if available_power > 0 and bess_state['soc'] < MarketParams.SOC_MAX:
                        max_charge = min(
                            remaining_pv,
                            (MarketParams.SOC_MAX - bess_state['soc']) * bess_state['capacity'] / MarketParams.BESS_EFF_CHARGE,
                            available_power
                        )
                        if max_charge > 0:
                            bess_state['soc'] += (max_charge * MarketParams.BESS_EFF_CHARGE) / bess_state['capacity']
                            results_tracker.p_charge_pv[t] = max_charge
                            remaining_pv -= max_charge
                    
                    # Export any remaining PV
                    if remaining_pv > 0:
                        remaining_grid_capacity = grid_limit - results_tracker.p_export[t]
                        if remaining_grid_capacity > 0:
                            export_amount = min(remaining_pv, remaining_grid_capacity)
                            results_tracker.p_export[t] += export_amount
                            ops.da_market_revenue += export_amount * market_price
                
                # Track SOC
                results_tracker.soc[t] = bess_state['soc']
                
                # Add O&M costs
                ops.total_costs += MarketParams.C_VAR_OM * (
                    results_tracker.p_charge_pv[t] + 
                    results_tracker.p_charge_grid[t] +
                    results_tracker.p_discharge[t] +
                    results_tracker.p_reg_up[t] +
                    results_tracker.p_reg_down[t]
                )
            
            results.append({
                'operations': ops,
                'results_tracker': results_tracker,
                'final_state': bess_state
            })
            
            total_profit += ops.get_revenue_breakdown()['net_profit']
        
        global latest_results_pv_grid
        latest_results_pv_grid = {
            'profit': total_profit,
            'results': results,
            'bess_config': {
                'delta': delta,
                'power': bess_power,
                'capacity': bess_capacity
            }
        }
        
        return float(-total_profit)
        
    except Exception as e:
        print(f"Error in PV+Grid scenario: {str(e)}")
        traceback.print_exc()
        return float('inf')

#%% Main Execution
def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    import numpy as np
    import random
    
    np.random.seed(seed)
    random.seed(seed)

latest_results_pv_only = None
latest_results_pv_grid = None
class OptimizationParams:
    # No changes needed in PSO parameters
    N_PARTICLES =1
    MAX_ITERATIONS = 1
    COGNITIVE_WEIGHT = 0.7
    SOCIAL_WEIGHT = 0.3    
    INERTIA_WEIGHT = 0.9   
    
    # BESS Limits - May need adjustment for FiP
    MAX_POWER = 4   # MW
    MAX_ENERGY = 8  # MWh
    MIN_POWER = 1 # MW
    MIN_ENERGY = 1.2 # MWh
    
    # Same convergence criteria
    MIN_DELTA = 1e-5
    STALL_ITERATIONS = 15
    

def initialize_particles(n_particles: int, n_buses: int, n_selected: int = 5, scenario_type: str = 'pv_only') -> np.ndarray:
    n_dimensions = n_buses * 3
    init_pos = np.zeros((n_particles, n_dimensions))
    
    for i in range(n_particles):
        # Select random buses for BESS placement
        selected = np.random.choice(n_buses, size=n_selected, replace=False)
        init_pos[i, selected] = 1
        
        for j in selected:
            pv_capacity = pv_data.iloc[j]['P_MWp']
            # Maximum possible power considering both power and energy costs
            max_power_with_min_duration = MarketParams.MAX_PROJECT_BUDGET / (
                MarketParams.BESS_COST_P * 1000 + 
                MarketParams.MIN_DURATION * MarketParams.BESS_COST_E * 1000
            )
            
            max_power_with_max_duration = MarketParams.MAX_PROJECT_BUDGET / (
                MarketParams.BESS_COST_P * 1000 + 
                MarketParams.MAX_DURATION * MarketParams.BESS_COST_E * 1000
            )
            
            # Consider original constraints as well
            max_power = min(
                max_power_with_min_duration,
                max_power_with_max_duration,
                OptimizationParams.MAX_POWER
            )
            
            # Initialize power within constraints
            power = np.random.uniform(OptimizationParams.MIN_POWER, max_power)
            init_pos[i, n_buses + j] = power
            
            # Calculate possible energy range for this power
            min_energy = power * MarketParams.MIN_DURATION
            max_energy_pv = pv_capacity * MarketParams.MAX_DURATION
            
            # Maximum energy considering budget constraint
            # BUDGET = P*COST_P + E*COST_E
            # E = (BUDGET - P*COST_P)/COST_E
            max_energy_budget = (MarketParams.MAX_PROJECT_BUDGET - 
                               power * MarketParams.BESS_COST_P * 1000) / (MarketParams.BESS_COST_E * 1000)

            max_energy = min(
                power * MarketParams.MAX_DURATION,
                max_energy_budget,
                max_energy_pv,
                OptimizationParams.MAX_ENERGY
            )
            
            # Verify we have valid range
            if max_energy < min_energy:
                # If invalid, adjust power down to ensure feasible energy range
                power = (MarketParams.MAX_PROJECT_BUDGET / 
                        (MarketParams.BESS_COST_P * 1000 + 
                         MarketParams.MIN_DURATION * MarketParams.BESS_COST_E * 1000))
                init_pos[i, n_buses + j] = power
                min_energy = power * MarketParams.MIN_DURATION
                max_energy = power * MarketParams.MAX_DURATION
            
            # Initialize energy within valid range
            energy = np.random.uniform(min_energy, max_energy)
            init_pos[i, n_buses*2 + j] = energy
            
            # Verify total cost is within budget
            total_cost = (power * MarketParams.BESS_COST_P * 1000 + 
                         energy * MarketParams.BESS_COST_E * 1000)
            
            if total_cost > MarketParams.MAX_PROJECT_BUDGET:
                # Scale both power and energy down proportionally to meet budget
                scale_factor = MarketParams.MAX_PROJECT_BUDGET / total_cost
                init_pos[i, n_buses + j] *= scale_factor
                init_pos[i, n_buses*2 + j] *= scale_factor
    
    return init_pos

def run_optimization(scenario_type, pv_data, hourly_data, forecast_data):
    print(f"\nStarting {scenario_type} optimization...")
    #np.random.seed(42)
    #random.seed(42)
    n_buses = len(pv_data)
    n_dimensions = n_buses * 3
    
    # Same bounds
    lb = np.zeros(n_dimensions)
    ub = np.ones(n_dimensions)
    for i in range(n_buses):
    # For power rating - use original constraints
        ub[n_buses + i] = OptimizationParams.MAX_POWER
    
    # For energy capacity - use power and duration constraints
        max_energy = min(
            OptimizationParams.MAX_ENERGY,
            (MarketParams.MAX_PROJECT_BUDGET) / (MarketParams.BESS_COST_E * 1000)
        )
        ub[n_buses*2 + i] = max_energy
    #ub[n_buses:n_buses*2] = OptimizationParams.MAX_POWER
    #ub[n_buses*2:] = OptimizationParams.MAX_ENERGY
    
    # Initialize with updated initialization function
    init_pos = initialize_particles(OptimizationParams.N_PARTICLES, n_buses, scenario_type=scenario_type)
    
    # Rest remains the same
    optimizer = GlobalBestPSO(
        n_particles=OptimizationParams.N_PARTICLES,
        dimensions=n_dimensions,
        options={
            'c1': OptimizationParams.COGNITIVE_WEIGHT,
            'c2': OptimizationParams.SOCIAL_WEIGHT,
            'w': OptimizationParams.INERTIA_WEIGHT
        },
        bounds=(lb, ub),
        init_pos=init_pos
    )
    
    # Select scenario function
    scenario_func = pv_only_scenario if scenario_type == 'pv_only' else pv_grid_scenario
    
    # Evaluate initial positions to find best
    initial_costs = np.array([scenario_func(pos, pv_data, hourly_data, forecast_data) 
                            for pos in init_pos])
    best_idx = np.argmin(initial_costs)
    
    # Set initial best position and cost
    optimizer.swarm.best_pos = init_pos[best_idx].copy()
    optimizer.swarm.best_cost = initial_costs[best_idx]
    
    try:
        # Run optimization with convergence monitoring
        prev_best_cost = float('inf')
        stall_count = 0
        best_cost = float('inf')
        best_pos = None
        
        for iteration in range(OptimizationParams.MAX_ITERATIONS):
            cost, pos = optimizer.optimize(
                scenario_func,
                iters=1,  # Run one iteration at a time
                pv_data=pv_data,
                hourly_data=hourly_data,
                forecast_data=forecast_data
            )
            
            # Check for improvement
            if abs(cost - prev_best_cost) < OptimizationParams.MIN_DELTA:
                stall_count += 1
            else:
                stall_count = 0
            
            # Update best solution
            if cost < best_cost:
                best_cost = cost
                best_pos = pos.copy()
            
            prev_best_cost = cost
            
            # Check for convergence
            if stall_count >= OptimizationParams.STALL_ITERATIONS:
                print(f"Optimization converged after {iteration+1} iterations")
                break
        
        # Return best found solution
        if best_pos is not None:
            print(f"\n{scenario_type} optimization completed successfully:")
            print(f"Best cost: {-best_cost:,.2f} €")  # Negative since we minimized negative profit
            print(f"Number of selected projects: {int(np.sum(np.round(best_pos[:n_buses])))}")
            return best_cost, best_pos
        else:
            return float('inf'), None
        
    except Exception as e:
        print(f"Error during {scenario_type} optimization: {str(e)}")
        traceback.print_exc()
        return float('inf'), None

# Add main execution with try-catch
if __name__ == "__main__":
    try:
        # First verify the data
        print("\nVerifying input data...")
        print(f"PV data shape: {pv_data.shape}")
        print(f"Hourly data shape: {hourly_data.shape}")
        print(f"Forecast data shape: {forecast_data.shape}")
        
        # Run PV-only optimization
        cost_pv_only, pos_pv_only = run_optimization(
            'pv_only',
            pv_data,
            hourly_data,
            forecast_data
        )
        
        # Run PV+Grid optimization
        cost_pv_grid, pos_pv_grid = run_optimization(
            'pv_grid',
            pv_data,
            hourly_data,
            forecast_data
        )
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        traceback.print_exc()
        
#%% Printing for Results
def create_top_projects_summary(pv_only_results, pv_grid_results, pv_data):
    summaries = {
        'pv_only': [],
        'pv_grid': []
    }
    
    for scenario, results in [('pv_only', pv_only_results), ('pv_grid', pv_grid_results)]:
        selected_buses = np.where(results['bess_config']['delta'] == 1)[0]
        
        sorted_projects = sorted(
            enumerate(zip(results['results'], selected_buses)),
            key=lambda x: x[1][0]['operations'].get_revenue_breakdown()['net_profit'],
            reverse=True
        )[:5]
        
        for rank, (_, (project, bus_idx)) in enumerate(sorted_projects, 1):
            ops = project['operations']
            tracker = project['results_tracker']
            final_state = project['final_state']
            revenue_breakdown = ops.get_revenue_breakdown()
            
            pv_plant = pv_data.iloc[bus_idx]
            
            total_discharge = np.sum(tracker.p_discharge)
            total_charge = np.sum(tracker.p_charge + tracker.p_charge_pv + tracker.p_charge_grid)
            annual_cycles = total_discharge / (2 * final_state['capacity'])
            
            capex = final_state['capacity'] * 450000
            annual_opex = capex * 0.02
            
            if scenario == 'pv_only':
                annual_revenue = revenue_breakdown['pv_fit'] + revenue_breakdown['regulation_up']
            else:
                annual_revenue = (revenue_breakdown['day_ahead'] + 
                                revenue_breakdown['regulation_up'] + 
                                revenue_breakdown['regulation_down'])
            
            annual_net_revenue = annual_revenue - revenue_breakdown['total_costs']-annual_opex
            payback_period = capex / annual_net_revenue if annual_net_revenue > 0 else float('inf')
            roi = (annual_net_revenue / capex) * 100 if capex > 0 else 0
            
            project_summary = {
                'rank': rank,
                'bus_number': int(pv_plant['bus']),
                'scenario': scenario,
                'bess_power_mw': final_state['power'],
                'bess_energy_mwh': final_state['capacity'],
                'power_to_energy_ratio': final_state['power'] / final_state['capacity'],
                'pv_capacity_mw': pv_plant['P_MWp'],
                'grid_connection_mw': pv_plant['max_grid'],
                'fit_price': pv_plant['FIT'],
                'annual_cycles': annual_cycles,
                'total_discharge_mwh': total_discharge,
                'total_charge_mwh': total_charge,
                'average_soc_percent': np.mean(tracker.soc) * 100,
                'soc_std_percent': np.std(tracker.soc) * 100,
                'final_degradation_percent': final_state['degradation'] * 100,
                'reg_up_hours': np.sum(tracker.p_reg_up > 0),
                'reg_down_hours': np.sum(tracker.p_reg_down > 0),
                'capex': capex,
                'annual_opex': annual_opex,
                'annual_revenue': annual_revenue,
                'annual_charging_costs': revenue_breakdown['total_costs'],
                'annual_net_revenue': annual_net_revenue,
                'payback_years': payback_period,
                'roi_percent': roi,
                'revenue_breakdown': revenue_breakdown
            }
            
            summaries[scenario].append(project_summary)
    
    return summaries
####################################################################################################################
# Generate summaries
top_projects = create_top_projects_summary(latest_results_pv_only, latest_results_pv_grid, pv_data)
####################################################################################################################
# Print formatted summaries
def print_project_summaries(summaries):
    for scenario in ['pv_only', 'pv_grid']:
        print(f"\n{'='*40} {scenario.upper()} SCENARIO {'='*40}")
        for project in summaries[scenario]:
            print(f"\nRANK {project['rank']} - BUS {project['bus_number']}")
            print("-" * 80)
            
            print("\nBESS Configuration:")
            print(f"Power Rating: {project['bess_power_mw']:.2f} MW")
            print(f"Energy Capacity: {project['bess_energy_mwh']:.2f} MWh")
            print(f"P/E Ratio: {project['power_to_energy_ratio']:.2f}")
            
            print("\nPV Plant Information:")
            print(f"PV Capacity: {project['pv_capacity_mw']:.2f} MW")
            print(f"Grid Connection: {project['grid_connection_mw']:.2f} MW")
            print(f"FiT Price: {project['fit_price']:.2f} €/MWh")
            
            print("\nOperational Metrics:")
            print(f"Annual Cycles: {project['annual_cycles']:.1f}")
            print(f"Average SOC: {project['average_soc_percent']:.1f}% (±{project['soc_std_percent']:.1f}%)")
            print(f"Final Degradation: {project['final_degradation_percent']:.2f}%")
            print(f"Total Discharge: {project['total_discharge_mwh']:.1f} MWh")
            print(f"Total Charge: {project['total_charge_mwh']:.1f} MWh")
            
            print("\nGrid Services:")
            print(f"Regulation Up Hours: {project['reg_up_hours']:.0f}")
            print(f"Regulation Down Hours: {project['reg_down_hours']:.0f}")
            
            print("\nFinancial Metrics:")
            print(f"CAPEX: {project['capex']:,.0f} €")
            print(f"Annual OPEX: {project['annual_opex']:,.0f} €")
            print(f"Annual Revenue: {project['annual_revenue']:,.0f} €")
            print(f"Annual Charging Costs: {project['annual_charging_costs']:,.0f} €")
            print(f"Annual Net Revenue: {project['annual_net_revenue']:,.0f} €")
            print(f"Payback Period: {project['payback_years']:.1f} years")
            print(f"ROI: {project['roi_percent']:.1f}%")
            
            print("\nRevenue Breakdown:")
            for key, value in project['revenue_breakdown'].items():
                print(f"{key}: {value:,.0f} €")
            
            print("\n" + "="*80)

# Print the summaries
print_project_summaries(top_projects)

# Save the data for future use
import pickle

with open('top_projects_summary.pkl', 'wb') as f:
    pickle.dump(top_projects, f)

print("\nData has been saved to 'top_projects_summary.pkl'")

with open('top_projects_summary.pkl', 'rb') as f:
    top_projects = pickle.load(f)

#####################################################################################################################
def plot_pv_bess_comparison_top_projects(top_projects):
    # Set style for professional plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Define colors
    colors = {
        'pv': '#4CAF50',    # Green for PV
        'power': '#2196F3',  # Blue for BESS power
        'energy': '#FFC107'  # Yellow for BESS energy
    }
    
    # Prepare data for both scenarios
    for ax, scenario, title in [(ax1, 'pv_only', 'PV-only Scenario'),
                               (ax2, 'pv_grid', 'PV+Grid Scenario')]:
        x = np.arange(5)
        width = 0.25
        
        # Extract data
        pv_sizes = [p['pv_capacity_mw'] for p in top_projects[scenario]]
        bess_power = [p['bess_power_mw'] for p in top_projects[scenario]]
        bess_energy = [p['bess_energy_mwh'] for p in top_projects[scenario]]
        bus_numbers = [p['bus_number'] for p in top_projects[scenario]]
        
        # Create bars
        ax.bar(x - width, pv_sizes, width, label='PV Plant Size (MW)', color=colors['pv'])
        ax.bar(x, bess_power, width, label='BESS Power (MW)', color=colors['power'])
        ax.bar(x + width, bess_energy, width, label='BESS Energy (MWh)', color=colors['energy'])
        
        # Customize plot
        ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
        ax.set_ylabel('Size (MW/MWh)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'P{i+1}\nBus {bus}' for i, bus in enumerate(bus_numbers)])
        
        # Add value labels
        for i, values in enumerate(zip(pv_sizes, bess_power, bess_energy)):
            for j, v in enumerate(values):
                ax.text(i + (j-1)*width, v, f'{v:.1f}',
                       ha='center', va='bottom', fontsize=10)
        
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('PV Plant and BESS Size Comparison - Top 5 Projects',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

# Plot both visualizations
plot_pv_bess_comparison_top_projects(top_projects)
####################################################################################################################
def plot_bess_operation_profiles_top(top_projects, scenario_type, start_hour=4344, duration_hours=168):
    # Set style for professional plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(5, 1, figsize=(15, 25))
    time_range = range(duration_hours)
    
    # Get project data - fix the key formatting
    scenario_key = 'pv_only' if scenario_type.lower() == 'pv-only' else 'pv_grid'
    projects = top_projects[scenario_key]
    
    for idx, project in enumerate(projects):
        # Extract the results tracker from the operations data
        results_tracker = latest_results_pv_only['results'][idx] if scenario_key == 'pv_only' else latest_results_pv_grid['results'][idx]
        results_tracker = results_tracker['results_tracker']
        time_slice = slice(start_hour, start_hour + duration_hours)
        
        ax_twin = axes[idx].twinx()
        
        # Separate charging components
        pv_charge = -results_tracker.p_charge_pv[time_slice]
        grid_charge = -results_tracker.p_charge_grid[time_slice]
        discharge = results_tracker.p_discharge[time_slice]
        
        # PV generation and export
        pv_gen = results_tracker.p_export[time_slice] + np.abs(results_tracker.p_charge_pv[time_slice])
        pv_export = results_tracker.p_export[time_slice]
        
        # Plot base layers with professional colors
        axes[idx].plot(time_range, pv_gen,
                      label='PV Generation', color='#2ECC71', linestyle='-')
        axes[idx].plot(time_range, pv_export,
                      label='PV Grid Export', color='#2ECC71', linestyle='--', alpha=0.5)
        
        # Plot charging and discharging
        axes[idx].fill_between(time_range, 0, pv_charge,
                             label='PV Charging', alpha=0.3, color='#3498DB')
        if scenario_key == 'pv_grid':
            axes[idx].fill_between(time_range, pv_charge, pv_charge + grid_charge,
                                 label='Grid Charging', alpha=0.3, color='#9B59B6')
        axes[idx].fill_between(time_range, 0, discharge,
                             label='Discharging', alpha=0.3, color='#E74C3C')
        
        # Plot SOC with thicker line
        ax_twin.plot(time_range, results_tracker.soc[time_slice] * 100,
                    label='State of Charge', color='#2C3E50', linewidth=2)
        
        # Formatting
        title = f'Project {idx + 1} (Bus {project["bus_number"]}) - Operation Profile'
        axes[idx].set_title(title, fontsize=12, fontweight='bold', pad=15)
        
        if idx == len(projects) - 1:  # Only show xlabel for bottom plot
            axes[idx].set_xlabel('Hour', fontsize=10)
        axes[idx].set_ylabel('Power (MW)', fontsize=10)
        ax_twin.set_ylabel('State of Charge (%)', fontsize=10)
        ax_twin.set_ylim([0, 100])
        
        # Add PV and BESS size information
        info_text = (f'PV: {project["pv_capacity_mw"]:.1f} MW | '
                    f'BESS: {project["bess_power_mw"]:.1f} MW / {project["bess_energy_mwh"]:.1f} MWh')
        axes[idx].text(0.02, 0.98, info_text,
                      transform=axes[idx].transAxes,
                      ha='left', va='top',
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Dynamic y-axis limits
        max_power = max(
            abs(np.min(pv_charge)),
            abs(np.min(grid_charge)) if scenario_key == 'pv_grid' else 0,
            np.max(discharge),
            np.max(pv_gen)
        )
        axes[idx].set_ylim([-max_power*1.1, max_power*1.1])
        
        # Combined legend with better positioning
        lines_1, labels_1 = axes[idx].get_legend_handles_labels()
        lines_2, labels_2 = ax_twin.get_legend_handles_labels()
        axes[idx].legend(lines_1 + lines_2, labels_1 + labels_2, 
                        loc='center left', bbox_to_anchor=(1.05, 0.5),
                        fancybox=True, shadow=True, ncol=1)
        
        # Enhance grid
        axes[idx].grid(True, alpha=0.3, linestyle='--')
        
        # Add hour markers
        axes[idx].set_xticks(np.arange(0, duration_hours, 24))
        axes[idx].set_xticklabels([f'{h}' for h in range(0, duration_hours, 24)])
    
    week_num = start_hour//168 + 1
    month = (week_num * 7) // 30 + 1
    season = "Summer" if 6 <= month <= 8 else "Winter" if month in [12, 1, 2] else "Spring" if 3 <= month <= 5 else "Fall"
    
    plt.suptitle(f'BESS Operation Profiles - {scenario_type} Scenario\n'
                 f'Week {week_num} ({season})',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()

# Example usage:
# Plot both winter periods (Week 3, January)
if top_projects is not None:
    plot_bess_operation_profiles_top(top_projects, "PV-only", start_hour=8016)
    plot_bess_operation_profiles_top(top_projects, "PV+Grid", start_hour=8016)

# Plot summer periods for comparison (Week 26, July)
if top_projects is not None:
    plot_bess_operation_profiles_top(top_projects, "PV-only", start_hour=4344)
    plot_bess_operation_profiles_top(top_projects, "PV+Grid", start_hour=4344)
####################################################################################################################
def print_battery_metrics(results_dict, scenario_type, top_projects):
    scenario_key = 'pv_only' if scenario_type == "PV-only" else 'pv_grid'
    print(f"\n{scenario_type} Scenario Battery Metrics")
    print("=" * 80)
    
    for project in top_projects[scenario_key]:
        project_idx = project['rank'] - 1  # Convert rank to index
        tracker = results_dict['results'][project_idx]['results_tracker']
        final_state = results_dict['results'][project_idx]['final_state']
        
        total_discharge = np.sum(tracker.p_discharge)
        total_charge = np.sum(tracker.p_charge + tracker.p_charge_pv + tracker.p_charge_grid)
        capacity = final_state['capacity']
        annual_cycles = total_discharge / (2 * capacity)
        
        avg_soc = np.mean(tracker.soc) * 100
        soc_std = np.std(tracker.soc) * 100
        
        max_possible_throughput = capacity * 365 * 2
        actual_throughput = total_discharge + total_charge
        utilization_factor = (actual_throughput / max_possible_throughput) * 100
        
        degradation = final_state['degradation'] * 100
        reg_up_hours = np.sum(tracker.p_reg_up > 0)
        reg_down_hours = np.sum(tracker.p_reg_down > 0)
        
        print(f"\nProject {project['rank']} (Bus {project['bus_number']}):")
        print("-" * 40)
        print(f"Battery Size: {capacity:.2f} MWh, {final_state['power']:.2f} MW")
        print("\nOperational Metrics:")
        print(f"Annual Full Equivalent Cycles: {annual_cycles:.1f}")
        print(f"Average State of Charge: {avg_soc:.1f}% (std: {soc_std:.1f}%)")
        print(f"Utilization Factor: {utilization_factor:.1f}%")
        print(f"Final Degradation: {degradation:.2f}%")
        
        print("\nEnergy Metrics:")
        print(f"Total Charging Energy: {total_charge:.1f} MWh")
        print(f"Total Discharging Energy: {total_discharge:.1f} MWh")
        print(f"Total Energy Throughput: {final_state['total_throughput']:.1f} MWh")
        
        print("\nGrid Services:")
        print(f"Regulation Up Hours: {reg_up_hours:.0f}")
        print(f"Regulation Down Hours: {reg_down_hours:.0f}")
        
        soc_array = np.array(tracker.soc) * 100
        soc_percentiles = {
            'min': np.min(soc_array),
            '25th': np.percentile(soc_array, 25),
            'median': np.median(soc_array),
            '75th': np.percentile(soc_array, 75),
            'max': np.max(soc_array)
        }
        
        print("\nSOC Distribution:")
        print(f"Min SOC: {soc_percentiles['min']:.1f}%")
        print(f"25th Percentile: {soc_percentiles['25th']:.1f}%")
        print(f"Median SOC: {soc_percentiles['median']:.1f}%")
        print(f"75th Percentile: {soc_percentiles['75th']:.1f}%")
        print(f"Max SOC: {soc_percentiles['max']:.1f}%")
        
        cycles_by_month = []
        for month in range(12):
            start_idx = month * 730
            end_idx = start_idx + 730
            monthly_discharge = np.sum(tracker.p_discharge[start_idx:end_idx])
            monthly_cycles = monthly_discharge / (2 * capacity)
            cycles_by_month.append(monthly_cycles)
        
        print("\nMonthly Cycling Pattern:")
        for month, cycles in enumerate(cycles_by_month, 1):
            print(f"Month {month}: {cycles:.1f} cycles")

# Print metrics using the saved top_projects
print_battery_metrics(latest_results_pv_only, "PV-only", top_projects)
print_battery_metrics(latest_results_pv_grid, "PV+Grid", top_projects)
####################################################################################################################
def calculate_npv(capex, annual_revenue, annual_opex, annual_charging_cost, 
                discount_rate=0.02, lifetime=20, degradation_rate=0.02):
    
    # Initial investment (negative cash flow)
    npv = -capex
    
    # Calculate yearly cash flows

    for year in range(1, lifetime + 1):
        # Apply degradation to revenue and charging cost
        degradation_factor = (1 - degradation_rate) ** (year - 1)
        year_revenue = annual_revenue * degradation_factor
        year_charging = annual_charging_cost * degradation_factor
        
        # Calculate net cash flow for the year
        cash_flow = year_revenue - annual_opex - year_charging
        
        # Discount the cash flow and add to NPV
        npv += cash_flow / (1 + discount_rate) ** year

    return npv
####################################################################################################################
def analyze_project_economics_fixed(power, energy, ops):
   #power_cost = power * MarketParams.BESS_COST_P * 1000
   energy_cost = energy * MarketParams.BESS_COST_E * 1000
   total_capex = energy_cost
   
   revenue_breakdown = ops.get_revenue_breakdown()
   
   if ops.is_pv_only:
       annual_revenue = (
           revenue_breakdown['pv_fit'] +
           revenue_breakdown['regulation_up']
       )
   else:
       annual_revenue = (
           revenue_breakdown['day_ahead'] +
           revenue_breakdown['regulation_up'] +
           revenue_breakdown['regulation_down']
       )
   
   annual_opex = total_capex * 0.02
   
   npv = calculate_npv(
       capex=total_capex,
       annual_revenue=annual_revenue,
       annual_opex=annual_opex,
       annual_charging_cost=0
       #annual_charging_cost=revenue_breakdown['total_costs']
   )
   
   annual_net_revenue = annual_revenue - annual_opex - revenue_breakdown['total_costs']
   #annual_net_revenue = annual_revenue - annual_opex 
   payback_period = total_capex / annual_net_revenue if annual_net_revenue > 0 else float('inf')
   roi = (annual_net_revenue / total_capex) * 100

   return {
       'npv': npv,
       'capex': total_capex,
       'annual_revenue': annual_revenue,
       'annual_opex': annual_opex,
       'annual_net_revenue': annual_net_revenue,
       'payback_period': payback_period,
       'roi': roi
   }


def print_economic_analysis(results_dict, scenario_type, top_projects):
    scenario_key = 'pv_only' if scenario_type == "PV-only" else 'pv_grid'
    print(f"\n{scenario_type} Projects Economic Analysis:")
    print("=" * 80)
    
    for project in top_projects[scenario_key]:
        project_idx = project['rank'] - 1
        ops = results_dict['results'][project_idx]['operations']
        
        # Use the power and energy from top_projects instead of final_state
        power = project['bess_power_mw']
        energy = project['bess_energy_mwh']
        
        economics = analyze_project_economics_fixed(power, energy, ops)
        
        print(f"\nProject {project['rank']} (Bus {project['bus_number']}):")
        print(f"Power Rating: {power:.2f} MW")
        print(f"Energy Capacity: {energy:.2f} MWh")
        print(f"CAPEX: {economics['capex']/1e6:.2f}M €")
        print(f"Annual Revenue: {economics['annual_revenue']/1e3:.2f}k €")
        print(f"Annual OPEX: {economics['annual_opex']/1e3:.2f}k €")
        print(f"Annual Net Revenue: {economics['annual_net_revenue']/1e3:.2f}k €")
        print(f"NPV: {economics['npv']/1e6:.2f}M €")
        print(f"Payback Period: {economics['payback_period']:.1f} years")
        print(f"ROI: {economics['roi']:.1f}%")

print_economic_analysis(latest_results_pv_only, "PV-only", top_projects)
print_economic_analysis(latest_results_pv_grid, "PV+Grid", top_projects)

####################################################################################
def analyze_charging_operations(results_dict, scenario_type, top_projects):
   """Analyze charging/discharging operations using same project selection as create_top_projects_summary"""
   scenario_key = 'pv_only' if scenario_type == 'PV-only' else 'pv_grid'
   results = []
   
   for project in top_projects[scenario_key]:
       project_idx = project['rank'] - 1
       tracker = results_dict['results'][project_idx]['results_tracker']
       ops = results_dict['results'][project_idx]['operations']
       
       total_charge = 0
       total_discharge = 0
       total_charge_cost = 0
       total_discharge_revenue = 0
       
       for t in range(len(tracker.p_charge)):
           charge_power = (tracker.p_charge[t] + 
                         tracker.p_charge_pv[t] + 
                         tracker.p_charge_grid[t])
           
           discharge_power = tracker.p_discharge[t]
           market_price = hourly_data['Price_Euro_per_MWh'].iloc[t]
           
           if charge_power > 0:
               total_charge += charge_power
               total_charge_cost += charge_power * market_price
           
           if discharge_power > 0:
               total_discharge += discharge_power
               total_discharge_revenue += discharge_power * market_price
       
       battery_capacity = results_dict['results'][project_idx]['final_state']['capacity']
       annual_cycles = total_discharge / (battery_capacity * 2)
       
       avg_charge_price = total_charge_cost / total_charge if total_charge > 0 else 0
       avg_discharge_price = total_discharge_revenue / total_discharge if total_discharge > 0 else 0
       
       results.append({
           'project': project['rank'],
           'bus': project['bus_number'],
           'total_charge_mwh': total_charge,
           'total_discharge_mwh': total_discharge,
           'avg_charge_price': avg_charge_price,
           'avg_discharge_price': avg_discharge_price,
           'annual_cycles': annual_cycles,
           'total_charge_cost': total_charge_cost,
           'total_discharge_revenue': total_discharge_revenue,
           'price_spread': avg_discharge_price - avg_charge_price
       })
   
   return results
####################################################################################################################
def print_charging_analysis(top_projects):
   print("\nCharging Analysis for PV-only Scenario:")
   print("=" * 100)
   pv_only_results = analyze_charging_operations(latest_results_pv_only, 'PV-only', top_projects)
   for result in pv_only_results:
       print(f"\nProject {result['project']} (Bus {result['bus']}):")
       print(f"Total Charging: {result['total_charge_mwh']:.2f} MWh")
       print(f"Total Discharging: {result['total_discharge_mwh']:.2f} MWh")
       print(f"Average Charge Price: {result['avg_charge_price']:.2f} €/MWh")
       print(f"Average Discharge Price: {result['avg_discharge_price']:.2f} €/MWh")
       print(f"Price Spread: {result['price_spread']:.2f} €/MWh")
       print(f"Annual Cycles: {result['annual_cycles']:.2f}")
       print(f"Total Charging Cost: {result['total_charge_cost']/1000:.2f}k €")
       print(f"Total Discharge Revenue: {result['total_discharge_revenue']/1000:.2f}k €")
   
   print("\nCharging Analysis for PV+Grid Scenario:")
   print("=" * 100)
   pv_grid_results = analyze_charging_operations(latest_results_pv_grid, 'PV+Grid', top_projects)
   for result in pv_grid_results:
       print(f"\nProject {result['project']} (Bus {result['bus']}):")
       print(f"Total Charging: {result['total_charge_mwh']:.2f} MWh")
       print(f"Total Discharging: {result['total_discharge_mwh']:.2f} MWh")
       print(f"Average Charge Price: {result['avg_charge_price']:.2f} €/MWh")
       print(f"Average Discharge Price: {result['avg_discharge_price']:.2f} €/MWh")
       print(f"Price Spread: {result['price_spread']:.2f} €/MWh")
       print(f"Annual Cycles: {result['annual_cycles']:.2f}")
       print(f"Total Charging Cost: {result['total_charge_cost']/1000:.2f}k €")
       print(f"Total Discharge Revenue: {result['total_discharge_revenue']/1000:.2f}k €")

print_charging_analysis(top_projects)

####################################################################################
def calculate_corrected_lcos(result, project_power, project_energy):
    YEARS = 20
    DISCOUNT_RATE = 0.02
    
    # Initial CAPEX
    energy_cost = project_energy * MarketParams.BESS_COST_E * 1000
    power_cost = project_power * MarketParams.BESS_COST_P * 1000
    capex_initial = energy_cost + power_cost
    
    # Annual values
    annual_discharge = result['total_discharge_mwh']
    annual_charge_cost = result['total_charge_cost']
    annual_opex = capex_initial * 0.02
    
    # Initialize summation terms
    sum_opex = 0
    sum_charging_cost = 0
    sum_electricity = 0
    
    # Calculate yearly terms
    for year in range(YEARS):
        discount_factor = (1 + DISCOUNT_RATE) ** year
        degradation_factor = (1 - MarketParams.ANNUAL_CALENDAR_DEG) ** year
        
        # Add OPEX term
        sum_opex += annual_opex / discount_factor
        
        # Add charging cost term
        sum_charging_cost += (annual_charge_cost * degradation_factor) / discount_factor
        
        # Add electricity discharged term
        sum_electricity += (annual_discharge * degradation_factor) / discount_factor
    
    # Calculate LCOS according to formula
    if sum_electricity > 0:
        lcos = (capex_initial + sum_opex + sum_charging_cost) / sum_electricity
    else:
        lcos = float('inf')
    
    total_costs = capex_initial + sum_opex + sum_charging_cost
    
    return lcos, total_costs, sum_electricity
####################################################################################################################
def print_and_plot_lcos_analysis(top_projects, latest_results_pv_only, latest_results_pv_grid):
   print("\nCorrected LCOS Analysis:")
   print("=" * 80)
   
   pv_only_results = analyze_charging_operations(latest_results_pv_only, 'PV-only', top_projects)
   print("\nPV-only Scenario:")
   for result in pv_only_results:
       project_idx = result['project'] - 1
       power = latest_results_pv_only['results'][project_idx]['final_state']['power']
       energy = latest_results_pv_only['results'][project_idx]['final_state']['capacity']
       
       lcos, total_cost, total_energy = calculate_corrected_lcos(result, power, energy)
       print(f"\nProject {result['project']} (Bus {result['bus']}):")
       print(f"Power Rating: {power:.2f} MW")
       print(f"Energy Capacity: {energy:.2f} MWh")
       print(f"LCOS: {lcos:.2f} €/MWh")
       print(f"Total Lifetime Cost: {total_cost/1e6:.2f}M €")
       print(f"Total Lifetime Energy: {total_energy:.2f} MWh")
   
   pv_grid_results = analyze_charging_operations(latest_results_pv_grid, 'PV+Grid', top_projects)
   print("\nPV+Grid Scenario:")
   for result in pv_grid_results:
       project_idx = result['project'] - 1
       power = latest_results_pv_grid['results'][project_idx]['final_state']['power']
       energy = latest_results_pv_grid['results'][project_idx]['final_state']['capacity']
       
       lcos, total_cost, total_energy = calculate_corrected_lcos(result, power, energy)
       print(f"\nProject {result['project']} (Bus {result['bus']}):")
       print(f"Power Rating: {power:.2f} MW")
       print(f"Energy Capacity: {energy:.2f} MWh")
       print(f"LCOS: {lcos:.2f} €/MWh")
       print(f"Total Lifetime Cost: {total_cost/1e6:.2f}M €")
       print(f"Total Lifetime Energy: {total_energy:.2f} MWh")
    
    # Plot comparison
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
   for ax, results, title in [(ax1, pv_only_results, 'PV-only Scenario'), 
                             (ax2, pv_grid_results, 'PV+Grid Scenario')]:
        projects = [f"P{r['project']}\nBus {r['bus']}" for r in results]
        lcos_values = []
        
        for result in results:
            project_idx = result['project'] - 1
            power = (latest_results_pv_only if title=='PV-only Scenario' else latest_results_pv_grid)['results'][project_idx]['final_state']['power']
            energy = (latest_results_pv_only if title=='PV-only Scenario' else latest_results_pv_grid)['results'][project_idx]['final_state']['capacity']
            lcos = calculate_corrected_lcos(result, power, energy)[0]
            # Handle infinite values
            if np.isinf(lcos):
                lcos_values.append(0)  # or any other placeholder value
            else:
                lcos_values.append(lcos)
        
        bars = ax.bar(projects, lcos_values)
        ax.set_title(f'LCOS for Top 5 Projects - {title}')
        ax.set_xlabel('Project')
        ax.set_ylabel('LCOS (€/MWh)')
        ax.grid(True, alpha=0.3)
        
        # Add value labels, handling infinite values
        for i, (bar, lcos) in enumerate(zip(bars, lcos_values)):
            height = bar.get_height()
            if height == 0:  # This was an infinite value
                ax.text(bar.get_x() + bar.get_width()/2., 0,
                       'N/A', ha='center', va='bottom')
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}', ha='center', va='bottom')
    
   plt.tight_layout()
   plt.show()

print_and_plot_lcos_analysis(top_projects, latest_results_pv_only, latest_results_pv_grid)
####################################################################################################################
def plot_pv_bess_comparison_separate(pv_data, results_dict, top_projects, scenario_type):
    # Get projects from top_projects for consistent ordering
    projects = []
    for project in top_projects[scenario_type]:
        bus_number = project['bus_number']
        profit = project['annual_net_revenue']
        pv_size = project['pv_capacity_mw']
        bess_power = project['bess_power_mw']
        bess_energy = project['bess_energy_mwh']
        projects.append((bus_number, profit, pv_size, bess_power, bess_energy))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(5)
    width = 0.25
    
    # Create bars
    pv_bars = ax.bar(x - width, [p[2] for p in projects], width, 
                     label='PV Plant Size (MW)', color='#4CAF50')
    power_bars = ax.bar(x, [p[3] for p in projects], width,
                       label='BESS Power (MW)', color='#2196F3')
    energy_bars = ax.bar(x + width, [p[4] for p in projects], width,
                        label='BESS Energy (MWh)', color='#FFC107')
    
    # Customize plot
    ax.set_ylabel('Size (MW/MWh)')
    ax.set_title(f'PV Plant and BESS Size Comparison - Top 5 Projects ({scenario_type})')
    ax.set_xticks(x)
    ax.set_xticklabels([f'P{i+1}\nBus {p[0]}' for i, p in enumerate(projects)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom')
    
    add_labels(pv_bars)
    add_labels(power_bars)
    add_labels(energy_bars)
    
    plt.tight_layout()
    plt.show()

# Plot both scenarios using filtered top projects data
print("\nPV-only Scenario:")
plot_pv_bess_comparison_separate(pv_data, latest_results_pv_only, top_projects, 'pv_only')

print("\nPV+Grid Scenario:")
plot_pv_bess_comparison_separate(pv_data, latest_results_pv_grid, top_projects, 'pv_grid')
####################################################################################################################
def plot_price_comparison(hourly_data, reg_prices, top_projects, summer_start=4344, winter_start=336):  # July and January weeks
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    duration_hours = 168  # One week
    hours = range(duration_hours)
    
    # Summer week
    summer_end = summer_start + duration_hours
    market_prices_summer = hourly_data['Price_Euro_per_MWh'][summer_start:summer_end]
    reg_up_prices_summer = reg_prices['Reg_up'][summer_start:summer_end]
    reg_down_prices_summer = reg_prices['Reg_down'][summer_start:summer_end]
    
    # Plot market prices for summer
    ax1.plot(hours, market_prices_summer, 
            label='Day-Ahead Price', color='#2196F3', alpha=0.8, linewidth=1.5)
    ax1.plot(hours, reg_up_prices_summer, 
            label='Regulation Up', color='#4CAF50', alpha=0.8, linewidth=1.5)
    ax1.plot(hours, reg_down_prices_summer, 
            label='Regulation Down', color='#F44336', alpha=0.8, linewidth=1.5)
    
    # FiT prices with different line styles and colors for PV-only
    fit_styles = [
        {'color': '#FF6B6B', 'linestyle': '--', 'linewidth': 2},    # Red dashed
        {'color': '#4ECDC4', 'linestyle': ':', 'linewidth': 3},     # Teal dotted
        {'color': '#FFD93D', 'linestyle': '-.', 'linewidth': 2},    # Yellow dash-dot
        {'color': '#95A5A6', 'linestyle': (0, (3, 1, 1, 1)), 'linewidth': 2},  # Gray complex dash
        {'color': '#8E44AD', 'linestyle': (0, (5, 1)), 'linewidth': 2}         # Purple dashed
    ]
    
    for i, project in enumerate(top_projects['pv_only']):
        ax1.axhline(y=project['fit_price'], 
                   label=f'FiT Bus {project["bus_number"]}', 
                   **fit_styles[i])
    
    # Customize summer plot
    ax1.set_title('Market Price Comparison - PV-only Scenario (Summer Week)', 
                 fontsize=14, pad=20)
    ax1.set_xlabel('Hours', fontsize=12)
    ax1.set_ylabel('Price (€/MWh)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Set x-ticks for days
    ax1.set_xticks(np.arange(0, duration_hours, 24))
    ax1.set_xticklabels([f'Day {i+1}' for i in range(7)])
    
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Winter week
    winter_end = winter_start + duration_hours
    market_prices_winter = hourly_data['Price_Euro_per_MWh'][winter_start:winter_end]
    reg_up_prices_winter = reg_prices['Reg_up'][winter_start:winter_end]
    reg_down_prices_winter = reg_prices['Reg_down'][winter_start:winter_end]
    
    # Plot market prices for winter
    ax2.plot(hours, market_prices_winter, 
            label='Day-Ahead Price', color='#2196F3', alpha=0.8, linewidth=1.5)
    ax2.plot(hours, reg_up_prices_winter, 
            label='Regulation Up', color='#4CAF50', alpha=0.8, linewidth=1.5)
    ax2.plot(hours, reg_down_prices_winter, 
            label='Regulation Down', color='#F44336', alpha=0.8, linewidth=1.5)
    
    # FiT prices with different line styles for PV+Grid
    for i, project in enumerate(top_projects['pv_grid']):
        ax2.axhline(y=project['fit_price'], 
                   label=f'FiT Bus {project["bus_number"]}', 
                   **fit_styles[i])
    
    # Customize winter plot
    ax2.set_title('Market Price Comparison - PV+Grid Scenario (Winter Week)', 
                 fontsize=14, pad=20)
    ax2.set_xlabel('Hours', fontsize=12)
    ax2.set_ylabel('Price (€/MWh)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Set x-ticks for days
    ax2.set_xticks(np.arange(0, duration_hours, 24))
    ax2.set_xticklabels([f'Day {i+1}' for i in range(7)])
    
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

# Call the function
plot_price_comparison(hourly_data, reg_prices, top_projects)
####################################################################################################################
def print_revenue_breakdown(results_dict, scenario_type, top_projects):
    scenario_key = 'pv_only' if scenario_type == "PV-only" else 'pv_grid'
    
    if scenario_key == 'pv_only':
        total_revenues = {
            'pv_fit': 0,
            'regulation_up': 0,
            'annual_opex': 0,
            'charging_costs': 0,
            'net_profit': 0
        }
    else:
        total_revenues = {
            'day_ahead': 0,
            'regulation_up': 0,
            'regulation_down': 0,
            'annual_opex': 0,
            'charging_costs': 0,
            'net_profit': 0
        }
    
    print(f"\n{'='*80}")
    print(f"REVENUE BREAKDOWN FOR {scenario_type.upper()} SCENARIO")
    print(f"{'='*80}")
    
    print("\nVALIDATING PROJECT SELECTION:")
    print("-" * 50)
    
    for project in top_projects[scenario_key]:
        print(f"\nPROJECT {project['rank']} (Bus {project['bus_number']}):")
        print("-" * 40)
        
        # Calculate NPV-style costs
        energy = project['bess_energy_mwh']
        energy_cost = energy * MarketParams.BESS_COST_E * 1000
        annual_opex = energy_cost * 0.02
        project_idx = project['rank'] - 1
        ops = results_dict['results'][project_idx]['operations']
        # Use stored operations object
        #ops = project['operations']
        revenue = ops.get_revenue_breakdown()
        charging_costs = revenue['total_costs']
        
        # Configuration Summary
        print("Project Configuration:")
        print(f"  BESS Power Rating: {project['bess_power_mw']:.2f} MW")
        print(f"  BESS Energy Capacity: {project['bess_energy_mwh']:.2f} MWh")
        print(f"  P/E Ratio: {project['bess_power_mw']/project['bess_energy_mwh']:.2f}")
        
        # Revenue Breakdown
        print("\nRevenue Streams:")
        if scenario_key == 'pv_only':
            print(f"  PV FIT Revenue: {revenue['pv_fit']:,.2f} €")
            print(f"  Regulation Up Revenue: {revenue['regulation_up']:,.2f} €")
            total_revenue = revenue['pv_fit'] + revenue['regulation_up']
            print(f"  Total Revenue: {total_revenue:,.2f} €")
            total_revenues['pv_fit'] += revenue['pv_fit']
            total_revenues['regulation_up'] += revenue['regulation_up']
        else:
            print(f"  Day-Ahead Market Revenue: {revenue['day_ahead']:,.2f} €")
            print(f"  Regulation Up Revenue: {revenue['regulation_up']:,.2f} €")
            print(f"  Regulation Down Revenue: {revenue['regulation_down']:,.2f} €")
            total_revenue = revenue['day_ahead'] + revenue['regulation_up'] + revenue['regulation_down']
            print(f"  Total Revenue: {total_revenue:,.2f} €")
            total_revenues['day_ahead'] += revenue['day_ahead']
            total_revenues['regulation_up'] += revenue['regulation_up']
            total_revenues['regulation_down'] += revenue['regulation_down']
        
        # Cost Breakdown
        print("\nCost Breakdown:")
        print(f"  Annual OPEX (NPV): {annual_opex:,.2f} €")
        print(f"  Charging Costs: {charging_costs:,.2f} €")
        print(f"  Total Costs: {(annual_opex + charging_costs):,.2f} €")
        
        # Net Profit
        net_profit = total_revenue - annual_opex - charging_costs
        print(f"  Net Annual Profit: {net_profit:,.2f} €")
        
        # Regulation Market Statistics
        print("\nRegulation Market Statistics:")
        print(f"  Regulation Up Activations: {ops.reg_up_activations}")
        if scenario_key != 'pv_only':
            print(f"  Regulation Down Activations: {ops.reg_down_activations}")
        print(f"  Total Regulation Up Energy: {ops.reg_up_energy:.2f} MWh")
        if scenario_key != 'pv_only':
            print(f"  Total Regulation Down Energy: {ops.reg_down_energy:.2f} MWh")
        
        total_revenues['annual_opex'] += annual_opex
        total_revenues['charging_costs'] += charging_costs
        total_revenues['net_profit'] += net_profit
    
    # Print portfolio totals
    print(f"\n{'='*80}")
    print("TOTAL PORTFOLIO RESULTS:")
    print(f"{'='*80}")
    
    if scenario_key == 'pv_only':
        print(f"Total PV FIT Revenue: {total_revenues['pv_fit']:,.2f} €")
        print(f"Total Regulation Up Revenue: {total_revenues['regulation_up']:,.2f} €")
        total_rev = total_revenues['pv_fit'] + total_revenues['regulation_up']
        print(f"Total Revenue: {total_rev:,.2f} €")
    else:
        print(f"Total Day-Ahead Revenue: {total_revenues['day_ahead']:,.2f} €")
        print(f"Total Regulation Up Revenue: {total_revenues['regulation_up']:,.2f} €")
        print(f"Total Regulation Down Revenue: {total_revenues['regulation_down']:,.2f} €")
        total_rev = total_revenues['day_ahead'] + total_revenues['regulation_up'] + total_revenues['regulation_down']
        print(f"Total Revenue: {total_rev:,.2f} €")
    
    print(f"Total Annual OPEX: {total_revenues['annual_opex']:,.2f} €")
    print(f"Total Charging Costs: {total_revenues['charging_costs']:,.2f} €")
    print(f"Total Costs: {(total_revenues['annual_opex'] + total_revenues['charging_costs']):,.2f} €")
    print(f"Total Portfolio Net Profit: {total_revenues['net_profit']:,.2f} €")

# Generate new top projects summary
top_projects = create_top_projects_summary(latest_results_pv_only, latest_results_pv_grid, pv_data)

# Print revenue breakdown for both scenarios
print_revenue_breakdown(latest_results_pv_only, "PV-only", top_projects)
print_revenue_breakdown(latest_results_pv_grid, "PV+Grid", top_projects)
####################################################################################################################
def plot_profit_breakdowns(results_pv_only, results_pv_grid, top_projects):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Colors
    COLOR_FIT = '#4CAF50'      # Green for FIT/Day-Ahead
    COLOR_REG_UP = '#2196F3'   # Blue for Reg Up
    COLOR_REG_DOWN = '#FFC107' # Yellow for Reg Down
    COLOR_COSTS = '#F44336'    # Red for annual OPEX
    COLOR_CHARGING = '#d35400' # Orange for charging costs
    
    def plot_scenario(ax, results_dict, scenario_projects, include_reg_down=False):
        projects_data = []
        for project in scenario_projects:
            project_idx = project['rank'] - 1
            ops = results_dict['results'][project_idx]['operations']
            revenue = ops.get_revenue_breakdown()
            
            # Calculate NPV-style cost (annual OPEX)
            energy = project['bess_energy_mwh']
            energy_cost = energy * MarketParams.BESS_COST_E * 1000
            annual_opex = energy_cost * 0.02  # 2% of CAPEX
            charging_costs = revenue['total_costs']
             
            # Use original revenue calculations but NPV-style costs
            if include_reg_down:
                total_revenue = revenue['day_ahead'] + revenue['regulation_up'] + revenue['regulation_down']
                projects_data.append({
                    'bus': project['bus_number'],
                    'day_ahead': revenue['day_ahead'],
                    'reg_up': revenue['regulation_up'],
                    'reg_down': revenue['regulation_down'],
                    'total_revenue': total_revenue,
                    'costs': annual_opex,  
                    'charging_costs': charging_costs,
                    'net_profit': total_revenue - annual_opex - charging_costs
                })
            else:
                total_revenue = revenue['pv_fit'] + revenue['regulation_up']
                projects_data.append({
                    'bus': project['bus_number'],
                    'fit': revenue['pv_fit'],
                    'reg_up': revenue['regulation_up'],
                    'total_revenue': total_revenue,
                    'costs': annual_opex,  
                    'charging_costs': charging_costs,
                    'net_profit': total_revenue - annual_opex - charging_costs
                })

        x = np.arange(5)
        width = 0.5
        max_height = max(p['total_revenue'] for p in projects_data)

        for i, project in enumerate(projects_data):
            if include_reg_down:
                revenues = [project['day_ahead'], project['reg_up'], project['reg_down']]
                colors = [COLOR_FIT, COLOR_REG_UP, COLOR_REG_DOWN]
                labels = ['Day-Ahead Revenue', 'Regulation Up Revenue', 'Regulation Down Revenue']
            else:
                revenues = [project['fit'], project['reg_up']]
                colors = [COLOR_FIT, COLOR_REG_UP]
                labels = ['FIT Revenue', 'Regulation Up Revenue']
    
            bottom = 0
            for revenue, color, label in zip(revenues, colors, labels):
                ax.bar(i, revenue, width, bottom=bottom, color=color, 
                      label=label if i == 0 else "")
                bottom += revenue
    
            # Add costs as negative bars
            ax.bar(i, -project['costs'], width, color=COLOR_COSTS, 
                  label='Annual OPEX (NPV)' if i == 0 else "")
            ax.bar(i, -project['charging_costs'], width, bottom=-project['costs'], 
                  color=COLOR_CHARGING, label='Charging Costs' if i == 0 else "")
            
            # Text annotations
            total_costs = project['costs'] + project['charging_costs']
            ax.text(i, -total_costs - max_height*0.05, f'Bus {project["bus"]}', 
                   ha='center', va='top', fontsize=9)
            
            if project['total_revenue'] > 0:
                ax.text(i, project['total_revenue']/2, 
                       f'Total Revenue:\n€{project["total_revenue"]:,.0f}', 
                       ha='center', va='center', fontsize=8)
            
            # Annotations for both types of costs
            ax.text(i, -project['costs']/2, 
                   f'Annual OPEX:\n€{project["costs"]:,.0f}', 
                   ha='center', va='center', color='white', fontsize=8)
            
            if project['charging_costs'] > 0:
                ax.text(i, -project['costs'] - project['charging_costs']/2, 
                       f'Charging Costs:\n€{project["charging_costs"]:,.0f}', 
                       ha='center', va='center', color='white', fontsize=8)
            
            ax.text(i, project['total_revenue'] + max_height*0.02, 
                   f'Net Profit:\n€{project["net_profit"]:,.0f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
   
        ax.set_xticks(x)
        ax.set_xticklabels([f'P{i+1}' for i in range(5)])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x/1000), ',') + 'k'))
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Move legend outside the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add padding to accommodate the legend
        ax.set_ylim([-max_height*0.2, max_height*1.2])

    # Plot both scenarios
    plot_scenario(ax1, results_pv_only, top_projects['pv_only'])
    plot_scenario(ax2, results_pv_grid, top_projects['pv_grid'], True)
    
    # Adjust titles position
    ax1.set_title('Top 5 Projects - PV Only Scenario', 
                 fontsize=14, pad=20, y=1.02)
    ax2.set_title('Top 5 Projects - PV+Grid Scenario', 
                 fontsize=14, pad=20, y=1.02)
    
    for ax in [ax1, ax2]:
        ax.set_ylabel('Euros (€)')
    
    # Adjust layout to prevent overlapping
    plt.subplots_adjust(right=0.85, hspace=0.3)
    plt.show()

plot_profit_breakdowns(latest_results_pv_only, latest_results_pv_grid, top_projects)  
  
####################################################################################################################
from typing import Dict

def plot_total_profit_comparison_pie(results_pv_only: Dict, results_pv_grid: Dict, top_projects: Dict) -> None:

    # Set style
    plt.style.use('seaborn-v0_8-pastel')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Color scheme
    colors_pv = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
    colors_grid = ['#f1c40f', '#3498db', '#9b59b6', '#e74c3c']  # Yellow, Blue, Purple, Red
    
    # Calculate values for PV-only
    pv_only_values = {
        'PV FIT\nRevenue': 0,
        'Regulation Up\nRevenue': 0,
        'Total Costs': 0
    }
    for project in top_projects['pv_only']:
        project_idx = project['rank'] - 1
        revenue = results_pv_only['results'][project_idx]['operations'].get_revenue_breakdown()
        pv_only_values['PV FIT\nRevenue'] += revenue['pv_fit']
        pv_only_values['Regulation Up\nRevenue'] += revenue['regulation_up']
        pv_only_values['Total Costs'] += revenue['total_costs']
    
    # Calculate values for PV+Grid
    pv_grid_values = {
        'Day-Ahead\nRevenue': 0,
        'Regulation Up\nRevenue': 0,
        'Regulation Down\nRevenue': 0,
        'Total Costs': 0
    }
    for project in top_projects['pv_grid']:
        project_idx = project['rank'] - 1
        revenue = results_pv_grid['results'][project_idx]['operations'].get_revenue_breakdown()
        pv_grid_values['Day-Ahead\nRevenue'] += revenue['day_ahead']
        pv_grid_values['Regulation Up\nRevenue'] += revenue['regulation_up']
        pv_grid_values['Regulation Down\nRevenue'] += revenue['regulation_down']
        pv_grid_values['Total Costs'] += revenue['total_costs']
    
    # Function to create percentage labels
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(abs(val) for val in values)
            val = pct * total / 100.0
            return f'{pct:.1f}%\n(€{val/1e6:.1f}M)'
        return my_autopct
    
    # Plot PV-only pie chart
    wedges1, texts1, autotexts1 = ax1.pie(
        [abs(val) for val in pv_only_values.values()],
        labels=pv_only_values.keys(),
        colors=colors_pv,
        autopct=make_autopct(list(pv_only_values.values())),
        startangle=90,
        pctdistance=1.25,  # <-- Increased from around 0.85 to move text out of the pie
        explode=[0.05] * len(pv_only_values)
    )
    
    # Plot PV+Grid pie chart
    wedges2, texts2, autotexts2 = ax2.pie(
        [abs(val) for val in pv_grid_values.values()],
        labels=pv_grid_values.keys(),
        colors=colors_grid,
        autopct=make_autopct(list(pv_grid_values.values())),
        startangle=90,
        pctdistance=1.25,  # <-- Likewise, move text out for the second pie
        explode=[0.05] * len(pv_grid_values)
    )
    
    # Format text properties
    plt.setp(autotexts1 + autotexts2, size=10, weight="bold")
    plt.setp(texts1 + texts2, size=12)
    
    # Add titles
    total_profit_pv = sum(pv_only_values.values())
    total_profit_grid = sum(pv_grid_values.values())
    
    ax1.set_title(
        f'PV-Only Scenario\nTotal Net Profit: €{total_profit_pv/1e6:.1f}M\n'
        f'(Buses: {", ".join(str(p["bus_number"]) for p in top_projects["pv_only"])})',
        pad=20, size=14, weight='bold'
    )
    
    ax2.set_title(
        f'PV+Grid Scenario\nTotal Net Profit: €{total_profit_grid/1e6:.1f}M\n'
        f'(Buses: {", ".join(str(p["bus_number"]) for p in top_projects["pv_grid"])})',
        pad=20, size=14, weight='bold'
    )
    
    # Add a centered super title
    plt.suptitle(
        'Revenue and Cost Breakdown Comparison\nTop 5 Projects for Each Scenario',
        size=16, weight='bold', y=1.05
    )
    
    # Add a legend
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.show()

# Create the pie chart visualization
plot_total_profit_comparison_pie(latest_results_pv_only, latest_results_pv_grid, top_projects)

#%% Base Case
def calculate_base_case_revenue(pv_data, hourly_data):

    base_results = []
    
    # Constants
    PV_DEGRADATION = 0.005  # 0.5% annual degradation for PV
    PV_OM_COST_RATIO = 0.02  # 2% of annual revenue for O&M
    DISCOUNT_RATE = 0.02
    YEARS = 10  # Remaining lifetime
    
    for _, plant in pv_data.iterrows():
        bus_number = int(plant['bus'])
        fit_price = plant['FIT']
        grid_limit = plant['max_grid']
        
        # Get PV generation data
        pv_column = f'pv_gen_{bus_number}'
        if pv_column not in hourly_data.columns:
            print(f"Warning: No generation data found for bus {bus_number}")
            continue
            
        pv_generation = hourly_data[pv_column]
        
        # Calculate generation metrics
        total_generation = pv_generation.sum()
        usable_generation = np.minimum(pv_generation, grid_limit)
        total_usable = usable_generation.sum()
        total_curtailed = total_generation - total_usable
        
        # Calculate first year revenue
        annual_revenue = total_usable * fit_price * 0.68 
        annual_om_cost = annual_revenue * PV_OM_COST_RATIO*1.2
        
        # Calculate NPV
        npv = 0
        for year in range(YEARS):
            # Apply degradation to revenue
            year_degradation = (1 - PV_DEGRADATION) ** year
            year_revenue = annual_revenue * year_degradation
            year_om_cost = annual_om_cost * year_degradation
            year_cashflow = year_revenue - year_om_cost
            
            # Discount cashflow
            npv += year_cashflow / (1 + DISCOUNT_RATE) ** year
        
        # Calculate capacity factor
        hours_per_year = len(hourly_data)
        capacity_factor = (total_generation / (plant['P_MWp'] * hours_per_year)) * 100
        
        # Store results
        base_results.append({
            'bus_number': bus_number,
            'pv_capacity_mw': plant['P_MWp'],
            'grid_connection_mw': grid_limit,
            'fit_price': fit_price,
            'total_generation_mwh': total_generation,
            'usable_generation_mwh': total_usable,
            'curtailed_generation_mwh': total_curtailed,
            'curtailment_percentage': (total_curtailed/total_generation*100 if total_generation>0 else 0),
            'capacity_factor': capacity_factor,
            'annual_revenue': annual_revenue,
            'annual_om_cost': annual_om_cost,
            'net_annual_profit': annual_revenue - annual_om_cost,
            'npv': npv
        })
    
    return base_results
####################################################################################################################
def create_comprehensive_comparison(base_results, top_projects, latest_results_pv_only, latest_results_pv_grid):
    comparison_results = {
        'pv_only': [],
        'pv_grid': []
    }
    
    # Create lookup for base case results
    base_by_bus = {r['bus_number']: r for r in base_results}
    
    # Process PV-only scenario
    for project in top_projects['pv_only']:
        bus = project['bus_number']
        project_idx = project['rank'] - 1
        ops = latest_results_pv_only['results'][project_idx]['operations']
        base = base_by_bus[bus]
        
        # Get economic results for BESS case
        economics = analyze_project_economics_fixed(
            project['bess_power_mw'],
            project['bess_energy_mwh'],
            ops
        )
        
        # Calculate improvements
        profit_increase = economics['annual_net_revenue'] - base['net_annual_profit']
        profit_increase_pct = (profit_increase / abs(base['net_annual_profit'])) * 100
        npv_increase = economics['npv'] - base['npv']
        npv_increase_pct = (npv_increase / abs(base['npv'])) * 100
        
        comparison_results['pv_only'].append({
            'bus': bus,
            'base_case': {
                'pv_capacity': base['pv_capacity_mw'],
                'annual_revenue': base['annual_revenue'],
                'annual_profit': base['net_annual_profit'],
                'npv': base['npv'],
                'curtailment': base['curtailed_generation_mwh'],
                'curtailment_pct': base['curtailment_percentage']
            },
            'bess_case': {
                'bess_power': project['bess_power_mw'],
                'bess_energy': project['bess_energy_mwh'],
                'capex': economics['capex'],
                'annual_revenue': economics['annual_revenue'],
                'annual_profit': economics['annual_net_revenue'],
                'npv': economics['npv'],
                'payback': economics['payback_period'],
                'roi': economics['roi']
            },
            'improvements': {
                'profit_increase': profit_increase,
                'profit_increase_pct': profit_increase_pct,
                'npv_increase': npv_increase,
                'npv_increase_pct': npv_increase_pct
            }
        })
    
    # Process PV+Grid scenario
    for project in top_projects['pv_grid']:
        bus = project['bus_number']
        project_idx = project['rank'] - 1
        ops = latest_results_pv_grid['results'][project_idx]['operations']
        base = base_by_bus[bus]
        
        # Get economic results for BESS case
        economics = analyze_project_economics_fixed(
            project['bess_power_mw'],
            project['bess_energy_mwh'],
            ops
        )
        
        # Calculate improvements
        profit_increase = economics['annual_net_revenue'] - base['net_annual_profit']
        profit_increase_pct = (profit_increase / abs(base['net_annual_profit'])) * 100
        npv_increase = economics['npv'] - base['npv']
        npv_increase_pct = (npv_increase / abs(base['npv'])) * 100
        
        comparison_results['pv_grid'].append({
            'bus': bus,
            'base_case': {
                'pv_capacity': base['pv_capacity_mw'],
                'annual_revenue': base['annual_revenue'],
                'annual_profit': base['net_annual_profit'],
                'npv': base['npv'],
                'curtailment': base['curtailed_generation_mwh'],
                'curtailment_pct': base['curtailment_percentage']
            },
            'bess_case': {
                'bess_power': project['bess_power_mw'],
                'bess_energy': project['bess_energy_mwh'],
                'capex': economics['capex'],
                'annual_revenue': economics['annual_revenue'],
                'annual_profit': economics['annual_net_revenue'],
                'npv': economics['npv'],
                'payback': economics['payback_period'],
                'roi': economics['roi']
            },
            'improvements': {
                'profit_increase': profit_increase,
                'profit_increase_pct': profit_increase_pct,
                'npv_increase': npv_increase,
                'npv_increase_pct': npv_increase_pct
            }
        })
    
    return comparison_results
####################################################################################################################
def print_comprehensive_results(comparison_results):
    """
    Print detailed comparison results
    """
    for scenario in ['pv_only', 'pv_grid']:
        print(f"\n{scenario.upper()} Scenario Analysis")
        print("=" * 80)
        
        for project in comparison_results[scenario]:
            print(f"\nBus {project['bus']}:")
            print("-" * 40)
            
            print("Base Case (Existing PV):")
            print(f"  PV Capacity: {project['base_case']['pv_capacity']:.2f} MW")
            print(f"  Annual Revenue: €{project['base_case']['annual_revenue']:,.2f}")
            print(f"  Annual Profit: €{project['base_case']['annual_profit']:,.2f}")
            print(f"  NPV: €{project['base_case']['npv']:,.2f}")
            print(f"  Curtailment: {project['base_case']['curtailment']:.2f} MWh ({project['base_case']['curtailment_pct']:.1f}%)")
            
            print("\nWith BESS:")
            print(f"  BESS Size: {project['bess_case']['bess_power']:.2f} MW / {project['bess_case']['bess_energy']:.2f} MWh")
            print(f"  CAPEX: €{project['bess_case']['capex']:,.2f}")
            print(f"  Annual Revenue: €{project['bess_case']['annual_revenue']:,.2f}")
            print(f"  Annual Profit: €{project['bess_case']['annual_profit']:,.2f}")
            print(f"  NPV: €{project['bess_case']['npv']:,.2f}")
            print(f"  Payback Period: {project['bess_case']['payback']:.1f} years")
            print(f"  ROI: {project['bess_case']['roi']:.1f}%")
            
            print("\nImprovements:")
            print(f"  Profit Increase: €{project['improvements']['profit_increase']:,.2f} ({project['improvements']['profit_increase_pct']:+.1f}%)")
            print(f"  NPV Increase: €{project['improvements']['npv_increase']:,.2f} ({project['improvements']['npv_increase_pct']:+.1f}%)")
            
            # Add recommendation
            viable = (project['bess_case']['payback'] < 10 and 
                     project['improvements']['npv_increase'] > 0)
            print("\nRecommendation:", 
                  "✓ BESS retrofit is economically viable" if viable else 
                  "✗ BESS retrofit may not be economically attractive")
####################################################################################################################
def plot_comparison_metrics(comparison_results):
    """
    Create visualizations for the comparison
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    for scenario in ['pv_only', 'pv_grid']:
        # Get data
        buses = [p['bus'] for p in comparison_results[scenario]]
        npv_base = [p['base_case']['npv']/1e6 for p in comparison_results[scenario]]
        npv_bess = [p['bess_case']['npv']/1e6 for p in comparison_results[scenario]]
        profit_base = [p['base_case']['annual_profit']/1e3 for p in comparison_results[scenario]]
        profit_bess = [p['bess_case']['annual_profit']/1e3 for p in comparison_results[scenario]]
        
        # Plot NPV comparison
        ax = ax1 if scenario == 'pv_only' else ax2
        x = np.arange(len(buses))
        width = 0.35
        
        ax.bar(x - width/2, npv_base, width, label='Base Case', color='#2ecc71', alpha=0.7)
        ax.bar(x + width/2, npv_bess, width, label='With BESS', color='#3498db', alpha=0.7)
        ax.set_title(f'{scenario.upper()}: NPV Comparison', fontsize=12, pad=20)
        ax.set_xlabel('Bus Number')
        ax.set_ylabel('NPV (M€)')
        ax.set_xticks(x)
        ax.set_xticklabels(buses)
        
        # Add improvement percentages
        for i in range(len(buses)):
            pct = comparison_results[scenario][i]['improvements']['npv_increase_pct']
            ax.text(i, max(npv_base[i], npv_bess[i]), 
                   f'{pct:+.1f}%', ha='center', va='bottom')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot Annual Profit comparison
        ax = ax3 if scenario == 'pv_only' else ax4
        
        ax.bar(x - width/2, profit_base, width, label='Base Case', color='#2ecc71', alpha=0.7)
        ax.bar(x + width/2, profit_bess, width, label='With BESS', color='#3498db', alpha=0.7)
        ax.set_title(f'{scenario.upper()}: Annual Profit Comparison', fontsize=12, pad=20)
        ax.set_xlabel('Bus Number')
        ax.set_ylabel('Annual Profit (k€)')
        ax.set_xticks(x)
        ax.set_xticklabels(buses)
        
        # Add improvement percentages
        for i in range(len(buses)):
            pct = comparison_results[scenario][i]['improvements']['profit_increase_pct']
            ax.text(i, max(profit_base[i], profit_bess[i]), 
                   f'{pct:+.1f}%', ha='center', va='bottom')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Comparison: Base Case vs BESS Retrofit\n(10-year remaining lifetime)', 
                fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

# Run the complete analysis
base_results = calculate_base_case_revenue(pv_data, hourly_data)
comparison_results = create_comprehensive_comparison(base_results, top_projects, 
                                                  latest_results_pv_only, latest_results_pv_grid)
print_comprehensive_results(comparison_results)
plot_comparison_metrics(comparison_results)
#%% Final Sensitivity
def calculate_base_case_npv(base_results):
    """Calculate NPV for base case (PV without BESS)"""
    base_npvs = {}
    for result in base_results:
        base_npvs[result['bus_number']] = result['npv']
    return base_npvs
####################################################################################################################
def calculate_scenario_npv(scenario_type, project, ops, params):
    # Non-linear CAPEX impact
    capex_variation = params.get('capex_mult', 0)
    capex_impact = np.sign(capex_variation) * (1 + abs(capex_variation))**1.3
    modified_capex = project['bess_energy_mwh'] * MarketParams.BESS_COST_E * 1000 * (1 + capex_impact)
    
    # Market price with volatility impact
    market_mult = np.sign(params.get('market_mult', 0)) * ((abs(params.get('market_mult', 0)))**1.2 + 1)
    
    # Revenue calculation
    revenue_breakdown = ops.get_revenue_breakdown()
    if scenario_type == 'pv_only':
        annual_revenue = (revenue_breakdown['pv_fit'] + revenue_breakdown['regulation_up']) * market_mult
    else:
        annual_revenue = sum([revenue_breakdown[k] for k in ['day_ahead', 'regulation_up', 'regulation_down']]) * market_mult
    
    # O&M with exponential growth
    #om_impact = params.get('om_mult', 0.02) * (1 + 0.05)**MarketParams.LIFETIME_YEARS
    #annual_opex = modified_capex * om_impact
    om_mult = params.get('om_mult', 0.02)  # Base OPEX multiplier
    om_variation = params.get('opex_variation', 0)  # OPEX variation (e.g., -15 for -15%, +15 for +15%)
    om_impact = om_mult * (1 + om_variation/100)  # Adjust OPEX multiplier based on variation
    annual_opex = modified_capex * om_impact
    
    # Efficiency with diminishing returns
    eff_impact = np.tanh(params.get('efficiency_mult', 0))
    annual_charging = revenue_breakdown['total_costs'] / (1 + eff_impact)
    
    return calculate_npv(
        capex=modified_capex,
        annual_revenue=annual_revenue,
        annual_opex=annual_opex,
        annual_charging_cost=annual_charging,
        discount_rate=params.get('discount_rate', 0.02),
        degradation_rate=params.get('degradation', 0.02)
    )
####################################################################################################################
def perform_sensitivity_analysis(base_results, top_projects, latest_results_pv_only, latest_results_pv_grid):
    results = {'pv_only': {}, 'pv_grid': {}}
    
    sensitivity_params = {
   'CAPEX': {'variations': [-30, -10, 0, 10, 20, 30]},  # Investment cost uncertainty
   'OPEX': {'variations': [-15, -7.5, 0, 7.5, 10, 15]},  # Operating cost variation
   'MARKET_PRICES': {'variations': [-45, -15, 0, 15, 30, 45]},  # Market price volatility
   'FIT_PRICES': {'variations': [-20, -7.5, 0, 7.5, 15, 20]},  # Feed-in tariff changes
   'DISCOUNT_RATE': {'variations': [-4, -2, 0, 2, 3, 4]},  # In percentage points
   'EFFICIENCY': {'variations': [-5, -2.5, 0, 2.5, 5]},  # Technical efficiency
   'PV_GENERATION': {'variations': [-20, -10, 0, 7.5, 15, 20]}  # Resource uncertainty
}
    
    for scenario in ['pv_only', 'pv_grid']:
        results[scenario] = {}
        scenario_results = latest_results_pv_only if scenario == 'pv_only' else latest_results_pv_grid
        projects = top_projects[scenario]
        
        for param, config in sensitivity_params.items():
            results[scenario][param] = []
            
            for project in projects:
                project_variations = []
                project_idx = project['rank'] - 1
                ops = scenario_results['results'][project_idx]['operations']
                
                # Get base NPV for this project
                base_economics = analyze_project_economics_fixed(
                    project['bess_power_mw'],
                    project['bess_energy_mwh'],
                    ops
                )
                base_npv = base_economics['npv']
                
                for variation in config['variations']:
                    # Calculate modified NPV
                    if param == 'CAPEX':
                        modified_capex = project['bess_energy_mwh'] * MarketParams.BESS_COST_E * 1000 * (1 + variation/100)
                        modified_npv = calculate_npv(
                            capex=modified_capex,
                            annual_revenue=base_economics['annual_revenue'],
                            annual_opex=base_economics['annual_opex'],
                            annual_charging_cost=0
                        )
                    elif param == 'OPEX':
                        # Adjust OPEX based on variation
                        om_mult = base_economics['annual_opex'] / (project['bess_energy_mwh'] * MarketParams.BESS_COST_E * 1000)
                        om_impact = om_mult * (1 + variation/100)
                        annual_opex = project['bess_energy_mwh'] * MarketParams.BESS_COST_E * 1000 * om_impact
                        modified_npv = calculate_npv(
                            capex=base_economics['capex'],
                            annual_revenue=base_economics['annual_revenue'],
                            annual_opex=annual_opex,
                            annual_charging_cost=0
                        )
                    
                    elif param == 'EFFICIENCY':
                        # Non-linear impact for efficiency
                        eff_impact = np.tanh(variation/100)
                        modified_npv = base_npv * (1 + eff_impact)
                        
                    elif param == 'DISCOUNT_RATE':
                        modified_npv = calculate_npv(
                            capex=base_economics['capex'],
                            annual_revenue=base_economics['annual_revenue'],
                            annual_opex=base_economics['annual_opex'],
                            annual_charging_cost=0,
                            discount_rate=0.02 * (1 + variation/100)
                        )
                    else:  # Other parameters
                        modified_npv = base_npv * (1 + variation/100)
                    
                    npv_change = ((modified_npv - base_npv) / abs(base_npv)) * 100
                    project_variations.append({
                        'variation': variation,
                        'npv_change': npv_change
                    })
                
                results[scenario][param].append(project_variations)
    
    return results
####################################################################################################################
def plot_sensitivity_with_ranges(sensitivity_results):
    import numpy as np
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    for ax, scenario in zip([ax1, ax2], ['pv_only', 'pv_grid']):
        param_results = []
        
        # Process data using the logic of the second function (averaging across all projects)
        for param, project_variations in sensitivity_results[scenario].items():
            # Collect all variations and NPV changes across all projects
            all_min_var = []
            all_max_var = []
            all_min_changes = []
            all_max_changes = []
            
            for project in project_variations:
                # Find min and max parameter variations for this project
                min_var = min(v['variation'] for v in project)
                max_var = max(v['variation'] for v in project)
                
                # Find NPV changes corresponding to min and max variations for this project
                min_changes = [v['npv_change'] for v in project if v['variation'] == min_var]
                max_changes = [v['npv_change'] for v in project if v['variation'] == max_var]
                
                # Append to the lists
                all_min_var.append(min_var)
                all_max_var.append(max_var)
                all_min_changes.extend(min_changes)
                all_max_changes.extend(max_changes)
            
            # Calculate average NPV changes and parameter variations across all projects
            min_change = np.mean(all_min_changes) if all_min_changes else None
            max_change = np.mean(all_max_changes) if all_max_changes else None
            min_var = np.mean(all_min_var) if all_min_var else None
            max_var = np.mean(all_max_var) if all_max_var else None
            
            param_results.append({
                'parameter': param,
                'min_change': min_change,  # Average NPV change for min variation
                'max_change': max_change,  # Average NPV change for max variation
                'min_var': min_var,       # Average min parameter variation
                'max_var': max_var         # Average max parameter variation
            })
        
        # Sort parameters by the magnitude of the NPV change range
        param_results.sort(key=lambda x: abs(x['max_change'] - x['min_change']), reverse=True)
        
        # Determine axis limits based on all NPV changes
        all_changes = []
        for r in param_results:
            if r['min_change'] is not None:
                all_changes.append(r['min_change'])
            if r['max_change'] is not None:
                all_changes.append(r['max_change'])
        max_abs_change = max(abs(x) for x in all_changes) if all_changes else 1
        axis_limit = max_abs_change * 1.2  # Add 20% buffer
        
        # Plotting style of the first function (color logic)
        y_pos = np.arange(len(param_results))
        bar_height = 0.45
        
        for i, result in enumerate(param_results):
            # Plot the bar for min_change (left side, negative NPV change)

            if result['min_change'] is not None:
                color = 'green' if result['min_var'] > 0 else 'red'
                ax.barh(y_pos[i], result['min_change'], left=0, height=bar_height,
                        color=color, alpha=0.4)
                # Position text at the end of the bar
                if result['min_change'] < 0:
                    # For negative values, move text a bit to the left
                    x_text = result['min_change'] - (axis_limit * 0.02)
                    ha = 'right'
                else:
                    x_text = result['min_change'] + (axis_limit * 0.02)
                    ha = 'left'
                ax.text(x_text, y_pos[i] - 0.15,
                        f"{result['min_change']:.1f}%", 
                        ha=ha, va='center', fontsize=11, color='dimgray')
            
            # Plot the bar for max_change (right side, positive NPV change)
            if result['max_change'] is not None:
                color = 'red' if result['max_var'] < 0 else 'green'
                ax.barh(y_pos[i], result['max_change'], left=0, height=bar_height,
                        color=color, alpha=0.4)
                if result['max_change'] >= 0:
                    x_text = result['max_change'] + (axis_limit * 0.02)
                    ha = 'left'
                else:
                    x_text = result['max_change'] - (axis_limit * 0.02)
                    ha = 'right'
                ax.text(x_text, y_pos[i] - 0.15,
                        f"{result['max_change']:.1f}%", 
                        ha=ha, va='center', fontsize=11, color='dimgray')
        
        # Final formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels([r['parameter'] for r in param_results])
        ax.grid(True, alpha=0.2)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=1)
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_title(f"{scenario.upper()}: NPV Sensitivity Analysis (Averaged Across Projects)")
        ax.set_xlabel("Change in NPV (%)")
    
    plt.tight_layout()
    plt.show()
####################################################################################################################
def print_sensitivity_results(sensitivity_results):
   for scenario in ['pv_only', 'pv_grid']:
       print(f"\nScenario: {scenario.upper()}")
       print("-" * 50)
       
       for param, project_variations in sensitivity_results[scenario].items():
           print(f"\nParameter: {param}")
           
           for i, project in enumerate(project_variations, 1):
               print(f"  Project #{i}:")
               for var in project:
                   print(f"    Variation: {var['variation']:+.1f}% | NPV Change: {var['npv_change']:+.2f}%")
# Run analysis
# First calculate base case NPVs if not already done
base_results = calculate_base_case_revenue(pv_data, hourly_data)

# Then run sensitivity analysis
sensitivity_results = perform_sensitivity_analysis(base_results, top_projects, 
                                                latest_results_pv_only, latest_results_pv_grid)

# Create visualization
plot_sensitivity_with_ranges(sensitivity_results)
print_sensitivity_results(sensitivity_results)






