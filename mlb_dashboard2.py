
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import json
import requests
from io import StringIO
import traceback

# Install required packages if needed
# pip install streamlit pandas numpy matplotlib plotly pybaseball

# Import pybaseball with only the working functions
try:
    from pybaseball import (
        batting_stats, pitching_stats, 
        playerid_lookup, statcast_batter, statcast_pitcher,
        standings, team_batting, team_pitching
    )
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False
    st.error("Pybaseball is not installed. Some features will be limited.")
    st.info("Install with: pip install pybaseball")

# Set page configuration
st.set_page_config(
    page_title="MLB Current Season Analytics Dashboard",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("⚾ MLB Current Season Analytics Dashboard")
st.markdown("### Analyze MLB performance with advanced metrics and hot hand analysis")

# Cache directory for saving data locally
CACHE_DIR = "mlb_dashboard_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Current year and month
CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month

# Define teams with colors
TEAMS = {
    "ARI": {"name": "Arizona Diamondbacks", "color": "#A71930"},
    "ATL": {"name": "Atlanta Braves", "color": "#CE1141"},
    "BAL": {"name": "Baltimore Orioles", "color": "#DF4601"},
    "BOS": {"name": "Boston Red Sox", "color": "#BD3039"},
    "CHC": {"name": "Chicago Cubs", "color": "#0E3386"},
    "CHW": {"name": "Chicago White Sox", "color": "#27251F"},
    "CIN": {"name": "Cincinnati Reds", "color": "#C6011F"},
    "CLE": {"name": "Cleveland Guardians", "color": "#00385D"},
    "COL": {"name": "Colorado Rockies", "color": "#333366"},
    "DET": {"name": "Detroit Tigers", "color": "#0C2340"},
    "HOU": {"name": "Houston Astros", "color": "#EB6E1F"},
    "KCR": {"name": "Kansas City Royals", "color": "#004687"},
    "LAA": {"name": "Los Angeles Angels", "color": "#BA0021"},
    "LAD": {"name": "Los Angeles Dodgers", "color": "#005A9C"},
    "MIA": {"name": "Miami Marlins", "color": "#00A3E0"},
    "MIL": {"name": "Milwaukee Brewers", "color": "#0A2351"},
    "MIN": {"name": "Minnesota Twins", "color": "#002B5C"},
    "NYM": {"name": "New York Mets", "color": "#FF5910"},
    "NYY": {"name": "New York Yankees", "color": "#0C2340"},
    "OAK": {"name": "Oakland Athletics", "color": "#003831"},
    "PHI": {"name": "Philadelphia Phillies", "color": "#E81828"},
    "PIT": {"name": "Pittsburgh Pirates", "color": "#27251F"},
    "SDP": {"name": "San Diego Padres", "color": "#2F241D"},
    "SFG": {"name": "San Francisco Giants", "color": "#FD5A1E"},
    "SEA": {"name": "Seattle Mariners", "color": "#0C2C56"},
    "STL": {"name": "St. Louis Cardinals", "color": "#C41E3A"},
    "TBR": {"name": "Tampa Bay Rays", "color": "#092C5C"},
    "TEX": {"name": "Texas Rangers", "color": "#003278"},
    "TOR": {"name": "Toronto Blue Jays", "color": "#134A8E"},
    "WSN": {"name": "Washington Nationals", "color": "#AB0003"}
}

# Sidebar for filters
st.sidebar.header("Filters")

# Season selection with current year as default
seasons = list(range(CURRENT_YEAR, CURRENT_YEAR-5, -1))
selected_season = st.sidebar.selectbox("Season", seasons, index=0)

# Team selection
team_options = ["All Teams"] + list(TEAMS.keys())
selected_team = st.sidebar.selectbox("Team", team_options, index=0)

# Player type
player_type = st.sidebar.radio("Stats Type", ["Batting", "Pitching"])

# Hot hand parameters
st.sidebar.header("Hot Hand Parameters")
rolling_window = st.sidebar.slider("Rolling Window (PA/BF)", 10, 100, 25)
hot_threshold = st.sidebar.slider("Hot Threshold", 0.05, 0.30, 0.10)
cold_threshold = st.sidebar.slider("Cold Threshold", -0.30, -0.05, -0.10)

# Advanced settings
st.sidebar.header("Advanced Settings")
cache_duration = st.sidebar.slider("Cache Duration (hours)", 1, 48, 24)
fallback_to_previous_season = st.sidebar.checkbox("Fallback to Previous Season if Current Unavailable", value=True)
show_debug_info = st.sidebar.checkbox("Show Debug Information", value=False)

# Functions for caching data
def get_cache_path(data_type, season, team="ALL"):
    """Get path for cached data file"""
    filename = f"{data_type}_{season}_{team}.csv"
    return os.path.join(CACHE_DIR, filename)

def get_cache_metadata_path(data_type, season, team="ALL"):
    """Get path for cached metadata file"""
    filename = f"{data_type}_{season}_{team}_metadata.json"
    return os.path.join(CACHE_DIR, filename)

def save_to_cache(data, data_type, season, team="ALL"):
    """Save data to cache"""
    if data is None or (isinstance(data, pd.DataFrame) and data.empty):
        return False
    
    try:
        # Save data
        cache_path = get_cache_path(data_type, season, team)
        data.to_csv(cache_path, index=False)
        
        # Save metadata
        cache_metadata_path = get_cache_metadata_path(data_type, season, team)
        metadata = {
            "timestamp": datetime.now().timestamp(),
            "rows": len(data),
            "columns": list(data.columns) if isinstance(data, pd.DataFrame) else []
        }
        with open(cache_metadata_path, "w") as f:
            json.dump(metadata, f)
        
        return True
    except Exception as e:
        if show_debug_info:
            st.error(f"Error saving to cache: {e}")
        return False

def load_from_cache(data_type, season, team="ALL", max_age_hours=24):
    """Load data from cache if it exists and is not too old"""
    try:
        cache_metadata_path = get_cache_metadata_path(data_type, season, team)
        if not os.path.exists(cache_metadata_path):
            return None
        
        # Check cache age
        with open(cache_metadata_path, "r") as f:
            metadata = json.load(f)
        
        cache_time = metadata.get("timestamp", 0)
        cache_age = (datetime.now().timestamp() - cache_time) / 3600  # in hours
        
        if cache_age > max_age_hours:
            if show_debug_info:
                st.info(f"Cache for {data_type} {season} {team} is too old ({cache_age:.1f} hours)")
            return None
        
        # Load data
        cache_path = get_cache_path(data_type, season, team)
        if os.path.exists(cache_path):
            data = pd.read_csv(cache_path)
            if show_debug_info:
                st.info(f"Loaded {data_type} {season} {team} from cache ({len(data)} rows)")
            return data
    except Exception as e:
        if show_debug_info:
            st.error(f"Error loading from cache: {e}")
    
    return None

# Functions to fetch data
@st.cache_data(ttl=3600)  # Cache for 1 hour in Streamlit (separate from file cache)
def fetch_batting_data(season, team="ALL"):
    """Fetch batting data for analysis"""
    # First check our file cache
    cached_data = load_from_cache("batting", season, team, max_age_hours=cache_duration)
    if cached_data is not None:
        return cached_data
    
    try:
        if not PYBASEBALL_AVAILABLE:
            raise ImportError("Pybaseball is not available")
            
        # Get batting stats for the season
        st.info(f"Fetching batting stats for {season} season... this may take a moment.")
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)
        
        # Get the data
        df = batting_stats(season)
        
        # Filter by team if specified
        if team != "All Teams":
            df = df[df['Team'] == team]
        
        # Filter to include only players with meaningful sample sizes
        df = df[df['PA'] >= 30]
        
        # Sort by key metrics
        if 'wOBA' in df.columns:
            df = df.sort_values('wOBA', ascending=False)
        else:
            df = df.sort_values('OPS', ascending=False)
        
        # Save to cache
        save_to_cache(df, "batting", season, team)
            
        return df
    except Exception as e:
        st.error(f"Error fetching batting data: {str(e)}")
        
        if show_debug_info:
            st.error(traceback.format_exc())
            
        if fallback_to_previous_season and season == CURRENT_YEAR:
            st.warning(f"Trying to fetch data from previous season ({CURRENT_YEAR-1})...")
            try:
                # Try with previous season
                previous_data = fetch_batting_data(CURRENT_YEAR-1, team)
                
                # Mark as previous season data
                if previous_data is not None and not previous_data.empty:
                    if 'Season' not in previous_data.columns:
                        previous_data['Season'] = CURRENT_YEAR-1
                    st.warning(f"Using data from {CURRENT_YEAR-1} season as a fallback")
                    return previous_data
            except Exception as prev_e:
                if show_debug_info:
                    st.error(f"Error fetching previous season data: {str(prev_e)}")
        
        # Provide fallback data
        st.warning("Using synthetic data as fallback due to API error.")
        
        # Use team-specific players for fallback data if available
        team_players = {
            "BAL": ["Adley Rutschman", "Gunnar Henderson", "Anthony Santander", "Ryan Mountcastle", 
                   "Cedric Mullins", "Austin Hays", "Ryan O'Hearn", "Jorge Mateo", "Jordan Westburg"],
            "NYY": ["Aaron Judge", "Juan Soto", "Anthony Rizzo", "Giancarlo Stanton", "Anthony Volpe", 
                   "DJ LeMahieu", "Gleyber Torres", "Alex Verdugo", "Austin Wells"],
            "BOS": ["Rafael Devers", "Triston Casas", "Jarren Duran", "Masataka Yoshida", 
                   "Trevor Story", "Ceddanne Rafaela", "Wilyer Abreu", "Connor Wong", "Enmanuel Valdez"],
            "LAD": ["Shohei Ohtani", "Mookie Betts", "Freddie Freeman", "Will Smith", "Max Muncy",
                   "Teoscar Hernández", "Gavin Lux", "Tommy Edman", "Jason Heyward"]
        }
        
        # Choose players for the selected team or use generic names
        if team in team_players:
            players = team_players[team]
        else:
            players = [f"Player {i}" for i in range(1, 20)]
        
        # Provide realistic fallback data
        fallback_data = pd.DataFrame({
            'playerid': range(1, len(players) + 1),
            'Name': players,
            'Team': [team] * len(players) if team != "All Teams" else np.random.choice(list(TEAMS.keys()), len(players)),
            'Season': [season] * len(players),
            'PA': np.random.randint(100, 600, len(players)),
            'AVG': np.random.uniform(0.220, 0.330, len(players)),
            'OBP': np.random.uniform(0.300, 0.420, len(players)),
            'SLG': np.random.uniform(0.380, 0.600, len(players)),
            'wOBA': np.random.uniform(0.310, 0.430, len(players)),
            'wRC+': np.random.randint(80, 160, len(players)),
            'HR': np.random.randint(5, 35, len(players)),
            'RBI': np.random.randint(20, 90, len(players)),
            'R': np.random.randint(20, 80, len(players)),
            'SB': np.random.randint(0, 25, len(players)),
            'BB%': np.random.uniform(0.05, 0.18, len(players)),
            'K%': np.random.uniform(0.12, 0.30, len(players)),
            'ISO': np.random.uniform(0.120, 0.300, len(players)),
            'BABIP': np.random.uniform(0.270, 0.350, len(players))
        })
        
        return fallback_data

@st.cache_data(ttl=3600)  # Cache for 1 hour in Streamlit
def fetch_pitching_data(season, team="ALL"):
    """Fetch pitching data for analysis"""
    # First check our file cache
    cached_data = load_from_cache("pitching", season, team, max_age_hours=cache_duration)
    if cached_data is not None:
        return cached_data
    
    try:
        if not PYBASEBALL_AVAILABLE:
            raise ImportError("Pybaseball is not available")
            
        # Get pitching stats for the season
        st.info(f"Fetching pitching stats for {season} season... this may take a moment.")
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)
        
        # Get the data
        df = pitching_stats(season)
        
        # Filter by team if specified
        if team != "All Teams":
            df = df[df['Team'] == team]
        
        # Filter to include only pitchers with meaningful sample sizes
        df = df[df['IP'] >= 10]
        
        # Sort by key metrics
        if 'ERA' in df.columns:
            df = df.sort_values('ERA', ascending=True)
        
        # Save to cache
        save_to_cache(df, "pitching", season, team)
            
        return df
    except Exception as e:
        st.error(f"Error fetching pitching data: {str(e)}")
        
        if show_debug_info:
            st.error(traceback.format_exc())
            
        if fallback_to_previous_season and season == CURRENT_YEAR:
            st.warning(f"Trying to fetch data from previous season ({CURRENT_YEAR-1})...")
            try:
                # Try with previous season
                previous_data = fetch_pitching_data(CURRENT_YEAR-1, team)
                
                # Mark as previous season data
                if previous_data is not None and not previous_data.empty:
                    if 'Season' not in previous_data.columns:
                        previous_data['Season'] = CURRENT_YEAR-1
                    st.warning(f"Using data from {CURRENT_YEAR-1} season as a fallback")
                    return previous_data
            except Exception as prev_e:
                if show_debug_info:
                    st.error(f"Error fetching previous season data: {str(prev_e)}")
        
        # Provide fallback data
        st.warning("Using synthetic pitching data as fallback due to API error.")
        
        # Use team-specific pitchers for fallback data if available
        team_pitchers = {
            "BAL": ["Corbin Burnes", "Grayson Rodriguez", "Cole Irvin", "Kyle Bradish", "Dean Kremer", 
                   "Albert Suárez", "John Means", "Trevor Rogers", "Félix Bautista"],
            "NYY": ["Gerrit Cole", "Carlos Rodón", "Clarke Schmidt", "Nestor Cortes", "Marcus Stroman",
                   "Clay Holmes", "Luke Weaver", "Tommy Kahnle", "Jonathan Loáisiga"],
            "BOS": ["Brayan Bello", "Kutter Crawford", "Nick Pivetta", "Tanner Houck", "James Paxton",
                   "Kenley Jansen", "Chris Martin", "Brennan Bernardino", "Josh Winckowski"],
            "LAD": ["Yoshinobu Yamamoto", "Tyler Glasnow", "Walker Buehler", "Gavin Stone", "Bobby Miller",
                   "Evan Phillips", "Blake Treinen", "Brusdar Graterol", "Daniel Hudson"]
        }
        
        # Choose pitchers for the selected team or use generic names
        if team in team_pitchers:
            pitchers = team_pitchers[team]
        else:
            pitchers = [f"Pitcher {i}" for i in range(1, 15)]
        
        # Provide realistic fallback data
        fallback_data = pd.DataFrame({
            'playerid': range(1, len(pitchers) + 1),
            'Name': pitchers,
            'Team': [team] * len(pitchers) if team != "All Teams" else np.random.choice(list(TEAMS.keys()), len(pitchers)),
            'Season': [season] * len(pitchers),
            'W': np.random.randint(0, 15, len(pitchers)),
            'L': np.random.randint(0, 12, len(pitchers)),
            'ERA': np.random.uniform(2.50, 5.50, len(pitchers)),
            'G': np.random.randint(5, 50, len(pitchers)),
            'IP': np.random.uniform(10, 180, len(pitchers)),
            'SO': np.random.randint(10, 220, len(pitchers)),
            'BB': np.random.randint(5, 80, len(pitchers)),
            'WHIP': np.random.uniform(0.95, 1.60, len(pitchers)),
            'K/9': np.random.uniform(5.5, 13.0, len(pitchers)),
            'BB/9': np.random.uniform(1.5, 5.0, len(pitchers)),
            'HR': np.random.randint(1, 30, len(pitchers)),
            'FIP': np.random.uniform(2.80, 5.80, len(pitchers)),
            'xFIP': np.random.uniform(3.20, 5.50, len(pitchers))
        })
        
        # Add role (starter/reliever)
        is_starter = fallback_data['IP'] > 40
        fallback_data['Role'] = np.where(is_starter, 'SP', 'RP')
        
        return fallback_data

@st.cache_data(ttl=3600)  # Cache for 1 hour in Streamlit
def fetch_player_statcast(player_name, start_date, end_date, player_type='batter'):
    """Fetch Statcast data for an individual player"""
    # Generate a cache key including all parameters
    cache_key = f"statcast_{player_name.replace(' ', '_')}_{start_date}_{end_date}_{player_type}"
    
    # First check our file cache
    cached_data = load_from_cache(cache_key, "", max_age_hours=cache_duration)
    if cached_data is not None:
        return cached_data
    
    try:
        if not PYBASEBALL_AVAILABLE:
            raise ImportError("Pybaseball is not available")
            
        # Look up player ID
        name_split = player_name.split(' ', 1)
        if len(name_split) != 2:
            st.warning(f"Could not parse player name: {player_name}. Please use format 'First Last'")
            return None
        
        first_name, last_name = name_split
        
        # Handle synthetic player names
        if player_name.startswith("Player ") or player_name.startswith("Pitcher "):
            # Create synthetic Statcast data
            days = (datetime.strptime(end_date, '%Y-%m-%d') - 
                   datetime.strptime(start_date, '%Y-%m-%d')).days + 1
            
            # Create synthetic data with appropriate columns based on player type
            if player_type == 'batter':
                synthetic_data = pd.DataFrame({
                    'game_date': [(datetime.strptime(start_date, '%Y-%m-%d') + 
                                  timedelta(days=i)).strftime('%Y-%m-%d') 
                                 for i in range(days)],
                    'launch_speed': np.random.normal(88, 7, days),
                    'launch_angle': np.random.normal(12, 8, days),
                    'estimated_ba_using_speedangle': np.random.uniform(0.2, 0.7, days),
                    'estimated_woba_using_speedangle': np.random.uniform(0.3, 0.8, days),
                    'events': np.random.choice(['single', 'double', 'home_run', 'field_out', 
                                              'strikeout', 'walk', None], days)
                })
            else:  # pitcher
                synthetic_data = pd.DataFrame({
                    'game_date': [(datetime.strptime(start_date, '%Y-%m-%d') + 
                                  timedelta(days=i)).strftime('%Y-%m-%d') 
                                 for i in range(days)],
                    'release_speed': np.random.normal(92, 4, days),
                    'release_spin_rate': np.random.normal(2200, 300, days),
                    'effective_speed': np.random.normal(93, 4, days),
                    'release_extension': np.random.normal(6.5, 0.3, days),
                    'pitch_type': np.random.choice(['FF', 'SI', 'SL', 'CH', 'CU', 'FC'], days),
                    'events': np.random.choice(['single', 'double', 'home_run', 'field_out', 
                                             'strikeout', 'walk', None], days)
                })
            
            # Save to cache
            save_to_cache(synthetic_data, cache_key, "")
            return synthetic_data
        
        st.info(f"Looking up player ID for {player_name}...")
        
        try:
            # Implement multiple lookup attempts with different name formatting
            attempts = [
                (last_name, first_name),  # Standard format
                (last_name.lower(), first_name.lower()),  # Lowercase
                (last_name.upper(), first_name.upper()),  # Uppercase
                # Add more attempts if needed
            ]
            
            player_lookup = None
            for last, first in attempts:
                try:
                    lookup_result = playerid_lookup(last, first)
                    if not lookup_result.empty:
                        player_lookup = lookup_result
                        break
                except Exception as e:
                    if show_debug_info:
                        st.warning(f"Lookup attempt failed for {first} {last}: {e}")
                    continue
            
            if player_lookup is None or player_lookup.empty:
                st.warning(f"Could not find player ID for {player_name}")
                return None
            
            # Get player's MLBAM ID
            mlbam_id = player_lookup.iloc[0]['key_mlbam']
            
            # Fetch a shorter date range to avoid API issues
            # Use last 30 days or the specified range if it's shorter
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            
            # If the range is more than 30 days, limit to last 30 days
            if (end_dt - start_dt).days > 30:
                start_dt = end_dt - timedelta(days=30)
                adjusted_start_date = start_dt.strftime('%Y-%m-%d')
                st.info(f"Limiting Statcast query to last 30 days ({adjusted_start_date} to {end_date}) due to API limitations")
            else:
                adjusted_start_date = start_date
            
            # Fetch Statcast data based on player type
            st.info(f"Fetching Statcast data for {player_name}...")
            
            if player_type == 'pitcher':
                data = statcast_pitcher(adjusted_start_date, end_date, mlbam_id)
            else:
                data = statcast_batter(adjusted_start_date, end_date, mlbam_id)
            
            # Handle empty results
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                st.warning(f"No Statcast data found for {player_name} in the specified date range.")
                return None
            
            # Save to cache
            save_to_cache(data, cache_key, "")
            
            return data
        except Exception as e:
            st.error(f"Error in player lookup or Statcast retrieval: {str(e)}")
            if show_debug_info:
                st.error(traceback.format_exc())
            return None
            
    except Exception as e:
        st.error(f"Error fetching Statcast data: {str(e)}")
        if show_debug_info:
            st.error(traceback.format_exc())
        return None

# Try direct Baseball Savant fetch as backup
def fetch_baseball_savant_direct(player_name, season):
    """
    Alternative direct fetch from Baseball Savant as backup method
    """
    try:
        # This is a basic implementation - would need to be expanded for a real application
        st.info(f"Attempting direct fetch from Baseball Savant for {player_name}...")
        
        # Would implement direct API calls or web scraping here
        # For now, return None to indicate this isn't implemented yet
        return None
    except Exception as e:
        if show_debug_info:
            st.error(f"Error with direct Baseball Savant fetch: {e}")
        return None

# Custom metrics for the dashboard
def add_custom_metrics(df):
    """Add custom calculated metrics to the dataframe"""
    if df is None or df.empty:
        return df
    
    try:
        # Create hot hand metrics
        
        # 1. First half vs second half performance (simulated)
        # In a real implementation, this would use actual first/second half data
        df['first_half_avg'] = df['AVG'] + np.random.normal(0, 0.02, len(df)) if 'AVG' in df.columns else np.random.uniform(0.220, 0.330, len(df))
        df['second_half_avg'] = df['AVG'] + np.random.normal(0, 0.02, len(df)) if 'AVG' in df.columns else np.random.uniform(0.220, 0.330, len(df))
        df['avg_diff'] = df['second_half_avg'] - df['first_half_avg']
        
        # 2. Recent "hot hand" metrics (simulated)
        # Would use actual recent vs. overall performance in production
        if 'wOBA' in df.columns:
            df['recent_wOBA'] = df['wOBA'] + np.random.uniform(-0.050, 0.050, len(df))
            df['wOBA_diff'] = df['recent_wOBA'] - df['wOBA']
        else:
            # Use OPS if wOBA not available
            if 'OPS' in df.columns:
                df['recent_OPS'] = df['OPS'] + np.random.uniform(-0.050, 0.050, len(df))
                df['OPS_diff'] = df['recent_OPS'] - df['OPS']
        
        # 3. Add hot/cold classification
        if 'wOBA_diff' in df.columns:
            df['hot_cold_status'] = pd.cut(
                df['wOBA_diff'],
                bins=[-float('inf'), cold_threshold, hot_threshold, float('inf')],
                labels=['Cold', 'Neutral', 'Hot']
            )
        elif 'OPS_diff' in df.columns:
            df['hot_cold_status'] = pd.cut(
                df['OPS_diff'],
                bins=[-float('inf'), cold_threshold, hot_threshold, float('inf')],
                labels=['Cold', 'Neutral', 'Hot']
            )
        
        # 4. Advanced metrics - could be implemented with actual mathematical models
        df['consistency_score'] = np.random.uniform(0.01, 0.15, len(df))
        df['clutch_factor'] = np.random.normal(0, 0.05, len(df))
        df['streak_potential'] = np.random.uniform(0.1, 0.9, len(df))
        
        return df
    
    except Exception as e:
        st.error(f"Error adding custom metrics: {e}")
        # Return the original dataframe if there's an error
        return df

# Function to calculate rolling performance from statcast data
def calculate_rolling_performance(player_data, window_size=10):
    """Calculate rolling performance metrics from statcast data"""
    if player_data is None or player_data.empty:
        return None
    
    try:
        # Group by date if not already
        if 'game_date' in player_data.columns:
            # For batters
            if 'estimated_ba_using_speedangle' in player_data.columns:
                daily_stats = player_data.groupby('game_date').agg({
                    'estimated_ba_using_speedangle': 'mean',
                    'estimated_woba_using_speedangle': 'mean' if 'estimated_woba_using_speedangle' in player_data.columns else None,
                    'launch_speed': 'mean' if 'launch_speed' in player_data.columns else None,
                    'launch_angle': 'mean' if 'launch_angle' in player_data.columns else None
                }).reset_index()
            # For pitchers
            elif 'release_speed' in player_data.columns:
                daily_stats = player_data.groupby('game_date').agg({
                    'release_speed': 'mean',
                    'release_spin_rate': 'mean' if 'release_spin_rate' in player_data.columns else None,
                    'effective_speed': 'mean' if 'effective_speed' in player_data.columns else None
                }).reset_index()
            else:
                return None
            
            # Remove None columns
            daily_stats = daily_stats.loc[:, ~daily_stats.columns.isin([None])]
            
            if daily_stats.empty:
                return None
                
            # Calculate rolling statistics
            if len(daily_stats) >= 3:  # Need at least 3 points for rolling window
                # Apply rolling window to each numeric column
                for col in daily_stats.columns:
                    if col != 'game_date' and pd.api.types.is_numeric_dtype(daily_stats[col]):
                        daily_stats[f'rolling_{col}'] = daily_stats[col].rolling(
                            window=min(window_size, len(daily_stats)), 
                            min_periods=1
                        ).mean()
            
            return daily_stats
        
        return None
    except Exception as e:
        if show_debug_info:
            st.error(f"Error calculating rolling performance: {e}")
        return None

# Function to direct fetch data from Baseball Savant if needed
def fetch_from_baseball_savant(player_id, start_date, end_date):
    """Direct fetch from Baseball Savant as a backup method"""
    # Implementation would go here for a direct API call
    # This is a placeholder for future enhancement
    return None

# Main app layout
def main():
    # Tabs for different dashboard views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Season Overview", 
        "Player Deep Dive", 
        "Hot Hand Analysis",
        "Predictions"
    ])
    
    # Fetch data based on user selection
    if player_type == "Batting":
        data = fetch_batting_data(selected_season, selected_team)
    else:
        data = fetch_pitching_data(selected_season, selected_team)
    
    if data is None or data.empty:
        st.error("No data available. Please try different filter settings.")
        return
        
    # Add custom metrics
    data = add_custom_metrics(data)
    
    with tab1:
        st.header(f"{selected_season} Season Overview")
        
        # Basic league-wide stats
        if selected_team == "All Teams":
            st.subheader("MLB League Overview")
            
            # Team summary metrics
            if 'Team' in data.columns:
                # Create team-level stats
                team_summary = data.groupby('Team').agg({
                    'Name': 'count',
                    'HR': 'sum' if 'HR' in data.columns else None,
                    'R': 'sum' if 'R' in data.columns else None,
                    'AVG': lambda x: np.average(x, weights=data.loc[x.index, 'PA']) if 'AVG' in data.columns and 'PA' in data.columns else None,
                    'OBP': lambda x: np.average(x, weights=data.loc[x.index, 'PA']) if 'OBP' in data.columns and 'PA' in data.columns else None,
                    'SLG': lambda x: np.average(x, weights=data.loc[x.index, 'PA']) if 'SLG' in data.columns and 'PA' in data.columns else None,
                    'ERA': lambda x: np.average(x, weights=data.loc[x.index, 'IP']) if 'ERA' in data.columns and 'IP' in data.columns else None,
                    'WHIP': lambda x: np.average(x, weights=data.loc[x.index, 'IP']) if 'WHIP' in data.columns and 'IP' in data.columns else None,
                    'SO': 'sum' if 'SO' in data.columns else None,
                    'W': 'sum' if 'W' in data.columns else None,
                    'SV': 'sum' if 'SV' in data.columns else None,
                }).reset_index()
                
                # Remove None columns
                team_summary = team_summary.loc[:, ~team_summary.columns.isin([None])]
                
                # Rename columns
                team_summary.rename(columns={'Name': 'Players'}, inplace=True)
                
                # Display team summary
                st.subheader("Team Summary")
                
                # Format the dataframe for display
                display_cols = [col for col in team_summary.columns if col != 'Team']
                
                # Add team colors
                if player_type == "Batting":
                    # Set up metrics for team comparison viz
                    if 'HR' in team_summary.columns:
                        metric_col = 'HR'
                        metric_name = 'Home Runs'
                    elif 'R' in team_summary.columns:
                        metric_col = 'R'
                        metric_name = 'Runs'
                    elif 'AVG' in team_summary.columns:
                        metric_col = 'AVG'
                        metric_name = 'Batting Average'
                    else:
                        metric_col = None
                else:  # Pitching
                    if 'ERA' in team_summary.columns:
                        metric_col = 'ERA'
                        metric_name = 'ERA'
                    elif 'WHIP' in team_summary.columns:
                        metric_col = 'WHIP'
                        metric_name = 'WHIP'
                    elif 'SO' in team_summary.columns:
                        metric_col = 'SO'
                        metric_name = 'Strikeouts'
                    else:
                        metric_col = None
                
                # Create a bar chart of team metrics
                if metric_col is not None:
                    # Sort data
                    if metric_col in ['ERA', 'WHIP']:
                        # Lower is better for these metrics
                        sorted_data = team_summary.sort_values(metric_col, ascending=True)
                    else:
                        # Higher is better for these metrics
                        sorted_data = team_summary.sort_values(metric_col, ascending=False)
                    
                    # Get team colors for visualization
                    team_colors = []
                    for team in sorted_data['Team']:
                        if team in TEAMS:
                            team_colors.append(TEAMS[team]['color'])
                        else:
                            team_colors.append('#CCCCCC')  # Default gray
                    
                    fig = px.bar(
                        sorted_data,
                        x='Team',
                        y=metric_col,
                        title=f"Team {metric_name} Comparison",
                        color='Team',
                        color_discrete_map=dict(zip(sorted_data['Team'], team_colors))
                    )
                    
                    # Adjust layout
                    fig.update_layout(
                        xaxis_title="Team",
                        yaxis_title=metric_name,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display the raw data table
                st.dataframe(team_summary, use_container_width=True)
        
        # Team-specific stats
        else:
            team_color = TEAMS.get(selected_team, {}).get('color', '#1E88E5')
            team_name = TEAMS.get(selected_team, {}).get('name', selected_team)
            
            st.subheader(f"{team_name} Team Overview")
            
            # Display team metrics on top
            if player_type == "Batting":
                metrics = {}
                if 'AVG' in data.columns and 'PA' in data.columns:
                    metrics['Batting Average'] = np.average(data['AVG'], weights=data['PA'])
                if 'HR' in data.columns:
                    metrics['Home Runs'] = data['HR'].sum()
                if 'R' in data.columns:
                    metrics['Runs'] = data['R'].sum()
                if 'OPS' in data.columns and 'PA' in data.columns:
                    metrics['OPS'] = np.average(data['OPS'], weights=data['PA'])
            else:  # Pitching
                metrics = {}
                if 'ERA' in data.columns and 'IP' in data.columns:
                    metrics['ERA'] = np.average(data['ERA'], weights=data['IP'])
                if 'WHIP' in data.columns and 'IP' in data.columns:
                    metrics['WHIP'] = np.average(data['WHIP'], weights=data['IP'])
                if 'SO' in data.columns:
                    metrics['Strikeouts'] = data['SO'].sum()
                if 'W' in data.columns:
                    metrics['Wins'] = data['W'].sum()
            
            # Display team metrics in columns
            if metrics:
                cols = st.columns(len(metrics))
                for i, (metric_name, metric_value) in enumerate(metrics.items()):
                    with cols[i]:
                        if isinstance(metric_value, float):
                            st.metric(metric_name, f"{metric_value:.3f}")
                        else:
                            st.metric(metric_name, metric_value)
            
            # Display player stats
            if player_type == "Batting":
                st.subheader(f"{team_name} Batting Leaders")
                
                # Common batting stats to display
                display_columns = ['Name', 'PA', 'AVG', 'OBP', 'SLG', 'OPS', 'wOBA', 'wRC+', 'HR']
                
                # Make sure all required columns exist
                valid_columns = [col for col in display_columns if col in data.columns]
                
                # Sort data
                if 'wOBA' in data.columns:
                    sorted_data = data.sort_values('wOBA', ascending=False)
                elif 'OPS' in data.columns:
                    sorted_data = data.sort_values('OPS', ascending=False)
                else:
                    sorted_data = data.sort_values('AVG', ascending=False)
                
                # Display the data table
                st.dataframe(sorted_data[valid_columns], use_container_width=True)
                
                # Create visualizations for team batting stats
                if 'HR' in data.columns:
                    # Home Run Leaders
                    hr_leaders = data.sort_values('HR', ascending=False).head(10)
                    
                    fig = px.bar(
                        hr_leaders,
                        x='Name',
                        y='HR',
                        title=f"{team_name} Home Run Leaders",
                        color_discrete_sequence=[team_color]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                if 'AVG' in data.columns:
                    # Batting Average Leaders (min 100 PA)
                    if 'PA' in data.columns:
                        avg_leaders = data[data['PA'] >= 100].sort_values('AVG', ascending=False).head(10)
                    else:
                        avg_leaders = data.sort_values('AVG', ascending=False).head(10)
                    
                    fig = px.bar(
                        avg_leaders,
                        x='Name',
                        y='AVG',
                        title=f"{team_name} Batting Average Leaders",
                        color_discrete_sequence=[team_color]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            else:  # Pitching
                st.subheader(f"{team_name} Pitching Leaders")
                
                # Common pitching stats to display
                display_columns = ['Name', 'W', 'L', 'ERA', 'G', 'IP', 'SO', 'WHIP']
                
                # Make sure all required columns exist
                valid_columns = [col for col in display_columns if col in data.columns]
                
                # Sort data
                if 'ERA' in data.columns and 'IP' in data.columns:
                    # Filter for minimum innings pitched
                    qualified = data[data['IP'] >= 10]
                    sorted_data = qualified.sort_values('ERA', ascending=True)
                else:
                    sorted_data = data.sort_values('IP', ascending=False)
                
                # Display the data table
                st.dataframe(sorted_data[valid_columns], use_container_width=True)
                
                # Create visualizations for team pitching stats
                if 'ERA' in data.columns and 'IP' in data.columns:
                    # ERA Leaders (min 20 IP for starters)
                    era_leaders = data[data['IP'] >= 20].sort_values('ERA', ascending=True).head(10)
                    
                    fig = px.bar(
                        era_leaders,
                        x='Name',
                        y='ERA',
                        title=f"{team_name} ERA Leaders (min 20 IP)",
                        color_discrete_sequence=[team_color]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                if 'SO' in data.columns:
                    # Strikeout Leaders
                    k_leaders = data.sort_values('SO', ascending=False).head(10)
                    
                    fig = px.bar(
                        k_leaders,
                        x='Name',
                        y='SO',
                        title=f"{team_name} Strikeout Leaders",
                        color_discrete_sequence=[team_color]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Display hot/cold status distribution across league or team
        if 'hot_cold_status' in data.columns:
            st.subheader("Hot/Cold Players Distribution")
            
            hot_count = (data['hot_cold_status'] == 'Hot').sum()
            neutral_count = (data['hot_cold_status'] == 'Neutral').sum()
            cold_count = (data['hot_cold_status'] == 'Cold').sum()
            
            fig = px.pie(
                values=[hot_count, neutral_count, cold_count],
                names=['Hot', 'Neutral', 'Cold'],
                title=f"Player Hot/Cold Distribution ({player_type})",
                color=['Hot', 'Neutral', 'Cold'],
                color_discrete_map={'Hot': 'red', 'Neutral': 'gray', 'Cold': 'blue'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Player Deep Dive")
        
        # Select player for deep dive
        if not data.empty:
            player_list = data['Name'].tolist()
            selected_player = st.selectbox("Select Player for Analysis", player_list)
            
            if selected_player:
                st.subheader(f"Analysis for {selected_player}")
                
                # Get player info
                player_row = data[data['Name'] == selected_player]
                player_team = player_row['Team'].values[0] if 'Team' in player_row.columns else selected_team
                team_color = TEAMS.get(player_team, {}).get('color', '#1E88E5')
                
                # Display basic player statistics
                st.subheader("Season Statistics")
                
                # Create columns for key metrics
                if player_type == "Batting":
                    # Define the metrics to show
                    metrics = []
                    if 'AVG' in player_row.columns:
                        metrics.append(("AVG", player_row['AVG'].values[0], None))
                    if 'OBP' in player_row.columns:
                        metrics.append(("OBP", player_row['OBP'].values[0], None))
                    if 'SLG' in player_row.columns:
                        metrics.append(("SLG", player_row['SLG'].values[0], None))
                    if 'HR' in player_row.columns:
                        metrics.append(("HR", player_row['HR'].values[0], None))
                    if 'RBI' in player_row.columns:
                        metrics.append(("RBI", player_row['RBI'].values[0], None))
                    if 'wOBA' in player_row.columns:
                        metrics.append(("wOBA", player_row['wOBA'].values[0], None))
                    
                    # Add hot/cold indicator if available
                    if 'hot_cold_status' in player_row.columns:
                        hot_status = player_row['hot_cold_status'].values[0]
                        if 'wOBA_diff' in player_row.columns:
                            woba_diff = player_row['wOBA_diff'].values[0]
                            metrics.append(("Hot/Cold", hot_status, f"{woba_diff:.3f}" if woba_diff is not None else None))
                    
                else:  # Pitching
                    # Define the metrics to show
                    metrics = []
                    if 'ERA' in player_row.columns:
                        metrics.append(("ERA", player_row['ERA'].values[0], None))
                    if 'WHIP' in player_row.columns:
                        metrics.append(("WHIP", player_row['WHIP'].values[0], None))
                    if 'W' in player_row.columns and 'L' in player_row.columns:
                        w = player_row['W'].values[0]
                        l = player_row['L'].values[0]
                        metrics.append(("W-L", f"{w}-{l}", None))
                    if 'SO' in player_row.columns:
                        metrics.append(("SO", player_row['SO'].values[0], None))
                    if 'IP' in player_row.columns:
                        metrics.append(("IP", player_row['IP'].values[0], None))
                    if 'K/9' in player_row.columns:
                        metrics.append(("K/9", player_row['K/9'].values[0], None))
                
                # Display metrics in columns
                if metrics:
                    # Calculate number of columns needed (up to 4 per row)
                    num_metrics = len(metrics)
                    num_rows = (num_metrics + 3) // 4  # Ceiling division
                    
                    for row in range(num_rows):
                        cols = st.columns(min(4, num_metrics - row * 4))
                        for i, (name, value, delta) in enumerate(metrics[row * 4:min((row + 1) * 4, num_metrics)]):
                            with cols[i]:
                                if isinstance(value, float):
                                    st.metric(name, f"{value:.3f}", delta)
                                else:
                                    st.metric(name, value, delta)
                
                # Attempt to get Statcast data
                today = datetime.now().strftime('%Y-%m-%d')
                thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                
                player_data = fetch_player_statcast(
                    selected_player, 
                    thirty_days_ago, 
                    today,
                    player_type.lower()
                )
                
                if player_data is not None and not player_data.empty:
                    st.success(f"Successfully retrieved Statcast data for {selected_player}")
                    
                    # Process Statcast data for visualization
                    if player_type == "Batting":
                        st.subheader("Batting Analysis")
                        
                        # Exit Velocity & Launch Angle
                        if 'launch_speed' in player_data.columns and 'launch_angle' in player_data.columns:
                            # Filter out null values
                            ev_data = player_data.dropna(subset=['launch_speed', 'launch_angle'])
                            
                            if not ev_data.empty:
                                fig = px.scatter(
                                    ev_data,
                                    x='launch_speed',
                                    y='launch_angle',
                                    color='events',
                                    hover_name='events',
                                    color_discrete_map={
                                        'home_run': team_color,
                                        'double': '#4CAF50',
                                        'single': '#FFC107',
                                        'triple': '#9C27B0',
                                        'field_out': '#9E9E9E',
                                        'strikeout': '#F44336'
                                    },
                                    title=f"{selected_player} - Exit Velocity & Launch Angle"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No valid exit velocity and launch angle data available.")
                        
                        # Calculate rolling performance
                        rolling_data = calculate_rolling_performance(player_data, rolling_window)
                        
                        if rolling_data is not None:
                            # Plot rolling metrics
                            st.subheader("Performance Trends")
                            
                            # Expected stats trends
                            if 'estimated_ba_using_speedangle' in rolling_data.columns and 'rolling_estimated_ba_using_speedangle' in rolling_data.columns:
                                fig = px.line(
                                    rolling_data,
                                    x='game_date',
                                    y=['estimated_ba_using_speedangle', 'rolling_estimated_ba_using_speedangle'],
                                    title=f"{selected_player} - Rolling Expected Batting Average ({rolling_window}-game window)",
                                    labels={'value': 'Expected Batting Average', 'variable': 'Metric'},
                                    color_discrete_map={
                                        'estimated_ba_using_speedangle': '#9E9E9E',
                                        'rolling_estimated_ba_using_speedangle': team_color
                                    }
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Exit velocity trends
                            if 'launch_speed' in rolling_data.columns and 'rolling_launch_speed' in rolling_data.columns:
                                fig = px.line(
                                    rolling_data,
                                    x='game_date',
                                    y=['launch_speed', 'rolling_launch_speed'],
                                    title=f"{selected_player} - Rolling Exit Velocity ({rolling_window}-game window)",
                                    labels={'value': 'Exit Velocity (mph)', 'variable': 'Metric'},
                                    color_discrete_map={
                                        'launch_speed': '#9E9E9E',
                                        'rolling_launch_speed': team_color
                                    }
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    
                    else:  # Pitching
                        st.subheader("Pitching Analysis")
                        
                        # Velocity analysis
                        if 'release_speed' in player_data.columns:
                            # Group by pitch type
                            if 'pitch_type' in player_data.columns:
                                pitch_velo = player_data.groupby('pitch_type')['release_speed'].agg(['mean', 'std', 'count']).reset_index()
                                pitch_velo = pitch_velo[pitch_velo['count'] >= 5]  # Filter for pitch types with enough samples
                                
                                if not pitch_velo.empty:
                                    fig = px.bar(
                                        pitch_velo,
                                        x='pitch_type',
                                        y='mean',
                                        error_y='std',
                                        title=f"{selected_player} - Average Velocity by Pitch Type",
                                        labels={'mean': 'Velocity (mph)', 'pitch_type': 'Pitch Type'},
                                        color_discrete_sequence=[team_color]
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # Calculate rolling performance
                            rolling_data = calculate_rolling_performance(player_data, rolling_window)
                            
                            if rolling_data is not None:
                                # Plot rolling metrics
                                st.subheader("Performance Trends")
                                
                                # Velocity trends
                                if 'release_speed' in rolling_data.columns and 'rolling_release_speed' in rolling_data.columns:
                                    fig = px.line(
                                        rolling_data,
                                        x='game_date',
                                        y=['release_speed', 'rolling_release_speed'],
                                        title=f"{selected_player} - Rolling Velocity ({rolling_window}-game window)",
                                        labels={'value': 'Velocity (mph)', 'variable': 'Metric'},
                                        color_discrete_map={
                                            'release_speed': '#9E9E9E',
                                            'rolling_release_speed': team_color
                                        }
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display raw data 
                    st.subheader("Recent Statcast Data")
                    
                    # Select columns based on player type
                    if player_type == "Batting":
                        display_cols = ['game_date', 'events', 'launch_speed', 'launch_angle']
                        if 'estimated_ba_using_speedangle' in player_data.columns:
                            display_cols.append('estimated_ba_using_speedangle')
                    else:  # Pitching
                        display_cols = ['game_date', 'events', 'pitch_type', 'release_speed']
                        if 'release_spin_rate' in player_data.columns:
                            display_cols.append('release_spin_rate')
                    
                    valid_display_cols = [col for col in display_cols if col in player_data.columns]
                    
                    # Only show a sample of the data
                    st.dataframe(player_data[valid_display_cols].tail(10))
                    
                    # Option to download full data
                    csv = player_data[valid_display_cols].to_csv(index=False)
                    st.download_button(
                        label="Download Complete Statcast Data",
                        data=csv,
                        file_name=f"{selected_player}_statcast_data.csv",
                        mime="text/csv"
                    )
                
                else:
                    st.info("Using simulated performance data since Statcast data is limited.")
                    
                    # Create simulated monthly data
                    if 'AVG' in player_row.columns or player_type == "Batting":
                        # Batting simulation
                        months = ["Apr", "May", "Jun", "Jul", "Aug", "Sep"]
                        stats = {}
                        
                        # Base metrics
                        base_avg = player_row['AVG'].values[0] if 'AVG' in player_row.columns else 0.250
                        base_obp = player_row['OBP'].values[0] if 'OBP' in player_row.columns else 0.320
                        base_slg = player_row['SLG'].values[0] if 'SLG' in player_row.columns else 0.400
                        
                        # Create random variations around the base value
                        stats['AVG'] = np.clip(base_avg + np.random.normal(0, 0.025, len(months)), 0.150, 0.400)
                        stats['OBP'] = np.clip(base_obp + np.random.normal(0, 0.030, len(months)), 0.200, 0.500)
                        stats['SLG'] = np.clip(base_slg + np.random.normal(0, 0.050, len(months)), 0.250, 0.700)
                        stats['PA'] = np.random.randint(50, 120, len(months))
                        stats['HR'] = np.random.randint(0, 10, len(months))
                        
                        # Create player data
                        player_monthly_data = pd.DataFrame({
                            'Month': months,
                            **stats
                        })
                        
                        # Compute OPS
                        player_monthly_data['OPS'] = player_monthly_data['OBP'] + player_monthly_data['SLG']
                        
                        # Plot monthly batting average
                        st.subheader("Monthly Performance (Simulated)")
                        fig = px.line(
                            player_monthly_data,
                            x='Month',
                            y='AVG',
                            title=f"{selected_player} - Monthly Batting Average",
                            markers=True,
                            line_shape='spline',
                            color_discrete_sequence=[team_color]
                        )
                        
                        # Add season average reference line
                        fig.add_hline(
                            y=base_avg,
                            line_dash="dash",
                            line_color="#000000",
                            annotation_text=f"Season AVG: {base_avg:.3f}",
                            annotation_position="bottom right"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Plot monthly OPS
                        fig = px.line(
                            player_monthly_data,
                            x='Month',
                            y='OPS',
                            title=f"{selected_player} - Monthly OPS",
                            markers=True,
                            line_shape='spline',
                            color_discrete_sequence=[team_color]
                        )
                        
                        # Add season average reference line
                        base_ops = base_obp + base_slg
                        fig.add_hline(
                            y=base_ops,
                            line_dash="dash",
                            line_color="#000000",
                            annotation_text=f"Season OPS: {base_ops:.3f}",
                            annotation_position="bottom right"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Plot monthly home runs
                        fig = px.bar(
                            player_monthly_data,
                            x='Month',
                            y='HR',
                            title=f"{selected_player} - Monthly Home Runs",
                            color_discrete_sequence=[team_color]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display monthly stats as a table
                        st.subheader("Monthly Breakdown (Simulated)")
                        st.dataframe(player_monthly_data)
                        
                    elif player_type == "Pitching":
                        # Pitching simulation
                        months = ["Apr", "May", "Jun", "Jul", "Aug", "Sep"]
                        stats = {}
                        
                        # Base metrics
                        base_era = player_row['ERA'].values[0] if 'ERA' in player_row.columns else 4.00
                        base_whip = player_row['WHIP'].values[0] if 'WHIP' in player_row.columns else 1.30
                        base_k9 = player_row['K/9'].values[0] if 'K/9' in player_row.columns else 8.5
                        
                        # Create random variations around the base value
                        stats['ERA'] = np.clip(base_era + np.random.normal(0, 0.50, len(months)), 1.50, 7.50)
                        stats['WHIP'] = np.clip(base_whip + np.random.normal(0, 0.15, len(months)), 0.80, 2.00)
                        stats['K/9'] = np.clip(base_k9 + np.random.normal(0, 0.75, len(months)), 5.0, 14.0)
                        stats['IP'] = np.random.uniform(10, 35, len(months))
                        stats['SO'] = np.random.randint(10, 40, len(months))
                        
                        # Create player data
                        player_monthly_data = pd.DataFrame({
                            'Month': months,
                            **stats
                        })
                        
                        # Plot monthly ERA
                        st.subheader("Monthly Performance (Simulated)")
                        fig = px.line(
                            player_monthly_data,
                            x='Month',
                            y='ERA',
                            title=f"{selected_player} - Monthly ERA",
                            markers=True,
                            line_shape='spline',
                            color_discrete_sequence=[team_color]
                        )
                        
                        # Add season average reference line
                        fig.add_hline(
                            y=base_era,
                            line_dash="dash",
                            line_color="#000000",
                            annotation_text=f"Season ERA: {base_era:.2f}",
                            annotation_position="top right"
                        )
                        
                        # Lower y-axis is better for ERA
                        fig.update_yaxes(autorange="reversed")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Plot monthly K/9
                        fig = px.line(
                            player_monthly_data,
                            x='Month',
                            y='K/9',
                            title=f"{selected_player} - Monthly K/9",
                            markers=True,
                            line_shape='spline',
                            color_discrete_sequence=[team_color]
                        )
                        
                        # Add season average reference line
                        fig.add_hline(
                            y=base_k9,
                            line_dash="dash",
                            line_color="#000000",
                            annotation_text=f"Season K/9: {base_k9:.2f}",
                            annotation_position="bottom right"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Plot monthly strikeouts
                        fig = px.bar(
                            player_monthly_data,
                            x='Month',
                            y='SO',
                            title=f"{selected_player} - Monthly Strikeouts",
                            color_discrete_sequence=[team_color]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display monthly stats as a table
                        st.subheader("Monthly Breakdown (Simulated)")
                        st.dataframe(player_monthly_data)
    
    with tab3:
        st.header("Hot Hand Analysis")
        
        if 'hot_cold_status' in data.columns:
            # Filter players by hot/cold status
            hot_players = data[data['hot_cold_status'] == 'Hot'].sort_values(
                'wOBA_diff' if 'wOBA_diff' in data.columns else 'OPS_diff' if 'OPS_diff' in data.columns else 'avg_diff',
                ascending=False
            )
            
            cold_players = data[data['hot_cold_status'] == 'Cold'].sort_values(
                'wOBA_diff' if 'wOBA_diff' in data.columns else 'OPS_diff' if 'OPS_diff' in data.columns else 'avg_diff',
                ascending=True
            )
            
            # Display hot players
            st.subheader("🔥 Hot Players")
            if not hot_players.empty:
                # Select columns for display, ensuring they exist
                if player_type == "Batting":
                    display_cols = ['Name', 'Team']
                    
                    # Add appropriate performance columns
                    if 'wOBA' in hot_players.columns and 'recent_wOBA' in hot_players.columns:
                        display_cols.extend(['wOBA', 'recent_wOBA', 'wOBA_diff'])
                    elif 'OPS' in hot_players.columns and 'recent_OPS' in hot_players.columns:
                        display_cols.extend(['OPS', 'recent_OPS', 'OPS_diff'])
                    elif 'AVG' in hot_players.columns:
                        display_cols.extend(['AVG', 'first_half_avg', 'second_half_avg', 'avg_diff'])
                else:  # Pitching
                    display_cols = ['Name', 'Team']
                    
                    # Add appropriate performance columns
                    if 'ERA' in hot_players.columns:
                        display_cols.append('ERA')
                    if 'WHIP' in hot_players.columns:
                        display_cols.append('WHIP')
                    if 'K/9' in hot_players.columns:
                        display_cols.append('K/9')
                    if 'avg_diff' in hot_players.columns:
                        display_cols.append('avg_diff')
                
                # Make sure all required columns exist
                valid_cols = [col for col in display_cols if col in hot_players.columns]
                
                hot_df = hot_players[valid_cols].copy()
                
                # Rename columns for display if needed
                column_mapping = {
                    'wOBA': 'Season wOBA', 
                    'recent_wOBA': 'Recent wOBA', 
                    'wOBA_diff': 'Difference',
                    'OPS': 'Season OPS',
                    'recent_OPS': 'Recent OPS', 
                    'OPS_diff': 'Difference',
                    'first_half_avg': 'First Half AVG',
                    'second_half_avg': 'Second Half AVG',
                    'avg_diff': 'Difference'
                }
                
                # Only rename columns that exist
                rename_cols = {k: v for k, v in column_mapping.items() if k in hot_df.columns}
                hot_df.rename(columns=rename_cols, inplace=True)
                
                # Format difference columns if they exist
                diff_cols = ['Difference']
                for col in diff_cols:
                    if col in hot_df.columns:
                        hot_df[col] = hot_df[col].map(lambda x: f"{x:+.3f}" if isinstance(x, (int, float)) else x)
                
                # Display the dataframe
                st.dataframe(hot_df, use_container_width=True)
                
                # Create visualization of hot players
                vis_metric = None
                if 'wOBA_diff' in hot_players.columns:
                    vis_metric = 'wOBA_diff'
                    metric_name = 'wOBA Difference'
                elif 'OPS_diff' in hot_players.columns:
                    vis_metric = 'OPS_diff'
                    metric_name = 'OPS Difference'
                elif 'avg_diff' in hot_players.columns:
                    vis_metric = 'avg_diff'
                    metric_name = 'AVG Difference'
                
                if vis_metric is not None:
                    # Use team colors for bars
                    team_colors = []
                    for team in hot_players.head(10)['Team']:
                        if team in TEAMS:
                            team_colors.append(TEAMS[team]['color'])
                        else:
                            team_colors.append('#CCCCCC')  # Default gray
                    
                    fig = px.bar(
                        hot_players.head(10),
                        x='Name',
                        y=vis_metric,
                        color='Team',
                        title=f"Hot Players ({metric_name})",
                        color_discrete_map=dict(zip(hot_players.head(10)['Team'], team_colors))
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title="Player",
                        yaxis_title=metric_name
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No hot players found based on current thresholds.")
            
            # Display cold players
            st.subheader("❄️ Cold Players")
            if not cold_players.empty:
                # Select columns for display, ensuring they exist
                if player_type == "Batting":
                    display_cols = ['Name', 'Team']
                    
                    # Add appropriate performance columns
                    if 'wOBA' in cold_players.columns and 'recent_wOBA' in cold_players.columns:
                        display_cols.extend(['wOBA', 'recent_wOBA', 'wOBA_diff'])
                    elif 'OPS' in cold_players.columns and 'recent_OPS' in cold_players.columns:
                        display_cols.extend(['OPS', 'recent_OPS', 'OPS_diff'])
                    elif 'AVG' in cold_players.columns:
                        display_cols.extend(['AVG', 'first_half_avg', 'second_half_avg', 'avg_diff'])
                else:  # Pitching
                    display_cols = ['Name', 'Team']
                    
                    # Add appropriate performance columns
                    if 'ERA' in cold_players.columns:
                        display_cols.append('ERA')
                    if 'WHIP' in cold_players.columns:
                        display_cols.append('WHIP')
                    if 'K/9' in cold_players.columns:
                        display_cols.append('K/9')
                    if 'avg_diff' in cold_players.columns:
                        display_cols.append('avg_diff')
                
                # Make sure all required columns exist
                valid_cols = [col for col in display_cols if col in cold_players.columns]
                
                cold_df = cold_players[valid_cols].copy()
                
                # Rename columns for display if needed
                column_mapping = {
                    'wOBA': 'Season wOBA', 
                    'recent_wOBA': 'Recent wOBA', 
                    'wOBA_diff': 'Difference',
                    'OPS': 'Season OPS',
                    'recent_OPS': 'Recent OPS', 
                    'OPS_diff': 'Difference',
                    'first_half_avg': 'First Half AVG',
                    'second_half_avg': 'Second Half AVG',
                    'avg_diff': 'Difference'
                }
                
                # Only rename columns that exist
                rename_cols = {k: v for k, v in column_mapping.items() if k in cold_df.columns}
                cold_df.rename(columns=rename_cols, inplace=True)
                
                # Format difference columns if they exist
                diff_cols = ['Difference']
                for col in diff_cols:
                    if col in cold_df.columns:
                        cold_df[col] = cold_df[col].map(lambda x: f"{x:+.3f}" if isinstance(x, (int, float)) else x)
                
                # Display the dataframe
                st.dataframe(cold_df, use_container_width=True)
                
                # Create visualization of cold players
                vis_metric = None
                if 'wOBA_diff' in cold_players.columns:
                    vis_metric = 'wOBA_diff'
                    metric_name = 'wOBA Difference'
                elif 'OPS_diff' in cold_players.columns:
                    vis_metric = 'OPS_diff'
                    metric_name = 'OPS Difference'
                elif 'avg_diff' in cold_players.columns:
                    vis_metric = 'avg_diff'
                    metric_name = 'AVG Difference'
                
                if vis_metric is not None:
                    # Use team colors for bars
                    team_colors = []
                    for team in cold_players.head(10)['Team']:
                        if team in TEAMS:
                            team_colors.append(TEAMS[team]['color'])
                        else:
                            team_colors.append('#CCCCCC')  # Default gray
                    
                    fig = px.bar(
                        cold_players.head(10),
                        x='Name',
                        y=vis_metric,
                        color='Team',
                        title=f"Cold Players ({metric_name})",
                        color_discrete_map=dict(zip(cold_players.head(10)['Team'], team_colors))
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title="Player",
                        yaxis_title=metric_name
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No cold players found based on current thresholds.")
            
            # Hot hand distribution
            vis_metric = None
            if 'wOBA_diff' in data.columns:
                vis_metric = 'wOBA_diff'
                metric_name = 'wOBA Difference'
            elif 'OPS_diff' in data.columns:
                vis_metric = 'OPS_diff'
                metric_name = 'OPS Difference'
            elif 'avg_diff' in data.columns:
                vis_metric = 'avg_diff'
                metric_name = 'AVG Difference'
            
            if vis_metric is not None:
                st.subheader("Hot Hand Distribution")
                fig = px.histogram(
                    data,
                    x=vis_metric,
                    color='hot_cold_status',
                    color_discrete_map={'Hot': 'red', 'Neutral': 'gray', 'Cold': 'blue'},
                    nbins=30,
                    title=f"Distribution of {metric_name}"
                )
                fig.add_vline(x=hot_threshold, line_dash="dash", line_color="red")
                fig.add_vline(x=cold_threshold, line_dash="dash", line_color="blue")
                st.plotly_chart(fig, use_container_width=True)
                
                # Custom hot hand insights
                st.subheader("Hot Hand Insights")
                
                # Additional hot hand metrics
                if 'consistency_score' in data.columns and 'streak_potential' in data.columns:
                    st.write("Players most likely to get hot (high streak potential, not currently hot):")
                    streak_candidates = data[
                        (data['hot_cold_status'] != 'Hot') & 
                        (data['streak_potential'] > 0.7)
                    ].sort_values('streak_potential', ascending=False).head(5)
                    
                    if not streak_candidates.empty:
                        st.dataframe(streak_candidates[['Name', 'Team', 'streak_potential', 'consistency_score']])
                    
                    st.write("Most consistent performers (low consistency score, regardless of hot/cold):")
                    consistent_players = data.sort_values('consistency_score').head(5)
                    
                    if not consistent_players.empty:
                        st.dataframe(consistent_players[['Name', 'Team', 'consistency_score', 'hot_cold_status']])
                
                # Correlation with other stats
                st.subheader("Hot Hand Correlations")
                
                # Choose a performance metric to correlate with hot hand
                perf_metric = None
                if player_type == "Batting":
                    if 'HR' in data.columns:
                        perf_metric = 'HR'
                        perf_name = 'Home Runs'
                    elif 'RBI' in data.columns:
                        perf_metric = 'RBI'
                        perf_name = 'RBIs'
                    elif 'AVG' in data.columns:
                        perf_metric = 'AVG'
                        perf_name = 'Batting Average'
                else:  # Pitching
                    if 'ERA' in data.columns:
                        perf_metric = 'ERA'
                        perf_name = 'ERA'
                    elif 'WHIP' in data.columns:
                        perf_metric = 'WHIP'
                        perf_name = 'WHIP'
                    elif 'K/9' in data.columns:
                        perf_metric = 'K/9'
                        perf_name = 'K/9'
                
                if perf_metric is not None and vis_metric is not None:
                    fig = px.scatter(
                        data,
                        x=vis_metric,
                        y=perf_metric,
                        color='hot_cold_status',
                        color_discrete_map={'Hot': 'red', 'Neutral': 'gray', 'Cold': 'blue'},
                        title=f"{metric_name} vs {perf_name}",
                        hover_data=['Name', 'Team']
                    )
                    
                    # Add vertical lines for thresholds
                    fig.add_vline(x=hot_threshold, line_dash="dash", line_color="red")
                    fig.add_vline(x=cold_threshold, line_dash="dash", line_color="blue")
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Hot hand analysis requires performance difference data which is not available in the current dataset.")
    
    with tab4:
        st.header("Predictions")
        
        st.subheader("Game Outcome Prediction")
        
        # Simplified game prediction interface
        col1, col2 = st.columns(2)
        
        with col1:
            home_team = st.selectbox("Home Team", list(TEAMS.keys()), index=0)
            home_starter = st.text_input("Home Starting Pitcher (optional)")
        
        with col2:
            away_team = st.selectbox("Away Team", list(TEAMS.keys()), index=1)
            away_starter = st.text_input("Away Starting Pitcher (optional)")
        
        if st.button("Generate Prediction"):
            st.info("Calculating prediction...")
            
            # Simulate a prediction model
            # In a real implementation, this would use actual team stats and models
            home_team_color = TEAMS.get(home_team, {}).get('color', '#1E88E5')
            away_team_color = TEAMS.get(away_team, {}).get('color', '#1E88E5')
            
            # Get team stats if available
            if 'Team' in data.columns:
                home_stats = data[data['Team'] == home_team]
                away_stats = data[data['Team'] == away_team]
            else:
                home_stats = pd.DataFrame()
                away_stats = pd.DataFrame()
            
            # Generate win probabilities
            # This is a simplified simulation - a real model would use actual team performance
            home_adv = 0.05  # Home field advantage
            
            # Factor in team offense/pitching strength
            team_factor = 0
            if player_type == "Batting" and not home_stats.empty and not away_stats.empty:
                if 'wOBA' in home_stats.columns and 'wOBA' in away_stats.columns:
                    home_woba = home_stats['wOBA'].mean()
                    away_woba = away_stats['wOBA'].mean()
                    team_factor = (home_woba - away_woba) * 2  # Scale factor
                elif 'OPS' in home_stats.columns and 'OPS' in away_stats.columns:
                    home_ops = home_stats['OPS'].mean()
                    away_ops = away_stats['OPS'].mean()
                    team_factor = (home_ops - away_ops)
            elif player_type == "Pitching" and not home_stats.empty and not away_stats.empty:
                if 'ERA' in home_stats.columns and 'ERA' in away_stats.columns:
                    home_era = home_stats['ERA'].mean()
                    away_era = away_stats['ERA'].mean()
                    team_factor = (away_era - home_era) / 5  # Scale factor, reversed since lower ERA is better
            
            # Factor in hot players
            hot_factor = 0
            if 'hot_cold_status' in data.columns:
                home_hot = data[(data['Team'] == home_team) & (data['hot_cold_status'] == 'Hot')].shape[0]
                home_cold = data[(data['Team'] == home_team) & (data['hot_cold_status'] == 'Cold')].shape[0]
                away_hot = data[(data['Team'] == away_team) & (data['hot_cold_status'] == 'Hot')].shape[0]
                away_cold = data[(data['Team'] == away_team) & (data['hot_cold_status'] == 'Cold')].shape[0]
                
                hot_factor = ((home_hot - home_cold) - (away_hot - away_cold)) * 0.01
            
            # Starting pitcher factor (placeholder, would use actual pitcher stats)
            pitcher_factor = 0
            if home_starter and away_starter:
                # This would use actual pitcher data in a real model
                pitcher_factor = np.random.uniform(-0.05, 0.05)
            
            # Combine factors for win probability
            home_win_prob = 0.5 + home_adv + team_factor + hot_factor + pitcher_factor
            
            # Clip to valid probability range
            home_win_prob = np.clip(home_win_prob, 0.05, 0.95)
            away_win_prob = 1 - home_win_prob
            
            # Display prediction
            st.subheader("Game Prediction")
            
            # Display win probabilities
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"{TEAMS.get(home_team, {}).get('name', home_team)}")
                st.markdown(f"<h1 style='text-align: center; color: {home_team_color};'>{home_win_prob:.1%}</h1>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;'>Win Probability</p>", unsafe_allow_html=True)
            
            with col2:
                st.subheader(f"{TEAMS.get(away_team, {}).get('name', away_team)}")
                st.markdown(f"<h1 style='text-align: center; color: {away_team_color};'>{away_win_prob:.1%}</h1>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;'>Win Probability</p>", unsafe_allow_html=True)
            
            # Display a gauge chart for the win probability
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = home_win_prob * 100,
                title = {'text': f"{home_team} vs {away_team}"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': home_team_color},
                    'steps': [
                        {'range': [0, 40], 'color': away_team_color},
                        {'range': [40, 60], 'color': "lightgray"},
                        {'range': [60, 100], 'color': home_team_color}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': home_win_prob * 100
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Factors influencing the prediction
            st.subheader("Prediction Factors")
            
            factors_df = pd.DataFrame({
                'Factor': ['Home Field Advantage', 'Team Strength', 'Hot/Cold Players', 'Starting Pitchers'],
                'Impact': [home_adv, team_factor, hot_factor, pitcher_factor],
                'Description': [
                    'Advantage for the home team',
                    'Difference in team performance metrics',
                    'Impact of hot/cold players on each team',
                    'Starting pitcher matchup impact'
                ]
            })
            
            # Format impact column
            factors_df['Impact'] = factors_df['Impact'].map(lambda x: f"{x:+.3f}")
            
            st.dataframe(factors_df, use_container_width=True)
            
            # Disclaimer
            st.info("Note: This is a simplified prediction model for demonstration purposes. A production model would incorporate many more factors and use more sophisticated algorithms.")

            # Future enhancements section
        st.subheader("Player Projection System")
        st.write("The Player Projection System allows you to predict future performance based on current trends and metrics.")
        
        # Simplified player projection interface
        player_name = st.text_input("Player Name")
        projection_period = st.selectbox("Projection Period", ["Next Game", "Next Week", "Next Month", "Rest of Season"])
        
        if player_name and st.button("Generate Player Projection"):
            st.info(f"Generating {projection_period.lower()} projection for {player_name}...")
            
            # Find the player in our dataset
            player_row = None
            if player_name in data['Name'].values:
                player_row = data[data['Name'] == player_name]
            
            if player_row is not None and not player_row.empty:
                # Get player info
                player_team = player_row['Team'].values[0] if 'Team' in player_row.columns else None
                team_color = TEAMS.get(player_team, {}).get('color', '#1E88E5') if player_team else '#1E88E5'
                
                # Generate projections
                # This is a simplified simulation - a real model would use historical patterns and actual performance data
                
                if player_type == "Batting":
                    # Base metrics
                    base_avg = player_row['AVG'].values[0] if 'AVG' in player_row.columns else 0.250
                    base_obp = player_row['OBP'].values[0] if 'OBP' in player_row.columns else 0.320
                    base_slg = player_row['SLG'].values[0] if 'SLG' in player_row.columns else 0.450
                    base_hr = player_row['HR'].values[0] if 'HR' in player_row.columns else 10
                    
                    # Factor in hot/cold status if available
                    performance_factor = 0
                    if 'hot_cold_status' in player_row.columns:
                        status = player_row['hot_cold_status'].values[0]
                        if status == 'Hot':
                            performance_factor = 0.02
                        elif status == 'Cold':
                            performance_factor = -0.02
                    
                    # Generate projections for different time periods
                    if projection_period == "Next Game":
                        pa = 4  # Typical PAs in a game
                        hr_rate = (base_hr / 600) * (1 + performance_factor * 2)  # HR per PA
                        projected_hr = hr_rate * pa
                        projected_avg = base_avg + performance_factor
                        projected_obp = base_obp + performance_factor
                        projected_slg = base_slg + performance_factor * 1.5
                    elif projection_period == "Next Week":
                        pa = 25  # Typical weekly PAs
                        hr_rate = (base_hr / 600) * (1 + performance_factor)
                        projected_hr = hr_rate * pa
                        projected_avg = base_avg + performance_factor * 0.8
                        projected_obp = base_obp + performance_factor * 0.8
                        projected_slg = base_slg + performance_factor * 1.2
                    elif projection_period == "Next Month":
                        pa = 100  # Typical monthly PAs
                        hr_rate = (base_hr / 600) * (1 + performance_factor * 0.7)
                        projected_hr = hr_rate * pa
                        projected_avg = base_avg + performance_factor * 0.5
                        projected_obp = base_obp + performance_factor * 0.5
                        projected_slg = base_slg + performance_factor * 0.8
                    else:  # Rest of Season
                        pa = 250  # Typical remaining PAs
                        hr_rate = (base_hr / 600) * (1 + performance_factor * 0.3)
                        projected_hr = hr_rate * pa
                        projected_avg = base_avg + performance_factor * 0.2
                        projected_obp = base_obp + performance_factor * 0.2
                        projected_slg = base_slg + performance_factor * 0.3
                    
                    # Display projections
                    st.subheader(f"{player_name} - {projection_period} Projection")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("AVG", f"{projected_avg:.3f}", f"{performance_factor:.3f}")
                    with col2:
                        st.metric("OBP", f"{projected_obp:.3f}", f"{performance_factor:.3f}")
                    with col3:
                        st.metric("SLG", f"{projected_slg:.3f}", f"{performance_factor * 1.5:.3f}")
                    with col4:
                        st.metric("HR", f"{projected_hr:.1f}")
                    
                    # Add a projected OPS
                    projected_ops = projected_obp + projected_slg
                    
                    # Create projection chart
                    metrics = ["AVG", "OBP", "SLG"]
                    base_values = [base_avg, base_obp, base_slg]
                    projected_values = [projected_avg, projected_obp, projected_slg]
                    
                    # Create a comparison bar chart
                    fig = go.Figure()
                    
                    # Add season baseline
                    fig.add_trace(go.Bar(
                        x=metrics,
                        y=base_values,
                        name='Season Average',
                        marker_color='lightgray'
                    ))
                    
                    # Add projection
                    fig.add_trace(go.Bar(
                        x=metrics,
                        y=projected_values,
                        name=f'{projection_period} Projection',
                        marker_color=team_color
                    ))
                    
                    # Add reference lines for league averages
                    league_avgs = [0.250, 0.320, 0.410]  # MLB averages (would be real data in production)
                    for i, (metric, avg) in enumerate(zip(metrics, league_avgs)):
                        fig.add_shape(
                            type="line",
                            x0=i-0.4, x1=i+0.4,
                            y0=avg, y1=avg,
                            line=dict(color="black", width=2, dash="dash")
                        )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{player_name} - {projection_period} Projection",
                        xaxis_title="Metric",
                        yaxis_title="Value",
                        barmode='group',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display projection explanation
                    st.subheader("Projection Factors")
                    
                    # Factor explanations
                    factors_df = pd.DataFrame({
                        'Factor': ['Season Performance', 'Hot/Cold Status', 'Recent Trends', 'Regression to Mean'],
                        'Impact': ['High', 'Medium' if performance_factor != 0 else 'Low', 'Medium', 'Medium-High'],
                        'Description': [
                            f"Based on {player_name}'s {selected_season} statistics",
                            f"Player is currently {'hot' if performance_factor > 0 else 'cold' if performance_factor < 0 else 'neutral'}",
                            f"Recent performance trends are factored into the projection",
                            f"Longer projections regress more toward career norms"
                        ]
                    })
                    
                    st.dataframe(factors_df, use_container_width=True)
                    
                else:  # Pitching
                    # Base metrics
                    base_era = player_row['ERA'].values[0] if 'ERA' in player_row.columns else 4.00
                    base_whip = player_row['WHIP'].values[0] if 'WHIP' in player_row.columns else 1.30
                    base_k9 = player_row['K/9'].values[0] if 'K/9' in player_row.columns else 8.5
                    
                    # Factor in hot/cold status if available
                    performance_factor = 0
                    if 'hot_cold_status' in player_row.columns:
                        status = player_row['hot_cold_status'].values[0]
                        if status == 'Hot':
                            performance_factor = -0.25  # Lower ERA/WHIP is better
                        elif status == 'Cold':
                            performance_factor = 0.25
                    
                    # Generate projections for different time periods
                    if projection_period == "Next Game":
                        ip = 6  # Typical IP in a start
                        projected_era = base_era + performance_factor * 1.5
                        projected_whip = base_whip + performance_factor / 5
                        projected_k9 = base_k9 + (performance_factor * -2)  # Inverse relationship
                    elif projection_period == "Next Week":
                        ip = 12  # Typical weekly IP
                        projected_era = base_era + performance_factor
                        projected_whip = base_whip + performance_factor / 8
                        projected_k9 = base_k9 + (performance_factor * -1.5)
                    elif projection_period == "Next Month":
                        ip = 30  # Typical monthly IP
                        projected_era = base_era + performance_factor * 0.7
                        projected_whip = base_whip + performance_factor / 10
                        projected_k9 = base_k9 + (performance_factor * -1)
                    else:  # Rest of Season
                        ip = 75  # Typical remaining IP
                        projected_era = base_era + performance_factor * 0.3
                        projected_whip = base_whip + performance_factor / 15
                        projected_k9 = base_k9 + (performance_factor * -0.5)
                    
                    # Calculate projected strikeouts
                    projected_k = (projected_k9 / 9) * ip
                    
                    # Display projections
                    st.subheader(f"{player_name} - {projection_period} Projection")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("ERA", f"{projected_era:.2f}", f"{performance_factor:.2f}")
                    with col2:
                        st.metric("WHIP", f"{projected_whip:.2f}", f"{performance_factor / 8:.2f}")
                    with col3:
                        st.metric("K/9", f"{projected_k9:.1f}", f"{performance_factor * -1.5:.1f}")
                    with col4:
                        st.metric("IP", f"{ip:.1f}")
                    
                    # Create projection chart
                    metrics = ["ERA", "WHIP", "K/9"]
                    base_values = [base_era, base_whip, base_k9]
                    projected_values = [projected_era, projected_whip, projected_k9]
                    
                    # Need to normalize the values for visualization
                    # ERA and WHIP are better when lower, K/9 is better when higher
                    # Also, the scales are very different
                    norm_base = [
                        (base_era - 2) / 3,  # ERA normalized around 2-5 range
                        (base_whip - 1) / 0.5,  # WHIP normalized around 1-1.5 range
                        (base_k9 - 7) / 3  # K/9 normalized around 7-10 range
                    ]
                    
                    norm_proj = [
                        (projected_era - 2) / 3,
                        (projected_whip - 1) / 0.5,
                        (projected_k9 - 7) / 3
                    ]
                    
                    # Create a radar chart for pitching projections
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=[1-norm_base[0], 1-norm_base[1], norm_base[2]],  # Invert ERA/WHIP so higher is better
                        theta=metrics,
                        fill='toself',
                        name='Season Average',
                        line_color='lightgray'
                    ))
                    
                    fig.add_trace(go.Scatterpolar(
                        r=[1-norm_proj[0], 1-norm_proj[1], norm_proj[2]],  # Invert ERA/WHIP so higher is better
                        theta=metrics,
                        fill='toself',
                        name=f'{projection_period} Projection',
                        line_color=team_color
                    ))
                    
                    fig.update_layout(
                        title=f"{player_name} - {projection_period} Projection",
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display raw projection numbers
                    projection_df = pd.DataFrame({
                        'Metric': metrics + ['IP', 'Projected K'],
                        'Season Average': base_values + [None, None],
                        f'{projection_period} Projection': projected_values + [ip, projected_k]
                    })
                    
                    st.dataframe(projection_df, use_container_width=True)
                    
                    # Display projection explanation
                    st.subheader("Projection Factors")
                    
                    # Factor explanations
                    factors_df = pd.DataFrame({
                        'Factor': ['Season Performance', 'Hot/Cold Status', 'Recent Trends', 'Regression to Mean'],
                        'Impact': ['High', 'Medium' if performance_factor != 0 else 'Low', 'Medium', 'Medium-High'],
                        'Description': [
                            f"Based on {player_name}'s {selected_season} statistics",
                            f"Player is currently {'hot' if performance_factor < 0 else 'cold' if performance_factor > 0 else 'neutral'} (for pitchers, hot means better ERA/WHIP)",
                            f"Recent performance trends are factored into the projection",
                            f"Longer projections regress more toward career norms"
                        ]
                    })
                    
                    st.dataframe(factors_df, use_container_width=True)
            else:
                st.warning(f"Player '{player_name}' not found in the current dataset. Try a different name or filter settings.")
        
        # Add disclaimer about the projections
        st.info("Note: These projections are simplified examples. A production system would incorporate many more factors including matchups, ballpark effects, weather, historical performance, aging curves, and more.")

# Run the app
if __name__ == "__main__":
    main()
def calculate_rolling_performance(player_data, window_size=10):
    """Calculate rolling performance metrics from statcast data"""
    if player_data is None or player_data.empty:
        return None
    
    try:
        # Group by date if not already
        if 'game_date' in player_data.columns:
            # For batters
            if 'estimated_ba_using_speedangle' in player_data.columns:
                daily_stats = player_data.groupby('game_date').agg({
                    'estimated_ba_using_speedangle': 'mean',
                    'estimated_woba_using_speedangle': 'mean' if 'estimated_woba_using_speedangle' in player_data.columns else None,
                    'launch_speed': 'mean' if 'launch_speed' in player_data.columns else None,
                    'launch_angle': 'mean' if 'launch_angle' in player_data.columns else None
                }).reset_index()
            # For pitchers
            elif 'release_speed' in player_data.columns:
                daily_stats = player_data.groupby('game_date').agg({
                    'release_speed': 'mean',
                    'release_spin_rate': 'mean' if 'release_spin_rate' in player_data.columns else None,
                    'effective_speed': 'mean' if 'effective_speed' in player_data.columns else None
                }).reset_index()
            else:
                return None
            
            # Remove None columns
            daily_stats = daily_stats.loc[:, ~daily_stats.columns.isin([None])]
            
            if daily_stats.empty:
                return None
                
            # Calculate rolling statistics
            if len(daily_stats) >= 3:  # Need at least 3 points for rolling window
                # Apply rolling window to each numeric column
                for col in daily_stats.columns:
                    if col != 'game_date' and pd.api.types.is_numeric_dtype(daily_stats[col]):
                        daily_stats[f'rolling_{col}'] = daily_stats[col].rolling(
                            window=min(window_size, len(daily_stats)), 
                            min_periods=1
                        ).mean()
            
            return daily_stats
        
        return None
    except Exception as e:
        if show_debug_info:
            st.error(f"Error calculating rolling performance: {e}")
        return None

# Function to direct fetch data from Baseball Savant if needed
def fetch_from_baseball_savant(player_id, start_date, end_date):
    """Direct fetch from Baseball Savant as a backup method"""
    # Implementation would go here for a direct API call
    # This is a placeholder for future enhancement
    return None

# Main app layout
def main():
    # Tabs for different dashboard views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Season Overview", 
        "Player Deep Dive", 
        "Hot Hand Analysis",
        "Predictions"
    ])
    
    # Fetch data based on user selection
    if player_type == "Batting":
        data = fetch_batting_data(selected_season, selected_team)
    else:
        data = fetch_pitching_data(selected_season, selected_team)
    
    if data is None or data.empty:
        st.error("No data available. Please try different filter settings.")
        return
        
    # Add custom metrics
    data = add_custom_metrics(data)
    
    with tab1:
        st.header(f"{selected_season} Season Overview")
        
        # Basic league-wide stats
        if selected_team == "All Teams":
            st.subheader("MLB League Overview")
            
            # Team summary metrics
            if 'Team' in data.columns:
                # Create team-level stats
                team_summary = data.groupby('Team').agg({
                    'Name': 'count',
                    'HR': 'sum' if 'HR' in data.columns else None,
                    'R': 'sum' if 'R' in data.columns else None,
                    'AVG': lambda x: np.average(x, weights=data.loc[x.index, 'PA']) if 'AVG' in data.columns and 'PA' in data.columns else None,
                    'OBP': lambda x: np.average(x, weights=data.loc[x.index, 'PA']) if 'OBP' in data.columns and 'PA' in data.columns else None,
                    'SLG': lambda x: np.average(x, weights=data.loc[x.index, 'PA']) if 'SLG' in data.columns and 'PA' in data.columns else None,
                    'ERA': lambda x: np.average(x, weights=data.loc[x.index, 'IP']) if 'ERA' in data.columns and 'IP' in data.columns else None,
                    'WHIP': lambda x: np.average(x, weights=data.loc[x.index, 'IP']) if 'WHIP' in data.columns and 'IP' in data.columns else None,
                    'SO': 'sum' if 'SO' in data.columns else None,
                    'W': 'sum' if 'W' in data.columns else None,
                    'SV': 'sum' if 'SV' in data.columns else None,
                }).reset_index()
                
                # Remove None columns
                team_summary = team_summary.loc[:, ~team_summary.columns.isin([None])]
                
                # Rename columns
                team_summary.rename(columns={'Name': 'Players'}, inplace=True)
                
                # Display team summary
                st.subheader("Team Summary")
                
                # Format the dataframe for display
                display_cols = [col for col in team_summary.columns if col != 'Team']
                
                # Add team colors
                if player_type == "Batting":
                    # Set up metrics for team comparison viz
                    if 'HR' in team_summary.columns:
                        metric_col = 'HR'
                        metric_name = 'Home Runs'
                    elif 'R' in team_summary.columns:
                        metric_col = 'R'
                        metric_name = 'Runs'
                    elif 'AVG' in team_summary.columns:
                        metric_col = 'AVG'
                        metric_name = 'Batting Average'
                    else:
                        metric_col = None
                else:  # Pitching
                    if 'ERA' in team_summary.columns:
                        metric_col = 'ERA'
                        metric_name = 'ERA'
                    elif 'WHIP' in team_summary.columns:
                        metric_col = 'WHIP'
                        metric_name = 'WHIP'
                    elif 'SO' in team_summary.columns:
                        metric_col = 'SO'
                        metric_name = 'Strikeouts'
                    else:
                        metric_col = None
                
                # Create a bar chart of team metrics
                if metric_col is not None:
                    # Sort data
                    if metric_col in ['ERA', 'WHIP']:
                        # Lower is better for these metrics
                        sorted_data = team_summary.sort_values(metric_col, ascending=True)
                    else:
                        # Higher is better for these metrics
                        sorted_data = team_summary.sort_values(metric_col, ascending=False)
                    
                    # Get team colors for visualization
                    team_colors = []
                    for team in sorted_data['Team']:
                        if team in TEAMS:
                            team_colors.append(TEAMS[team]['color'])
                        else:
                            team_colors.append('#CCCCCC')  # Default gray
                    
                    fig = px.bar(
                        sorted_data,
                        x='Team',
                        y=metric_col,
                        title=f"Team {metric_name} Comparison",
                        color='Team',
                        color_discrete_map=dict(zip(sorted_data['Team'], team_colors))
                    )
                    
                    # Adjust layout
                    fig.update_layout(
                        xaxis_title="Team",
                        yaxis_title=metric_name,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display the raw data table
                st.dataframe(team_summary, use_container_width=True)
        
        # Team-specific stats
        else:
            team_color = TEAMS.get(selected_team, {}).get('color', '#1E88E5')
            team_name = TEAMS.get(selected_team, {}).get('name', selected_team)
            
            st.subheader(f"{team_name} Team Overview")
            
            # Display team metrics on top
            if player_type == "Batting":
                metrics = {}
                if 'AVG' in data.columns and 'PA' in data.columns:
                    metrics['Batting Average'] = np.average(data['AVG'], weights=data['PA'])
                if 'HR' in data.columns:
                    metrics['Home Runs'] = data['HR'].sum()
                if 'R' in data.columns:
                    metrics['Runs'] = data['R'].sum()
                if 'OPS' in data.columns and 'PA' in data.columns:
                    metrics['OPS'] = np.average(data['OPS'], weights=data['PA'])
            else:  # Pitching
                metrics = {}
                if 'ERA' in data.columns and 'IP' in data.columns:
                    metrics['ERA'] = np.average(data['ERA'], weights=data['IP'])
                if 'WHIP' in data.columns and 'IP' in data.columns:
                    metrics['WHIP'] = np.average(data['WHIP'], weights=data['IP'])
                if 'SO' in data.columns:
                    metrics['Strikeouts'] = data['SO'].sum()
                if 'W' in data.columns:
                    metrics['Wins'] = data['W'].sum()
            
            # Display team metrics in columns
            if metrics:
                cols = st.columns(len(metrics))
                for i, (metric_name, metric_value) in enumerate(metrics.items()):
                    with cols[i]:
                        if isinstance(metric_value, float):
                            st.metric(metric_name, f"{metric_value:.3f}")
                        else:
                            st.metric(metric_name, metric_value)
            
            # Display player stats
            if player_type == "Batting":
                st.subheader(f"{team_name} Batting Leaders")
                
                # Common batting stats to display
                display_columns = ['Name', 'PA', 'AVG', 'OBP', 'SLG', 'OPS', 'wOBA', 'wRC+', 'HR']
                
                # Make sure all required columns exist
                valid_columns = [col for col in display_columns if col in data.columns]
                
                # Sort data
                if 'wOBA' in data.columns:
                    sorted_data = data.sort_values('wOBA', ascending=False)
                elif 'OPS' in data.columns:
                    sorted_data = data.sort_values('OPS', ascending=False)
                else:
                    sorted_data = data.sort_values('AVG', ascending=False)
                
                # Display the data table
                st.dataframe(sorted_data[valid_columns], use_container_width=True)
                
                # Create visualizations for team batting stats
                if 'HR' in data.columns:
                    # Home Run Leaders
                    hr_leaders = data.sort_values('HR', ascending=False).head(10)
                    
                    fig = px.bar(
                        hr_leaders,
                        x='Name',
                        y='HR',
                        title=f"{team_name} Home Run Leaders",
                        color_discrete_sequence=[team_color]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                if 'AVG' in data.columns:
                    # Batting Average Leaders (min 100 PA)
                    if 'PA' in data.columns:
                        avg_leaders = data[data['PA'] >= 100].sort_values('AVG', ascending=False).head(10)
                    else:
                        avg_leaders = data.sort_values('AVG', ascending=False).head(10)
                    
                    fig = px.bar(
                        avg_leaders,
                        x='Name',
                        y='AVG',
                        title=f"{team_name} Batting Average Leaders",
                        color_discrete_sequence=[team_color]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            else:  # Pitching
                st.subheader(f"{team_name} Pitching Leaders")
                
                # Common pitching stats to display
                display_columns = ['Name', 'W', 'L', 'ERA', 'G', 'IP', 'SO', 'WHIP']
                
                # Make sure all required columns exist
                valid_columns = [col for col in display_columns if col in data.columns]
                
                # Sort data
                if 'ERA' in data.columns and 'IP' in data.columns:
                    # Filter for minimum innings pitched
                    qualified = data[data['IP'] >= 10]
                    sorted_data = qualified.sort_values('ERA', ascending=True)
                else:
                    sorted_data = data.sort_values('IP', ascending=False)
                
                # Display the data table
                st.dataframe(sorted_data[valid_columns], use_container_width=True)
                
                # Create visualizations for team pitching stats
                if 'ERA' in data.columns and 'IP' in data.columns:
                    # ERA Leaders (min 20 IP for starters)
                    era_leaders = data[data['IP'] >= 20].sort_values('ERA', ascending=True).head(10)
                    
                    fig = px.bar(
                        era_leaders,
                        x='Name',
                        y='ERA',
                        title=f"{team_name} ERA Leaders (min 20 IP)",
                        color_discrete_sequence=[team_color]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                if 'SO' in data.columns:
                    # Strikeout Leaders
                    k_leaders = data.sort_values('SO', ascending=False).head(10)
                    
                    fig = px.bar(
                        k_leaders,
                        x='Name',
                        y='SO',
                        title=f"{team_name} Strikeout Leaders",
                        color_discrete_sequence=[team_color]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Display hot/cold status distribution across league or team
        if 'hot_cold_status' in data.columns:
            st.subheader("Hot/Cold Players Distribution")
            
            hot_count = (data['hot_cold_status'] == 'Hot').sum()
            neutral_count = (data['hot_cold_status'] == 'Neutral').sum()
            cold_count = (data['hot_cold_status'] == 'Cold').sum()
            
            fig = px.pie(
                values=[hot_count, neutral_count, cold_count],
                names=['Hot', 'Neutral', 'Cold'],
                title=f"Player Hot/Cold Distribution ({player_type})",
                color=['Hot', 'Neutral', 'Cold'],
                color_discrete_map={'Hot': 'red', 'Neutral': 'gray', 'Cold': 'blue'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Player Deep Dive")
        
        # Select player for deep dive
        if not data.empty:
            player_list = data['Name'].tolist()
            selected_player = st.selectbox("Select Player for Analysis", player_list)
            
            if selected_player:
                st.subheader(f"Analysis for {selected_player}")
                
                # Get player info
                player_row = data[data['Name'] == selected_player]
                player_team = player_row['Team'].values[0] if 'Team' in player_row.columns else selected_team
                team_color = TEAMS.get(player_team, {}).get('color', '#1E88E5')
                
                # Display basic player statistics
                st.subheader("Season Statistics")
                
                # Create columns for key metrics
                if player_type == "Batting":
                    # Define the metrics to show
                    metrics = []
                    if 'AVG' in player_row.columns:
                        metrics.append(("AVG", player_row['AVG'].values[0], None))
                    if 'OBP' in player_row.columns:
                        metrics.append(("OBP", player_row['OBP'].values[0], None))
                    if 'SLG' in player_row.columns:
                        metrics.append(("SLG", player_row['SLG'].values[0], None))
                    if 'HR' in player_row.columns:
                        metrics.append(("HR", player_row['HR'].values[0], None))
                    if 'RBI' in player_row.columns:
                        metrics.append(("RBI", player_row['RBI'].values[0], None))
                    if 'wOBA' in player_row.columns:
                        metrics.append(("wOBA", player_row['wOBA'].values[0], None))
                    
                    # Add hot/cold indicator if available
                    if 'hot_cold_status' in player_row.columns:
                        hot_status = player_row['hot_cold_status'].values[0]
                        if 'wOBA_diff' in player_row.columns:
                            woba_diff = player_row['wOBA_diff'].values[0]
                            metrics.append(("Hot/Cold", hot_status, f"{woba_diff:.3f}" if woba_diff is not None else None))
                    
                else:  # Pitching
                    # Define the metrics to show
                    metrics = []
                    if 'ERA' in player_row.columns:
                        metrics.append(("ERA", player_row['ERA'].values[0], None))
                    if 'WHIP' in player_row.columns:
                        metrics.append(("WHIP", player_row['WHIP'].values[0], None))
                    if 'W' in player_row.columns and 'L' in player_row.columns:
                        w = player_row['W'].values[0]
                        l = player_row['L'].values[0]
                        metrics.append(("W-L", f"{w}-{l}", None))
                    if 'SO' in player_row.columns:
                        metrics.append(("SO", player_row['SO'].values[0], None))
                    if 'IP' in player_row.columns:
                        metrics.append(("IP", player_row['IP'].values[0], None))
                    if 'K/9' in player_row.columns:
                        metrics.append(("K/9", player_row['K/9'].values[0], None))
                
                # Display metrics in columns
                if metrics:
                    # Calculate number of columns needed (up to 4 per row)
                    num_metrics = len(metrics)
                    num_rows = (num_metrics + 3) // 4  # Ceiling division
                    
                    for row in range(num_rows):
                        cols = st.columns(min(4, num_metrics - row * 4))
                        for i, (name, value, delta) in enumerate(metrics[row * 4:min((row + 1) * 4, num_metrics)]):
                            with cols[i]:
                                if isinstance(value, float):
                                    st.metric(name, f"{value:.3f}", delta)
                                else:
                                    st.metric(name, value, delta)
                
                # Attempt to get Statcast data
                today = datetime.now().strftime('%Y-%m-%d')
                thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                
                player_data = fetch_player_statcast(
                    selected_player, 
                    thirty_days_ago, 
                    today,
                    player_type.lower()
                )
                
                if player_data is not None and not player_data.empty:
                    st.success(f"Successfully retrieved Statcast data for {selected_player}")
                    
                    # Process Statcast data for visualization
                    if player_type == "Batting":
                        st.subheader("Batting Analysis")
                        
                        # Exit Velocity & Launch Angle
                        if 'launch_speed' in player_data.columns and 'launch_angle' in player_data.columns:
                            # Filter out null values
                            ev_data = player_data.dropna(subset=['launch_speed', 'launch_angle'])
                            
                            if not ev_data.empty:
                                fig = px.scatter(
                                    ev_data,
                                    x='launch_speed',
                                    y='launch_angle',
                                    color='events',
                                    hover_name='events',
                                    color_discrete_map={
                                        'home_run': team_color,
                                        'double': '#4CAF50',
                                        'single': '#FFC107',
                                        'triple': '#9C27B0',
                                        'field_out': '#9E9E9E',
                                        'strikeout': '#F44336'
                                    },
                                    title=f"{selected_player} - Exit Velocity & Launch Angle"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No valid exit velocity and launch angle data available.")
                        
                        # Calculate rolling performance
                        rolling_data = calculate_rolling_performance(player_data, rolling_window)
                        
                        if rolling_data is not None:
                            # Plot rolling metrics
                            st.subheader("Performance Trends")
                            
                            # Expected stats trends
                            if 'estimated_ba_using_speedangle' in rolling_data.columns and 'rolling_estimated_ba_using_speedangle' in rolling_data.columns:
                                fig = px.line(
                                    rolling_data,
                                    x='game_date',
                                    y=['estimated_ba_using_speedangle', 'rolling_estimated_ba_using_speedangle'],
                                    title=f"{selected_player} - Rolling Expected Batting Average ({rolling_window}-game window)",
                                    labels={'value': 'Expected Batting Average', 'variable': 'Metric'},
                                    color_discrete_map={
                                        'estimated_ba_using_speedangle': '#9E9E9E',
                                        'rolling_estimated_ba_using_speedangle': team_color
                                    }
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Exit velocity trends
                            if 'launch_speed' in rolling_data.columns and 'rolling_launch_speed' in rolling_data.columns:
                                fig = px.line(
                                    rolling_data,
                                    x='game_date',
                                    y=['launch_speed', 'rolling_launch_speed'],
                                    title=f"{selected_player} - Rolling Exit Velocity ({rolling_window}-game window)",
                                    labels={'value': 'Exit Velocity (mph)', 'variable': 'Metric'},
                                    color_discrete_map={
                                        'launch_speed': '#9E9E9E',
                                        'rolling_launch_speed': team_color
                                    }
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    
                    else:  # Pitching
                        st.subheader("Pitching Analysis")
                        
                        # Velocity analysis
                        if 'release_speed' in player_data.columns:
                            # Group by pitch type
                            if 'pitch_type' in player_data.columns:
                                pitch_velo = player_data.groupby('pitch_type')['release_speed'].agg(['mean', 'std', 'count']).reset_index()
                                pitch_velo = pitch_velo[pitch_velo['count'] >= 5]  # Filter for pitch types with enough samples
                                
                                if not pitch_velo.empty:
                                    fig = px.bar(
                                        pitch_velo,
                                        x='pitch_type',
                                        y='mean',
                                        error_y='std',
                                        title=f"{selected_player} - Average Velocity by Pitch Type",
                                        labels={'mean': 'Velocity (mph)', 'pitch_type': 'Pitch Type'},
                                        color_discrete_sequence=[team_color]
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # Calculate rolling performance
                            rolling_data = calculate_rolling_performance(player_data, rolling_window)
                            
                            if rolling_data is not None:
                                # Plot rolling metrics
                                st.subheader("Performance Trends")
                                
                                # Velocity trends
                                if 'release_speed' in rolling_data.columns and 'rolling_release_speed' in rolling_data.columns:
                                    fig = px.line(
                                        rolling_data,
                                        x='game_date',
                                        y=['release_speed', 'rolling_release_speed'],
                                        title=f"{selected_player} - Rolling Velocity ({rolling_window}-game window)",
                                        labels={'value': 'Velocity (mph)', 'variable': 'Metric'},
                                        color_discrete_map={
                                            'release_speed': '#9E9E9E',
                                            'rolling_release_speed': team_color
                                        }
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display raw data 
                    st.subheader("Recent Statcast Data")
                    
                    # Select columns based on player type
                    if player_type == "Batting":
                        display_cols = ['game_date', 'events', 'launch_speed', 'launch_angle']
                        if 'estimated_ba_using_speedangle' in player_data.columns:
                            display_cols.append('estimated_ba_using_speedangle')
                    else:  # Pitching
                        display_cols = ['game_date', 'events', 'pitch_type', 'release_speed']
                        if 'release_spin_rate' in player_data.columns:
                            display_cols.append('release_spin_rate')
                    
                    valid_display_cols = [col for col in display_cols if col in player_data.columns]
                    
                    # Only show a sample of the data
                    st.dataframe(player_data[valid_display_cols].tail(10))
                    
                    # Option to download full data
                    csv = player_data[valid_display_cols].to_csv(index=False)
                    st.download_button(
                        label="Download Complete Statcast Data",
                        data=csv,
                        file_name=f"{selected_player}_statcast_data.csv",
                        mime="text/csv"
                    )
                
                else:
                    st.info("Using simulated performance data since Statcast data is limited.")
                    
                    # Create simulated monthly data
                    if 'AVG' in player_row.columns or player_type == "Batting":
                        # Batting simulation
                        months = ["Apr", "May", "Jun", "Jul", "Aug", "Sep"]
                        stats = {}
                        
                        # Base metrics
                        base_avg = player_row['AVG'].values[0] if 'AVG' in player_row.columns else 0.250
                        base_obp = player_row['OBP'].values[0] if 'OBP' in player_row.columns else 0.320
                        base_slg = player_row['SLG'].values[0] if 'SLG' in player_row.columns else 0.400
                        
                        # Create random variations around the base value
                        stats['AVG'] = np.clip(base_avg + np.random.normal(0, 0.025, len(months)), 0.150, 0.400)
                        stats['OBP'] = np.clip(base_obp + np.random.normal(0, 0.030, len(months)), 0.200, 0.500)
                        stats['SLG'] = np.clip(base_slg + np.random.normal(0, 0.050, len(months)), 0.250, 0.700)
                        stats['PA'] = np.random.randint(50, 120, len(months))
                        stats['HR'] = np.random.randint(0, 10, len(months))
                        
                        # Create player data
                        player_monthly_data = pd.DataFrame({
                            'Month': months,
                            **stats
                        })
                        
                        # Compute OPS
                        player_monthly_data['OPS'] = player_monthly_data['OBP'] + player_monthly_data['SLG']
                        
                        # Plot monthly batting average
                        st.subheader("Monthly Performance (Simulated)")
                        fig = px.line(
                            player_monthly_data,
                            x='Month',
                            y='AVG',
                            title=f"{selected_player} - Monthly Batting Average",
                            markers=True,
                            line_shape='spline',
                            color_discrete_sequence=[team_color]
                        )
                        
                        # Add season average reference line
                        fig.add_hline(
                            y=base_avg,
                            line_dash="dash",
                            line_color="#000000",
                            annotation_text=f"Season AVG: {base_avg:.3f}",
                            annotation_position="bottom right"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Plot monthly OPS
                        fig = px.line(
                            player_monthly_data,
                            x='Month',
                            y='OPS',
                            title=f"{selected_player} - Monthly OPS",
                            markers=True,
                            line_shape='spline',
                            color_discrete_sequence=[team_color]
                        )
                        
                        # Add season average reference line
                        base_ops = base_obp + base_slg
                        fig.add_hline(
                            y=base_ops,
                            line_dash="dash",
                            line_color="#000000",
                            annotation_text=f"Season OPS: {base_ops:.3f}",
                            annotation_position="bottom right"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Plot monthly home runs
                        
                        fig = px.bar(
                            player_monthly_data,
                            x='Month',
                            y='HR',
                            title=f"{selected_player} - Monthly Home Runs",
                            color_discrete_sequence=[team_color]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display monthly stats as a table
                        st.subheader("Monthly Breakdown (Simulated)")
                        st.dataframe(player_monthly_data)
                        
                    elif player_type == "Pitching":
                        # Pitching simulation
                        months = ["Apr", "May", "Jun", "Jul", "Aug", "Sep"]
                        stats = {}
                        
                        # Base metrics
                        base_era = player_row['ERA'].values[0] if 'ERA' in player_row.columns else 4.00
                        base_whip = player_row['WHIP'].values[0] if 'WHIP' in player_row.columns else 1.30
                        base_k9 = player_row['K/9'].values[0] if 'K/9' in player_row.columns else 8.5
                        
                        # Create random variations around the base value
                        stats['ERA'] = np.clip(base_era + np.random.normal(0, 0.50, len(months)), 1.50, 7.50)
                        stats['WHIP'] = np.clip(base_whip + np.random.normal(0, 0.15, len(months)), 0.80, 2.00)
                        stats['K/9'] = np.clip(base_k9 + np.random.normal(0, 0.75, len(months)), 5.0, 14.0)
                        stats['IP'] = np.random.uniform(10, 35, len(months))
                        stats['SO'] = np.random.randint(10, 40, len(months))
                        
                        # Create player data
                        player_monthly_data = pd.DataFrame({
                            'Month': months,
                            **stats
                        })
                        
                        # Plot monthly ERA
                        st.subheader("Monthly Performance (Simulated)")
                        fig = px.line(
                            player_monthly_data,
                            x='Month',
                            y='ERA',
                            title=f"{selected_player} - Monthly ERA",
                            markers=True,
                            line_shape='spline',
                            color_discrete_sequence=[team_color]
                        )
                        
                        # Add season average reference line
                        fig.add_hline(
                            y=base_era,
                            line_dash="dash",
                            line_color="#000000",
                            annotation_text=f"Season ERA: {base_era:.2f}",
                            annotation_position="top right"
                        )
                        
                        # Lower y-axis is better for ERA
                        fig.update_yaxes(autorange="reversed")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Plot monthly K/9
                        fig = px.line(
                            player_monthly_data,
                            x='Month',
                            y='K/9',
                            title=f"{selected_player} - Monthly K/9",
                            markers=True,
                            line_shape='spline',
                            color_discrete_sequence=[team_color]
                        )
                        
                        # Add season average reference line
                        fig.add_hline(
                            y=base_k9,
                            line_dash="dash",
                            line_color="#000000",
                            annotation_text=f"Season K/9: {base_k9:.2f}",
                            annotation_position="bottom right"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Plot monthly strikeouts
                        fig = px.bar(
                            player_monthly_data,
                            x='Month',
                            y='SO',
                            title=f"{selected_player} - Monthly Strikeouts",
                            color_discrete_sequence=[team_color]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display monthly stats as a table
                        st.subheader("Monthly Breakdown (Simulated)")
                        st.dataframe(player_monthly_data)
    
    with tab3:
        st.header("Hot Hand Analysis")
        
        if 'hot_cold_status' in data.columns:
            # Filter players by hot/cold status
            hot_players = data[data['hot_cold_status'] == 'Hot'].sort_values(
                'wOBA_diff' if 'wOBA_diff' in data.columns else 'OPS_diff' if 'OPS_diff' in data.columns else 'avg_diff',
                ascending=False
            )
            
            cold_players = data[data['hot_cold_status'] == 'Cold'].sort_values(
                'wOBA_diff' if 'wOBA_diff' in data.columns else 'OPS_diff' if 'OPS_diff' in data.columns else 'avg_diff',
                ascending=True
            )
            
            # Display hot players
            st.subheader("🔥 Hot Players")
            if not hot_players.empty:
                # Select columns for display, ensuring they exist
                if player_type == "Batting":
                    display_cols = ['Name', 'Team']
                    
                    # Add appropriate performance columns
                    if 'wOBA' in hot_players.columns and 'recent_wOBA' in hot_players.columns:
                        display_cols.extend(['wOBA', 'recent_wOBA', 'wOBA_diff'])
                    elif 'OPS' in hot_players.columns and 'recent_OPS' in hot_players.columns:
                        display_cols.extend(['OPS', 'recent_OPS', 'OPS_diff'])
                    elif 'AVG' in hot_players.columns:
                        display_cols.extend(['AVG', 'first_half_avg', 'second_half_avg', 'avg_diff'])
                else:  # Pitching
                    display_cols = ['Name', 'Team']
                    
                    # Add appropriate performance columns
                    if 'ERA' in hot_players.columns:
                        display_cols.append('ERA')
                    if 'WHIP' in hot_players.columns:
                        display_cols.append('WHIP')
                    if 'K/9' in hot_players.columns:
                        display_cols.append('K/9')
                    if 'avg_diff' in hot_players.columns:
                        display_cols.append('avg_diff')
                
                # Make sure all required columns exist
                valid_cols = [col for col in display_cols if col in hot_players.columns]
                
                hot_df = hot_players[valid_cols].copy()
                
                # Rename columns for display if needed
                column_mapping = {
                    'wOBA': 'Season wOBA', 
                    'recent_wOBA': 'Recent wOBA', 
                    'wOBA_diff': 'Difference',
                    'OPS': 'Season OPS',
                    'recent_OPS': 'Recent OPS', 
                    'OPS_diff': 'Difference',
                    'first_half_avg': 'First Half AVG',
                    'second_half_avg': 'Second Half AVG',
                    'avg_diff': 'Difference'
                }
                
                # Only rename columns that exist
                rename_cols = {k: v for k, v in column_mapping.items() if k in hot_df.columns}
                hot_df.rename(columns=rename_cols, inplace=True)
                
                # Format difference columns if they exist
                diff_cols = ['Difference']
                for col in diff_cols:
                    if col in hot_df.columns:
                        hot_df[col] = hot_df[col].map(lambda x: f"{x:+.3f}" if isinstance(x, (int, float)) else x)
                
                # Display the dataframe
                st.dataframe(hot_df, use_container_width=True)
                
                # Create visualization of hot players
                vis_metric = None
                if 'wOBA_diff' in hot_players.columns:
                    vis_metric = 'wOBA_diff'
                    metric_name = 'wOBA Difference'
                elif 'OPS_diff' in hot_players.columns:
                    vis_metric = 'OPS_diff'
                    metric_name = 'OPS Difference'
                elif 'avg_diff' in hot_players.columns:
                    vis_metric = 'avg_diff'
                    metric_name = 'AVG Difference'
                
                if vis_metric is not None:
                    # Use team colors for bars
                    team_colors = []
                    for team in hot_players.head(10)['Team']:
                        if team in TEAMS:
                            team_colors.append(TEAMS[team]['color'])
                        else:
                            team_colors.append('#CCCCCC')  # Default gray
                    
                    fig = px.bar(
                        hot_players.head(10),
                        x='Name',
                        y=vis_metric,
                        color='Team',
                        title=f"Hot Players ({metric_name})",
                        color_discrete_map=dict(zip(hot_players.head(10)['Team'], team_colors))
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title="Player",
                        yaxis_title=metric_name
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No hot players found based on current thresholds.")
            
            # Display cold players
            st.subheader("❄️ Cold Players")
            if not cold_players.empty:
                # Select columns for display, ensuring they exist
                if player_type == "Batting":
                    display_cols = ['Name', 'Team']
                    
                    # Add appropriate performance columns
                    if 'wOBA' in cold_players.columns and 'recent_wOBA' in cold_players.columns:
                        display_cols.extend(['wOBA', 'recent_wOBA', 'wOBA_diff'])
                    elif 'OPS' in cold_players.columns and 'recent_OPS' in cold_players.columns:
                        display_cols.extend(['OPS', 'recent_OPS', 'OPS_diff'])
                    elif 'AVG' in cold_players.columns:
                        display_cols.extend(['AVG', 'first_half_avg', 'second_half_avg', 'avg_diff'])
                else:  # Pitching
                    display_cols = ['Name', 'Team']
                    
                    # Add appropriate performance columns
                    if 'ERA' in cold_players.columns:
                        display_cols.append('ERA')
                    if 'WHIP' in cold_players.columns:
                        display_cols.append('WHIP')
                    if 'K/9' in cold_players.columns:
                        display_cols.append('K/9')
                    if 'avg_diff' in cold_players.columns:
                        display_cols.append('avg_diff')
                
                # Make sure all required columns exist
                valid_cols = [col for col in display_cols if col in cold_players.columns]
                
                cold_df = cold_players[valid_cols].copy()
                
                # Rename columns for display if needed
                column_mapping = {
                    'wOBA': 'Season wOBA', 
                    'recent_wOBA': 'Recent wOBA', 
                    'wOBA_diff': 'Difference',
                    'OPS': 'Season OPS',
                    'recent_OPS': 'Recent OPS', 
                    'OPS_diff': 'Difference',
                    'first_half_avg': 'First Half AVG',
                    'second_half_avg': 'Second Half AVG',
                    'avg_diff': 'Difference'
                }
                
                # Only rename columns that exist
                rename_cols = {k: v for k, v in column_mapping.items() if k in cold_df.columns}
                cold_df.rename(columns=rename_cols, inplace=True)
                
                # Format difference columns if they exist
                diff_cols = ['Difference']
                for col in diff_cols:
                    if col in cold_df.columns:
                        cold_df[col] = cold_df[col].map(lambda x: f"{x:+.3f}" if isinstance(x, (int, float)) else x)
                
                # Display the dataframe
                st.dataframe(cold_df, use_container_width=True)
                
                # Create visualization of cold players
                vis_metric = None
                if 'wOBA_diff' in cold_players.columns:
                    vis_metric = 'wOBA_diff'
                    metric_name = 'wOBA Difference'
                elif 'OPS_diff' in cold_players.columns:
                    vis_metric = 'OPS_diff'
                    metric_name = 'OPS Difference'
                elif 'avg_diff' in cold_players.columns:
                    vis_metric = 'avg_diff'
                    metric_name = 'AVG Difference'
                
                if vis_metric is not None:
                    # Use team colors for bars
                    team_colors = []
                    for team in cold_players.head(10)['Team']:
                        if team in TEAMS:
                            team_colors.append(TEAMS[team]['color'])
                        else:
                            team_colors.append('#CCCCCC')  # Default gray
                    
                    fig = px.bar(
                        cold_players.head(10),
                        x='Name',
                        y=vis_metric,
                        color='Team',
                        title=f"Cold Players ({metric_name})",
                        color_discrete_map=dict(zip(cold_players.head(10)['Team'], team_colors))
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title="Player",
                        yaxis_title=metric_name
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No cold players found based on current thresholds.")
            
            # Hot hand distribution
            vis_metric = None
            if 'wOBA_diff' in data.columns:
                vis_metric = 'wOBA_diff'
                metric_name = 'wOBA Difference'
            elif 'OPS_diff' in data.columns:
                vis_metric = 'OPS_diff'
                metric_name = 'OPS Difference'
            elif 'avg_diff' in data.columns:
                vis_metric = 'avg_diff'
                metric_name = 'AVG Difference'
            
            if vis_metric is not None:
                st.subheader("Hot Hand Distribution")
                fig = px.histogram(
                    data,
                    x=vis_metric,
                    color='hot_cold_status',
                    color_discrete_map={'Hot': 'red', 'Neutral': 'gray', 'Cold': 'blue'},
                    nbins=30,
                    title=f"Distribution of {metric_name}"
                )
                fig.add_vline(x=hot_threshold, line_dash="dash", line_color="red")
                fig.add_vline(x=cold_threshold, line_dash="dash", line_color="blue")
                st.plotly_chart(fig, use_container_width=True)
                
                # Custom hot hand insights
                st.subheader("Hot Hand Insights")
                
                # Additional hot hand metrics
                if 'consistency_score' in data.columns and 'streak_potential' in data.columns:
                    st.write("Players most likely to get hot (high streak potential, not currently hot):")
                    streak_candidates = data[
                        (data['hot_cold_status'] != 'Hot') & 
                        (data['streak_potential'] > 0.7)
                    ].sort_values('streak_potential', ascending=False).head(5)
                    
                    if not streak_candidates.empty:
                        st.dataframe(streak_candidates[['Name', 'Team', 'streak_potential', 'consistency_score']])
                    
                    st.write("Most consistent performers (low consistency score, regardless of hot/cold):")
                    consistent_players = data.sort_values('consistency_score').head(5)
                    
                    if not consistent_players.empty:
                        st.dataframe(consistent_players[['Name', 'Team', 'consistency_score', 'hot_cold_status']])
                
                # Correlation with other stats
                st.subheader("Hot Hand Correlations")
                
                # Choose a performance metric to correlate with hot hand
                perf_metric = None
                if player_type == "Batting":
                    if 'HR' in data.columns:
                        perf_metric = 'HR'
                        perf_name = 'Home Runs'
                    elif 'RBI' in data.columns:
                        perf_metric = 'RBI'
                        perf_name = 'RBIs'
                    elif 'AVG' in data.columns:
                        perf_metric = 'AVG'
                        perf_name = 'Batting Average'
                else:  # Pitching
                    if 'ERA' in data.columns:
                        perf_metric = 'ERA'
                        perf_name = 'ERA'
                    elif 'WHIP' in data.columns:
                        perf_metric = 'WHIP'
                        perf_name = 'WHIP'
                    elif 'K/9' in data.columns:
                        perf_metric = 'K/9'
                        perf_name = 'K/9'
                
                if perf_metric is not None and vis_metric is not None:
                    fig = px.scatter(
                        data,
                        x=vis_metric,
                        y=perf_metric,
                        color='hot_cold_status',
                        color_discrete_map={'Hot': 'red', 'Neutral': 'gray', 'Cold': 'blue'},
                        title=f"{metric_name} vs {perf_name}",
                        hover_data=['Name', 'Team']
                    )
                    
                    # Add vertical lines for thresholds
                    fig.add_vline(x=hot_threshold, line_dash="dash", line_color="red")
                    fig.add_vline(x=cold_threshold, line_dash="dash", line_color="blue")
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Hot hand analysis requires performance difference data which is not available in the current dataset.")
    
    with tab4:
        st.header("Predictions")
        
        st.subheader("Game Outcome Prediction")
        
        # Simplified game prediction interface
        col1, col2 = st.columns(2)
        
        with col1:
            home_team = st.selectbox("Home Team", list(TEAMS.keys()), index=0)
            home_starter = st.text_input("Home Starting Pitcher (optional)")
        
        with col2:
            away_team = st.selectbox("Away Team", list(TEAMS.keys()), index=1)
            away_starter = st.text_input("Away Starting Pitcher (optional)")
        
        if st.button("Generate Prediction"):
            st.info("Calculating prediction...")
            
            # Simulate a prediction model
            # In a real implementation, this would use actual team stats and models
            home_team_color = TEAMS.get(home_team, {}).get('color', '#1E88E5')
            away_team_color = TEAMS.get(away_team, {}).get('color', '#1E88E5')
            
            # Get team stats if available
            if 'Team' in data.columns:
                home_stats = data[data['Team'] == home_team]
                away_stats = data[data['Team'] == away_team]
            else:
                home_stats = pd.DataFrame()
                away_stats = pd.DataFrame()
            
            # Generate win probabilities
            # This is a simplified simulation - a real model would use actual team performance
            home_adv = 0.05  # Home field advantage
            
            # Factor in team offense/pitching strength
            team_factor = 0
            if player_type == "Batting" and not home_stats.empty and not away_stats.empty:
                if 'wOBA' in home_stats.columns and 'wOBA' in away_stats.columns:
                    home_woba = home_stats['wOBA'].mean()
                    away_woba = away_stats['wOBA'].mean()
                    team_factor = (home_woba - away_woba) * 2  # Scale factor
                elif 'OPS' in home_stats.columns and 'OPS' in away_stats.columns:
                    home_ops = home_stats['OPS'].mean()
                    away_ops = away_stats['OPS'].mean()
                    team_factor = (home_ops - away_ops)
            elif player_type == "Pitching" and not home_stats.empty and not away_stats.empty:
                if 'ERA' in home_stats.columns and 'ERA' in away_stats.columns:
                    home_era = home_stats['ERA'].mean()
                    away_era = away_stats['ERA'].mean()
                    team_factor = (away_era - home_era) / 5  # Scale factor, reversed since lower ERA is better
            
            # Factor in hot players
            hot_factor = 0
            if 'hot_cold_status' in data.columns:
                home_hot = data[(data['Team'] == home_team) & (data['hot_cold_status'] == 'Hot')].shape[0]
                home_cold = data[(data['Team'] == home_team) & (data['hot_cold_status'] == 'Cold')].shape[0]
                away_hot = data[(data['Team'] == away_team) & (data['hot_cold_status'] == 'Hot')].shape[0]
                away_cold = data[(data['Team'] == away_team) & (data['hot_cold_status'] == 'Cold')].shape[0]
                
                hot_factor = ((home_hot - home_cold) - (away_hot - away_cold)) * 0.01
            
            # Starting pitcher factor (placeholder, would use actual pitcher stats)
            pitcher_factor = 0
            if home_starter and away_starter:
                # This would use actual pitcher data in a real model
                pitcher_factor = np.random.uniform(-0.05, 0.05)
            
            # Combine factors for win probability
            home_win_prob = 0.5 + home_adv + team_factor + hot_factor + pitcher_factor
            
            # Clip to valid probability range
            home_win_prob = np.clip(home_win_prob, 0.05, 0.95)
            away_win_prob = 1 - home_win_prob
            
            # Display prediction
            st.subheader("Game Prediction")
            
            # Display win probabilities
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"{TEAMS.get(home_team, {}).get('name', home_team)}")
                st.markdown(f"<h1 style='text-align: center; color: {home_team_color};'>{home_win_prob:.1%}</h1>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;'>Win Probability</p>", unsafe_allow_html=True)
            
            with col2:
                st.subheader(f"{TEAMS.get(away_team, {}).get('name', away_team)}")
                st.markdown(f"<h1 style='text-align: center; color: {away_team_color};'>{away_win_prob:.1%}</h1>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;'>Win Probability</p>", unsafe_allow_html=True)
            
            # Display a gauge chart for the win probability
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = home_win_prob * 100,
                title = {'text': f"{home_team} vs {away_team}"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': home_team_color},
                    'steps': [
                        {'range': [0, 40], 'color': away_team_color},
                        {'range': [40, 60], 'color': "lightgray"},
                        {'range': [60, 100], 'color': home_team_color}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': home_win_prob * 100
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Factors influencing the prediction
            st.subheader("Prediction Factors")
            
            factors_df = pd.DataFrame({
                'Factor': ['Home Field Advantage', 'Team Strength', 'Hot/Cold Players', 'Starting Pitchers'],
                'Impact': [home_adv, team_factor, hot_factor, pitcher_factor],
                'Description': [
                    'Advantage for the home team',
                    'Difference in team performance metrics',
                    'Impact of hot/cold players on each team',
                    'Starting pitcher matchup impact'
                ]
            })
            
            # Format impact column
            factors_df['Impact'] = factors_df['Impact'].map(lambda x: f"{x:+.3f}")
            
            st.dataframe(factors_df, use_container_width=True)
            
            # Disclaimer
            st.info("Note: This is a simplified prediction model for demonstration purposes. A production model would incorporate many more factors and use more sophisticated algorithms.")
        