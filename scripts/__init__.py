import sys
import os

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

# Import modules from the parent or sibling directories
from scripts._load_data import load_data
from scripts._preprocessing import drop_unneeded_columns
from scripts._preprocessing import handle_missing_values
from scripts._preprocessing import handle_outliers
from scripts._plotting import plot_distribution
from scripts._plotting import plot_feature_distribution
from scripts._plotting import plot_feature_vs_target
from scripts._plotting import plot_missing_data
from scripts._plotting import plot_accidents_by_day_of_week
from scripts._plotting import plot_accidents_by_month
from scripts._plotting import plot_accidents_by_time
