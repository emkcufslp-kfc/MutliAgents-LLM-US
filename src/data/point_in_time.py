from datetime import datetime, date
from typing import Optional, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class PointInTimeContext:
    """
    Enforces the 'As-Of-Date' rule to prevent lookahead bias.
    No data retrieved via this context should contain information
    published after the `analysis_date`.
    """
    
    def __init__(self, analysis_date: str | date | datetime):
        if isinstance(analysis_date, str):
            self.analysis_date = pd.to_datetime(analysis_date).date()
        elif isinstance(analysis_date, datetime):
            self.analysis_date = analysis_date.date()
        else:
            self.analysis_date = analysis_date
            
    def validate_date(self, data_date: date) -> bool:
        """
        Returns True if the data is valid to use on the analysis_date.
        """
        return data_date <= self.analysis_date
        
    def filter_dataframe(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Filters a DataFrame to only include rows where the date_column is <= analysis_date.
        Assumes date_column is a datetime type.
        """
        mask = df[date_column].dt.date <= self.analysis_date
        filtered = df.loc[mask].copy()
        
        if len(df) != len(filtered):
            dropped = len(df) - len(filtered)
            logger.debug(f"PointInTimeContext: Dropped {dropped} rows from the future.")
            
        return filtered

    def __repr__(self):
        return f"<PointInTimeContext(analysis_date={self.analysis_date})>"
