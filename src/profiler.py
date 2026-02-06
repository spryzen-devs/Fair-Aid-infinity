import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List

class ColumnProfiler:
    """
    Analyzes columns in a DataFrame to extract metadata useful for semantic inference.
    """

    @staticmethod
    def profile(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Profiles each column in the DataFrame.
        Returns a dictionary keyed by column name containing metadata.
        """
        profile_data = {}

        for col in df.columns:
            series = df[col]
            col_type = 'numeric' if pd.api.types.is_numeric_dtype(series) else 'categorical'
            
            # Basic stats
            missing_count = series.isna().sum()
            total_count = len(series)
            missing_ratio = missing_count / total_count if total_count > 0 else 0
            
            meta = {
                "type": col_type,
                "missing_ratio": missing_ratio,
                "tokens": ColumnProfiler._tokenize(col),
                "original_name": col
            }

            if col_type == 'numeric':
                meta.update({
                    "min": float(series.min()) if not series.empty else None,
                    "max": float(series.max()) if not series.empty else None,
                    "mean": float(series.mean()) if not series.empty else None,
                    "std": float(series.std()) if not series.empty else None,
                    # Simple heuristic for distribution shape
                    "skew": float(series.skew()) if not series.empty and total_count > 2 else 0
                })
            else:
                meta.update({
                    "unique_count": series.nunique(),
                    "top_value": str(series.mode()[0]) if not series.mode().empty else None
                })
            
            profile_data[col] = meta
            
        return profile_data

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Splits column names into tokens (e.g., 'total_funds_2022' -> ['total', 'funds', '2022']).
        """
        # Split by underscore, space, or camelCase
        text = str(text).lower()
        tokens = re.split(r'[_\s\W]+', text)
        return [t for t in tokens if t]
