import pandas as pd
import io
from typing import Dict, List, Optional

class DataLoader:
    """
    Handles loading user-uploaded CSV files into Pandas DataFrames.
    """

    @staticmethod
    def load_files(uploaded_files) -> Dict[str, pd.DataFrame]:
        """
        Reads a list of uploaded files (from Streamlit) and returns a dictionary
        mapping filenames to DataFrames.
        """
        data_frames = {}
        errors = []

        for uploaded_file in uploaded_files:
            try:
                # Attempt to read CSV with default utf-8 encoding
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                # Fallback to latin-1 if utf-8 fails
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')
            except Exception as e:
                errors.append(f"Error loading {uploaded_file.name}: {str(e)}")
                continue

            # Basic cleaning: strip whitespace from column names
            df.columns = df.columns.astype(str).str.strip()
            
            data_frames[uploaded_file.name] = df

        if errors:
            # In a real app we might raise or return errors, for now just print or log
            print(f"Errors encountered: {errors}")
            
        return data_frames
