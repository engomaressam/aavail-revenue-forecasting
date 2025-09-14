#!/usr/bin/env python
"""
Data Ingestion Module for AAVAIL Revenue Prediction
Handles loading and processing of JSON transaction data
"""

import os
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    """
    Handles data ingestion from multiple JSON sources with quality assurance checks
    """
    
    def __init__(self, data_directory: str):
        """
        Initialize DataIngestion class
        
        Args:
            data_directory: Path to directory containing JSON files
        """
        self.data_directory = data_directory
        self.required_columns = ['country', 'customer_id', 'day', 'invoice', 
                               'month', 'price', 'stream_id', 'times_viewed', 'year']
        
    def validate_data_directory(self) -> bool:
        """
        Validate that data directory exists and contains files
        
        Returns:
            bool: True if valid, raises Exception otherwise
        """
        if not os.path.isdir(self.data_directory):
            raise Exception(f"Data directory does not exist: {self.data_directory}")
        
        files = os.listdir(self.data_directory)
        json_files = [f for f in files if f.endswith('.json')]
        
        if len(json_files) == 0:
            raise Exception("No JSON files found in data directory")
            
        logger.info(f"Found {len(json_files)} JSON files in data directory")
        return True
    
    def load_json_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a single JSON file and perform basic validation
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            DataFrame: Loaded and validated data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} records from {os.path.basename(file_path)}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise
    
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to handle variations across files
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: DataFrame with standardized column names
        """
        df = df.copy()
        
        # Handle common column name variations
        column_mapping = {
            'StreamID': 'stream_id',
            'TimesViewed': 'times_viewed',
            'CustomerID': 'customer_id',
            'Country': 'country',
            'Invoice': 'invoice',
            'Price': 'price',
            'total_price': 'price',   # five monthly files (Dec 2017, Jan 2018, Dec 2018, Jan 2019, Dec 2019) use this name
            'Day': 'day',
            'Month': 'month',
            'Year': 'year'
        }
        
        df.rename(columns=column_mapping, inplace=True)
        return df
    
    def clean_invoice_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean invoice IDs by removing letters for better matching
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: DataFrame with cleaned invoice IDs
        """
        df = df.copy()
        if 'invoice' in df.columns:
            # Remove letters from invoice IDs, keep only numbers
            df['invoice_clean'] = df['invoice'].astype(str).str.extract('(\d+)')
            df['invoice_clean'] = pd.to_numeric(df['invoice_clean'], errors='coerce')
        return df
    
    def create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create date-related features for time series analysis
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: DataFrame with date features
        """
        df = df.copy()
        
        # Create date column
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
        
        # Create additional time features
        df['month_year'] = df['date'].dt.to_period('M')
        df['quarter'] = df['date'].dt.quarter
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        return df
    
    def quality_assurance_checks(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Perform quality assurance checks on the data
        
        Args:
            df: Input DataFrame
            filename: Name of source file for logging
            
        Returns:
            DataFrame: Cleaned DataFrame
        """
        df = df.copy()
        initial_rows = len(df)
        
        # Check for required columns
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in {filename}: {missing_cols}")
        
        # Remove rows with missing critical data
        critical_columns = ['country', 'price', 'date']
        df = df.dropna(subset=[col for col in critical_columns if col in df.columns])
        
        # Remove rows with negative prices
        if 'price' in df.columns:
            df = df[df['price'] >= 0]
        
        # Remove future dates
        if 'date' in df.columns:
            df = df[df['date'] <= datetime.now()]
        
        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            logger.info(f"Removed {rows_removed} rows during QA for {filename}")
            
        return df
    
    def load_all_data(self) -> pd.DataFrame:
        """
        Load all JSON files and combine into a single DataFrame
        
        Returns:
            DataFrame: Combined and processed dataset
        """
        self.validate_data_directory()
        
        all_dataframes = []
        file_list = [f for f in os.listdir(self.data_directory) if f.endswith('.json')]
        
        for filename in sorted(file_list):
            file_path = os.path.join(self.data_directory, filename)
            
            try:
                # Load and process each file
                df = self.load_json_file(file_path)
                df = self.standardize_column_names(df)
                df = self.clean_invoice_ids(df)
                df = self.create_date_features(df)
                df = self.quality_assurance_checks(df, filename)
                
                # Add source file information
                df['source_file'] = filename
                
                all_dataframes.append(df)
                
            except Exception as e:
                logger.error(f"Failed to process {filename}: {str(e)}")
                continue
        
        if not all_dataframes:
            raise Exception("No data files were successfully loaded")
        
        # Combine all data
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Final processing
        combined_df = combined_df.sort_values('date')
        combined_df = combined_df.reset_index(drop=True)
        
        logger.info(f"Successfully loaded {len(combined_df)} total records from {len(all_dataframes)} files")
        logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        logger.info(f"Countries: {combined_df['country'].nunique()}")
        
        return combined_df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            },
            'countries': {
                'total': df['country'].nunique(),
                'list': df['country'].unique().tolist()
            },
            'revenue': {
                'total': df['price'].sum(),
                'mean_transaction': df['price'].mean(),
                'median_transaction': df['price'].median()
            },
            'missing_data': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        return summary

def load_retail_data(data_path: str) -> pd.DataFrame:
    """
    Load retail data from Excel file for AAVAIL Revenue Prediction
    
    Args:
        data_path: Path to Excel file (Online Retail.xlsx)
        
    Returns:
        DataFrame: Processed retail data
    """
    try:
        # Load Excel file
        logger.info(f"Loading data from: {data_path}")
        df = pd.read_excel(data_path, sheet_name=0)
        
        # Basic data processing
        logger.info(f"Raw data shape: {df.shape}")
        
        # Convert InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Remove rows with missing CustomerID for most analyses (keep for some)
        logger.info(f"Missing CustomerIDs: {df['CustomerID'].isnull().sum()}")
        
        # Basic data quality checks
        logger.info(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
        logger.info(f"Countries: {df['Country'].nunique()}")
        logger.info(f"Unique customers: {df['CustomerID'].nunique()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading retail data: {str(e)}")
        raise Exception(f"Failed to load retail data from {data_path}: {str(e)}")


def load_retail_data_with_summary(data_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Load retail data and return with summary statistics
    
    Args:
        data_path: Path to Excel file
        
    Returns:
        tuple: (processed_dataframe, summary_statistics)
    """
    df = load_retail_data(data_path)
    
    # Generate summary statistics
    summary = {
        'total_records': len(df),
        'date_range': f"{df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}",
        'countries': df['Country'].nunique(),
        'unique_customers': df['CustomerID'].nunique(),
        'missing_customer_ids': df['CustomerID'].isnull().sum(),
        'total_revenue': (df['Quantity'] * df['UnitPrice']).sum()
    }
    
    return df, summary

if __name__ == "__main__":
    # Example usage
    data_dir = "../ai-workflow-capstone-master/cs-train"
    df, summary = load_retail_data(data_dir)
    print("Data loading completed successfully!")
    print(f"Loaded {summary['total_records']} records")
