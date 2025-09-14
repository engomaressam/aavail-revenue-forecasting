"""Tests for loading the AAVAIL invoices (src/data_ingestion.py)."""

import json
import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_ingestion import DataIngestion  # noqa: E402


class TestDataIngestion(unittest.TestCase):
    """Test cases for data ingestion functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "cs-train")
        
    def test_data_ingestion_initialization(self):
        """Test DataIngestion class initialization"""
        ingestion = DataIngestion(self.test_data_dir)
        self.assertEqual(ingestion.data_directory, self.test_data_dir)
        self.assertIsInstance(ingestion.required_columns, list)
        self.assertGreater(len(ingestion.required_columns), 0)
    
    def test_column_standardization(self):
        """Test column name standardization"""
        ingestion = DataIngestion(self.test_data_dir)
        
        # Create test dataframe with non-standard column names
        test_df = pd.DataFrame({
            'StreamID': [1, 2, 3],
            'TimesViewed': [10, 20, 30],
            'CustomerID': ['A', 'B', 'C']
        })
        
        standardized_df = ingestion.standardize_column_names(test_df)
        
        self.assertIn('stream_id', standardized_df.columns)
        self.assertIn('times_viewed', standardized_df.columns)
        self.assertIn('customer_id', standardized_df.columns)
    
    def test_date_feature_creation(self):
        """Test date feature engineering"""
        ingestion = DataIngestion(self.test_data_dir)
        
        test_df = pd.DataFrame({
            'year': [2018, 2019, 2020],
            'month': [1, 6, 12],
            'day': [15, 20, 25]
        })
        
        df_with_dates = ingestion.create_date_features(test_df)
        
        self.assertIn('date', df_with_dates.columns)
        self.assertIn('month_year', df_with_dates.columns)
        self.assertIn('quarter', df_with_dates.columns)
        self.assertIn('day_of_week', df_with_dates.columns)
        self.assertIn('is_weekend', df_with_dates.columns)

class TestRevenueColumn(unittest.TestCase):
    """Five monthly files name the revenue column total_price; it must load as price."""

    def test_total_price_is_loaded_as_price(self):
        rows = [{"country": "France", "customer_id": 1, "invoice": "489434", "total_price": 12.5,
                 "StreamID": "85048", "TimesViewed": 3, "year": 2018, "month": 12, "day": 1}]
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "invoices-2018-12.json"), "w") as fh:
                json.dump(rows, fh)
            df = DataIngestion(tmp).load_all_data()
        self.assertIn("price", df.columns)
        self.assertNotIn("total_price", df.columns)
        self.assertEqual(df["price"].sum(), 12.5)


if __name__ == "__main__":
    unittest.main()
