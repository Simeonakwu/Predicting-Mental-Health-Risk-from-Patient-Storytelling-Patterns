"""
DAIC-WOZ Dataset Data Cleaning and Processing Pipeline
=====================================================

This module contains all the functions and code for cleaning and preprocessing
the DAIC-WOZ depression dataset for machine learning analysis using existing local data.

Author: Extracted from dcapswoz_analysis.ipynb
Date: October 2025
"""

import os
import zipfile
import pandas as pd
import numpy as np
import warnings
import io
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration constants
DATA_DIR = "./Dataset"  # Use existing Dataset folder
MAX_FEATURES = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 42

def setup_directories():
    """Verify existing directories for data storage."""
    print("🚀 Verifying directories...")
    
    # Main data directory (Dataset folder already exists)
    raw_data_dir = os.path.join(DATA_DIR, 'raw_data')
    participant_data_dir = os.path.join(DATA_DIR, 'raw_data', 'participant_data')
    extracted_transcripts_dir = os.path.join(DATA_DIR, 'raw_data', 'extracted_transcripts')
    processed_data_dir = os.path.join(DATA_DIR, 'processed_data')
    
    # Create extracted_transcripts_dir if it doesn't exist (needed for processing)
    os.makedirs(extracted_transcripts_dir, exist_ok=True)
    
    print(f"✅ Using existing data directories:")
    print(f"   📁 Raw data: {raw_data_dir}")
    print(f"   📁 Participant data: {participant_data_dir}")
    print(f"   📁 Processed data: {processed_data_dir}")
    
    return participant_data_dir, extracted_transcripts_dir

def load_and_combine_labels():
    """Load and combine the train/dev label files from existing local files."""
    print("📊 Step 1: Loading and combining existing label files...")
    
    raw_data_dir = os.path.join(DATA_DIR, 'raw_data')
    
    try:
        # Check if files exist
        train_file = os.path.join(raw_data_dir, 'train_split_Depression_AVEC2017.csv')
        dev_file = os.path.join(raw_data_dir, 'dev_split_Depression_AVEC2017.csv')
        test_file = os.path.join(raw_data_dir, 'test_split_Depression_AVEC2017.csv')
        
        for file_path, file_name in [(train_file, 'train'), (dev_file, 'dev'), (test_file, 'test')]:
            if not os.path.exists(file_path):
                print(f"❌ {file_name} file not found: {file_path}")
                return None, None
                
        # Load the labels from raw_data directory
        train_df = pd.read_csv(train_file)
        dev_df = pd.read_csv(dev_file)
        test_df = pd.read_csv(test_file)
        
        # Combine train and dev splits (test split doesn't contain labels)
        df = pd.concat([train_df, dev_df], ignore_index=True)
        
        print(f"  ✅ Train set: {len(train_df)} participants")
        print(f"  ✅ Dev set: {len(dev_df)} participants") 
        print(f"  ✅ Test set: {len(test_df)} participants")
        print(f"  ✅ Combined train+dev: {len(df)} participants")
        print(f"  Available columns: {df.columns.tolist()}")
        
        return df, test_df
        
    except FileNotFoundError as e:
        print(f"❌ Error loading label files: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Error processing label files: {e}")
        return None, None

def extract_participant_transcripts(df, participant_data_dir, extracted_transcripts_dir):
    """
    Extract transcripts from existing local participant zip files.
    
    Args:
        df (pd.DataFrame): Combined dataframe with participant IDs
        participant_data_dir (str): Directory containing existing zip files
        extracted_transcripts_dir (str): Directory for extracted transcripts
    
    Returns:
        dict: Dictionary mapping participant_id to extracted text
    """
    print("📝 Step 2: Extracting participant transcripts from existing local files...")
    
    texts = {}
    unique_participants = df['Participant_ID'].unique()
    total_participants = len(unique_participants)
    processed_count = 0
    missing_files = []
    
    for i, participant_id in enumerate(unique_participants):
        if i % 10 == 0:  # Progress indicator
            print(f"  Processing participant {i+1}/{total_participants}: {participant_id}")
            
        zip_filename = f"{participant_id}_P.zip"
        zip_path = os.path.join(participant_data_dir, zip_filename)
        
        # Check if zip file exists locally
        if not os.path.exists(zip_path):
            missing_files.append(participant_id)
            continue
            
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract only the transcript CSV file
                for file_info in zip_ref.infolist():
                    if file_info.filename.endswith('_TRANSCRIPT.csv'):
                        zip_ref.extract(file_info.filename, extracted_transcripts_dir)
                        transcript_path = os.path.join(extracted_transcripts_dir, file_info.filename)
                        
                        # Read transcript and extract participant text
                        transcript_df = pd.read_csv(transcript_path, sep='\t', header=0)
                        participant_text = ' '.join(
                            transcript_df[transcript_df['speaker'] == 'Participant']['value'].astype(str).tolist()
                        )
                        texts[participant_id] = participant_text
                        processed_count += 1
                        
                        # Clean up the extracted transcript file
                        os.remove(transcript_path)
                        break  # Only one transcript CSV per zip
                        
        except zipfile.BadZipFile:
            print(f"  ⚠️ Bad zip file for participant {participant_id}. Skipping.")
        except Exception as e:
            print(f"  ⚠️ Error processing zip file for participant {participant_id}: {e}")
    
    print(f"✅ Successfully extracted {processed_count} participant transcripts")
    if missing_files:
        print(f"⚠️ Missing zip files for {len(missing_files)} participants")
        if len(missing_files) <= 10:
            print(f"   Missing: {missing_files}")
        else:
            print(f"   Missing: {missing_files[:10]}... and {len(missing_files)-10} more")
    
    return texts

def merge_texts_with_labels(df, texts):
    """
    Merge extracted transcript texts with the label dataframe.
    
    Args:
        df (pd.DataFrame): Combined dataframe with labels
        texts (dict): Dictionary mapping participant_id to text
    
    Returns:
        pd.DataFrame: Cleaned dataframe with text and labels
    """
    print("🔗 Step 3: Merging texts with labels...")
    
    # Add text column
    df['text'] = df['Participant_ID'].map(texts)
    
    # Remove participants without text data
    initial_count = len(df)
    df.dropna(subset=['text'], inplace=True)
    final_count = len(df)
    
    print(f"  Participants with text data: {final_count}/{initial_count}")
    print(f"  Removed {initial_count - final_count} participants without text")
    
    # Ensure PHQ8_Binary is integer type
    df['PHQ8_Binary'] = df['PHQ8_Binary'].astype(int)
    
    # Display label distribution
    print(f"  Label distribution:\n{df['PHQ8_Binary'].value_counts()}")
    
    return df

def create_train_test_split(df):
    """
    Create train/test split from the cleaned dataframe.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe with text and labels
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print("✂️ Step 4: Creating train/test split...")
    
    # Prepare features and labels
    X = df['text']
    y = df['PHQ8_Binary']
    
    # Create stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y
    )
    
    print(f"  Training set size: {len(X_train)}")
    print(f"  Test set size: {len(X_test)}")
    print(f"  Training label distribution:\n{y_train.value_counts()}")
    print(f"  Test label distribution:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test

def create_tfidf_features(X_train, X_test):
    """
    Create TF-IDF features from text data.
    
    Args:
        X_train (pd.Series): Training text data
        X_test (pd.Series): Test text data
    
    Returns:
        tuple: X_train_tfidf, X_test_tfidf, tfidf_vectorizer
    """
    print("🔤 Step 5: Creating TF-IDF features...")
    
    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES, 
        stop_words='english'
    )
    
    # Fit and transform training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    print(f"  TF-IDF matrix shape - Train: {X_train_tfidf.shape}, Test: {X_test_tfidf.shape}")
    print(f"  Number of features: {len(tfidf_vectorizer.get_feature_names_out())}")
    
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer

def save_processed_data(df, X_train, X_test, y_train, y_test, X_train_tfidf, X_test_tfidf, tfidf_vectorizer):
    """
    Save all processed data to disk for future use.
    
    Args:
        df (pd.DataFrame): Complete processed dataframe
        X_train, X_test, y_train, y_test: Split datasets
        X_train_tfidf, X_test_tfidf: TF-IDF transformed data
        tfidf_vectorizer: Fitted TF-IDF vectorizer
    """
    print("💾 Step 6: Saving processed data...")
    
    processed_data_dir = os.path.join(DATA_DIR, 'processed_data')
    
    # Save processed dataframe
    df.to_csv(os.path.join(processed_data_dir, 'processed_dataframe.csv'), index=False)
    df.to_pickle(os.path.join(processed_data_dir, 'processed_dataframe.pkl'))
    
    # Save train/test splits
    splits_data = {
        'X_train': X_train,
        'X_test': X_test, 
        'y_train': y_train,
        'y_test': y_test
    }
    
    with open(os.path.join(processed_data_dir, 'train_test_splits.pkl'), 'wb') as f:
        pickle.dump(splits_data, f)
    
    # Save TF-IDF data
    tfidf_data = {
        'X_train_tfidf': X_train_tfidf,
        'X_test_tfidf': X_test_tfidf,
        'vectorizer': tfidf_vectorizer
    }
    
    with open(os.path.join(processed_data_dir, 'tfidf_features.pkl'), 'wb') as f:
        pickle.dump(tfidf_data, f)
    
    print("✅ All processed data saved successfully")

def load_processed_data():
    """
    Load previously processed data from disk.
    
    Returns:
        tuple: All processed data components or None if not found
    """
    try:
        print("📂 Loading previously processed data...")
        
        processed_data_dir = os.path.join(DATA_DIR, 'processed_data')
        
        # Load processed dataframe
        df = pd.read_pickle(os.path.join(processed_data_dir, 'processed_dataframe.pkl'))
        
        # Load train/test splits
        with open(os.path.join(processed_data_dir, 'train_test_splits.pkl'), 'rb') as f:
            splits_data = pickle.load(f)
        
        # Load TF-IDF data
        with open(os.path.join(processed_data_dir, 'tfidf_features.pkl'), 'rb') as f:
            tfidf_data = pickle.load(f)
        
        print("✅ Successfully loaded processed data from disk")
        return (df, splits_data['X_train'], splits_data['X_test'], 
                splits_data['y_train'], splits_data['y_test'],
                tfidf_data['X_train_tfidf'], tfidf_data['X_test_tfidf'], 
                tfidf_data['vectorizer'])
        
    except (FileNotFoundError, ModuleNotFoundError, ImportError) as e:
        if isinstance(e, (ModuleNotFoundError, ImportError)):
            print("⚠️ Pickle compatibility issue detected (likely NumPy version mismatch)")
            print("   Will reprocess data with current environment...")
        else:
            print("⚠️ No previously processed data found")
        return None
    except Exception as e:
        print(f"⚠️ Error loading processed data: {e}")
        print("   Will reprocess data...")
        return None

def run_complete_pipeline(force_reprocess=False):
    """
    Run the complete data cleaning and processing pipeline using existing local files.
    
    Args:
        force_reprocess (bool): If True, reprocess even if cached data exists
    
    Returns:
        tuple: All processed data components
    """
    print("="*60)
    print("🌟 DAIC-WOZ DATA CLEANING AND PROCESSING PIPELINE")
    print("🗂️  Using Existing Local Data Files")
    print("="*60)
    
    # Check if processed data already exists
    if not force_reprocess:
        processed_data = load_processed_data()
        if processed_data is not None:
            print("📂 Using previously processed data from cache")
            return processed_data
    
    print("🔄 Starting fresh data processing pipeline with existing local files...")
    
    # Step 1: Verify directories
    participant_data_dir, extracted_transcripts_dir = setup_directories()
    
    # Step 2: Load and combine labels from existing local files
    df, test_df = load_and_combine_labels()
    if df is None:
        print("❌ Failed to load label files. Please ensure CSV files are in Dataset/raw_data/")
        return None
    
    # Step 3: Extract participant transcripts from existing local zip files
    texts = extract_participant_transcripts(df, participant_data_dir, extracted_transcripts_dir)
    if not texts:
        print("❌ Failed to extract any transcripts. Please ensure ZIP files are in Dataset/raw_data/participant_data/")
        return None
    
    # Step 4: Merge texts with labels
    df = merge_texts_with_labels(df, texts)
    if df.empty:
        print("❌ No participants with both text and labels.")
        return None
    
    # Step 5: Create train/test split
    X_train, X_test, y_train, y_test = create_train_test_split(df)
    
    # Step 6: Create TF-IDF features
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = create_tfidf_features(X_train, X_test)
    
    # Step 7: Save processed data
    save_processed_data(df, X_train, X_test, y_train, y_test, 
                       X_train_tfidf, X_test_tfidf, tfidf_vectorizer)
    
    print("="*60)
    print("🎉 DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return df, X_train, X_test, y_train, y_test, X_train_tfidf, X_test_tfidf, tfidf_vectorizer

def get_data_summary(df):
    """
    Print a comprehensive summary of the processed data.
    
    Args:
        df (pd.DataFrame): Processed dataframe
    """
    print("\n📋 DATA SUMMARY")
    print("-" * 40)
    print(f"Total participants: {len(df)}")
    print(f"Total features available: {df.columns.tolist()}")
    print(f"Text samples available: {df['text'].notna().sum()}")
    print(f"Average text length: {df['text'].str.len().mean():.0f} characters")
    print(f"PHQ8_Binary distribution:\n{df['PHQ8_Binary'].value_counts()}")
    
    if 'Gender' in df.columns:
        print(f"Gender distribution:\n{df['Gender'].value_counts()}")
    
    print(f"Sample participant text (first 200 chars):")
    print(f"'{df['text'].iloc[0][:200]}...'")
    print("-" * 40)

def check_local_data_availability():
    """
    Check what data files are available locally in the Dataset folder.
    
    Returns:
        dict: Summary of available data files
    """
    print("🔍 Checking local data availability in Dataset folder...")
    
    raw_data_dir = os.path.join(DATA_DIR, 'raw_data')
    participant_data_dir = os.path.join(DATA_DIR, 'raw_data', 'participant_data')
    processed_data_dir = os.path.join(DATA_DIR, 'processed_data')
    
    availability = {
        'csv_files': {},
        'zip_files': 0,
        'processed_files': {}
    }
    
    # Check CSV files
    csv_files = ['train_split_Depression_AVEC2017.csv', 'dev_split_Depression_AVEC2017.csv', 'test_split_Depression_AVEC2017.csv']
    for csv_file in csv_files:
        file_path = os.path.join(raw_data_dir, csv_file)
        availability['csv_files'][csv_file] = os.path.exists(file_path)
    
    # Check ZIP files
    if os.path.exists(participant_data_dir):
        zip_files = [f for f in os.listdir(participant_data_dir) if f.endswith('_P.zip')]
        availability['zip_files'] = len(zip_files)
        
        # Show sample of available participants
        if zip_files:
            participant_ids = [f.replace('_P.zip', '') for f in zip_files[:10]]
            print(f"   Sample participants: {participant_ids}...")
    
    # Check processed files
    processed_files = ['processed_dataframe.pkl', 'train_test_splits.pkl', 'tfidf_features.pkl']
    for proc_file in processed_files:
        file_path = os.path.join(processed_data_dir, proc_file)
        availability['processed_files'][proc_file] = os.path.exists(file_path)
    
    # Print summary
    print(f"📊 CSV Label Files:")
    for file, exists in availability['csv_files'].items():
        status = "✅" if exists else "❌"
        print(f"   {status} {file}")
    
    print(f"📦 Participant ZIP Files: {availability['zip_files']} files found")
    
    print(f"💾 Processed Cache Files:")
    for file, exists in availability['processed_files'].items():
        status = "✅" if exists else "❌"
        print(f"   {status} {file}")
    
    return availability

if __name__ == "__main__":
    print("🚀 DAIC-WOZ Local Data Processing Pipeline")
    print("Working with existing data in Dataset folder")
    print("-" * 50)
    
    # Check data availability first
    availability = check_local_data_availability()
    
    # Determine if we can proceed
    csv_available = all(availability['csv_files'].values())
    zip_available = availability['zip_files'] > 0
    
    if not csv_available:
        print("❌ Required CSV files are missing. Cannot proceed.")
        exit(1)
    
    if not zip_available:
        print("❌ No participant ZIP files found. Cannot proceed.")
        exit(1)
    
    print(f"\n✅ Found {availability['zip_files']} participant files. Proceeding with analysis...")
    
    # Run the complete pipeline
    results = run_complete_pipeline()
    
    if results is not None:
        df, X_train, X_test, y_train, y_test, X_train_tfidf, X_test_tfidf, tfidf_vectorizer = results
        get_data_summary(df)
        
        print(f"\n🎯 DATA READY FOR MACHINE LEARNING!")
        print(f"   📊 Training samples: {len(X_train)}")
        print(f"   📊 Test samples: {len(X_test)}")
        print(f"   🔤 TF-IDF features: {X_train_tfidf.shape[1]}")
        print(f"   💾 All data saved to: {os.path.join(DATA_DIR, 'processed_data')}")
        print(f"\n🚀 Ready for model training and analysis!")
    else:
        print("❌ Pipeline failed. Please check the error messages above.")