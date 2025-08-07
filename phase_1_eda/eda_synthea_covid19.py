#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for Synthea COVID-19 Dataset
============================================================

This script performs comprehensive EDA on the Synthea COVID-19 dataset to:
1. Understand data structure and quality
2. Analyze patient demographics and comorbidities
3. Explore temporal patterns in clinical data
4. Identify COVID-19 specific features and outcomes
5. Generate insights for model development

Author: COVID-19 TFT Project Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from pathlib import Path
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SyntheaEDA:
    """
    Exploratory Data Analysis for Synthea COVID-19 Dataset
    """
    
    def __init__(self, data_path: str = "../10k_synthea_covid19_csv"):
        """
        Initialize EDA with data path
        
        Args:
            data_path: Path to Synthea CSV files
        """
        self.data_path = Path(data_path)
        self.tables = {}
        self.covid_patients = None
        self.eda_results = {}
        
        # Create output directory for plots
        self.output_dir = Path("eda_results")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized EDA for data path: {data_path}")
    
    def load_all_tables(self):
        """Load all CSV files from the Synthea dataset"""
        logger.info("Loading all CSV tables...")
        
        csv_files = [
            'patients.csv', 'observations.csv', 'medications.csv',
            'procedures.csv', 'devices.csv', 'encounters.csv',
            'conditions.csv', 'careplans.csv', 'immunizations.csv',
            'allergies.csv', 'imaging_studies.csv', 'organizations.csv',
            'providers.csv', 'payers.csv', 'payer_transitions.csv',
            'supplies.csv'
        ]
        
        for file in csv_files:
            file_path = self.data_path / file
            if file_path.exists():
                table_name = file.replace('.csv', '')
                try:
                    self.tables[table_name] = pd.read_csv(file_path)
                    logger.info(f"Loaded {table_name}: {self.tables[table_name].shape}")
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")
            else:
                logger.warning(f"File not found: {file}")
        
        logger.info(f"Loaded {len(self.tables)} tables")
        return self.tables
    
    def identify_covid_patients(self):
        """Identify patients with COVID-19 diagnosis"""
        logger.info("Identifying COVID-19 patients...")
        
        if 'conditions' not in self.tables:
            logger.error("Conditions table not loaded")
            return None
        
        # Look for COVID-19 related conditions
        covid_keywords = ['covid', 'coronavirus', 'sars-cov-2', 'covid-19']
        
        covid_conditions = self.tables['conditions'].copy()
        covid_conditions['is_covid'] = covid_conditions['DESCRIPTION'].str.contains(
            '|'.join(covid_keywords), case=False, na=False
        )
        
        covid_patients = covid_conditions[covid_conditions['is_covid']]['PATIENT'].unique()
        
        logger.info(f"Found {len(covid_patients)} COVID-19 patients")
        self.covid_patients = covid_patients
        
        return covid_patients
    
    def analyze_patient_demographics(self):
        """Analyze patient demographics"""
        logger.info("Analyzing patient demographics...")
        
        if 'patients' not in self.tables:
            logger.error("Patients table not loaded")
            return
        
        patients = self.tables['patients'].copy()
        
        # Basic demographics
        demographics = {
            'total_patients': len(patients),
            'covid_patients': len(self.covid_patients) if self.covid_patients is not None else 0,
            'avg_age': patients['BIRTHDATE'].apply(lambda x: 2024 - pd.to_datetime(x).year).mean(),
            'gender_distribution': patients['GENDER'].value_counts().to_dict(),
            'race_distribution': patients['RACE'].value_counts().to_dict(),
            'ethnicity_distribution': patients['ETHNICITY'].value_counts().to_dict()
        }
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Age distribution
        ages = patients['BIRTHDATE'].apply(lambda x: 2024 - pd.to_datetime(x).year)
        axes[0, 0].hist(ages, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Age Distribution')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Count')
        
        # Gender distribution
        gender_counts = patients['GENDER'].value_counts()
        axes[0, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Gender Distribution')
        
        # Race distribution
        race_counts = patients['RACE'].value_counts()
        axes[1, 0].bar(range(len(race_counts)), race_counts.values, color='lightcoral')
        axes[1, 0].set_title('Race Distribution')
        axes[1, 0].set_xticks(range(len(race_counts)))
        axes[1, 0].set_xticklabels(race_counts.index, rotation=45)
        
        # Ethnicity distribution
        ethnicity_counts = patients['ETHNICITY'].value_counts()
        axes[1, 1].bar(range(len(ethnicity_counts)), ethnicity_counts.values, color='lightgreen')
        axes[1, 1].set_title('Ethnicity Distribution')
        axes[1, 1].set_xticks(range(len(ethnicity_counts)))
        axes[1, 1].set_xticklabels(ethnicity_counts.index, rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'patient_demographics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.eda_results['demographics'] = demographics
        logger.info("Demographics analysis completed")
        
        return demographics
    
    def analyze_covid_conditions(self):
        """Analyze COVID-19 related conditions"""
        logger.info("Analyzing COVID-19 conditions...")
        
        if 'conditions' not in self.tables or self.covid_patients is None:
            logger.error("Conditions table or COVID patients not available")
            return
        
        covid_conditions = self.tables['conditions'][
            self.tables['conditions']['PATIENT'].isin(self.covid_patients)
        ].copy()
        
        # Analyze condition types
        condition_analysis = {
            'total_covid_conditions': len(covid_conditions),
            'unique_condition_types': covid_conditions['CODE'].nunique(),
            'condition_frequency': covid_conditions['DESCRIPTION'].value_counts().head(20).to_dict()
        }
        
        # Visualize top conditions
        top_conditions = covid_conditions['DESCRIPTION'].value_counts().head(15)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_conditions)), top_conditions.values, color='salmon')
        plt.yticks(range(len(top_conditions)), top_conditions.index)
        plt.xlabel('Count')
        plt.title('Top 15 COVID-19 Related Conditions')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'covid_conditions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.eda_results['covid_conditions'] = condition_analysis
        logger.info("COVID-19 conditions analysis completed")
        
        return condition_analysis
    
    def analyze_medications(self):
        """Analyze medication patterns"""
        logger.info("Analyzing medication patterns...")
        
        if 'medications' not in self.tables:
            logger.error("Medications table not loaded")
            return
        
        medications = self.tables['medications'].copy()
        
        # Filter for COVID patients if available
        if self.covid_patients is not None:
            covid_medications = medications[
                medications['PATIENT'].isin(self.covid_patients)
            ]
        else:
            covid_medications = medications
        
        # Analyze medication patterns
        medication_analysis = {
            'total_medications': len(covid_medications),
            'unique_medications': covid_medications['DESCRIPTION'].nunique(),
            'medication_frequency': covid_medications['DESCRIPTION'].value_counts().head(20).to_dict()
        }
        
        # Visualize top medications
        top_medications = covid_medications['DESCRIPTION'].value_counts().head(15)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_medications)), top_medications.values, color='lightblue')
        plt.yticks(range(len(top_medications)), top_medications.index)
        plt.xlabel('Count')
        plt.title('Top 15 COVID-19 Related Medications')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'covid_medications.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.eda_results['medications'] = medication_analysis
        logger.info("Medication analysis completed")
        
        return medication_analysis
    
    def analyze_observations(self):
        """Analyze clinical observations and vital signs"""
        logger.info("Analyzing clinical observations...")
        
        if 'observations' not in self.tables:
            logger.error("Observations table not loaded")
            return
        
        observations = self.tables['observations'].copy()
        
        # Filter for COVID patients if available
        if self.covid_patients is not None:
            covid_observations = observations[
                observations['PATIENT'].isin(self.covid_patients)
            ]
        else:
            covid_observations = observations
        
        # Analyze observation types
        observation_analysis = {
            'total_observations': len(covid_observations),
            'unique_observation_types': covid_observations['DESCRIPTION'].nunique(),
            'observation_frequency': covid_observations['DESCRIPTION'].value_counts().head(20).to_dict()
        }
        
        # Visualize top observations
        top_observations = covid_observations['DESCRIPTION'].value_counts().head(15)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_observations)), top_observations.values, color='lightgreen')
        plt.yticks(range(len(top_observations)), top_observations.index)
        plt.xlabel('Count')
        plt.title('Top 15 Clinical Observations')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'clinical_observations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.eda_results['observations'] = observation_analysis
        logger.info("Observations analysis completed")
        
        return observation_analysis
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in the data"""
        logger.info("Analyzing temporal patterns...")
        
        temporal_analysis = {}
        
        # Analyze encounters over time
        if 'encounters' in self.tables:
            encounters = self.tables['encounters'].copy()
            encounters['START'] = pd.to_datetime(encounters['START'])
            
            # Check if END column exists, otherwise use START
            if 'END' in encounters.columns:
                encounters['END'] = pd.to_datetime(encounters['END'])
            else:
                # If no END column, use START as the reference time
                encounters['END'] = encounters['START']
            
            # Filter for COVID patients if available
            if self.covid_patients is not None:
                covid_encounters = encounters[
                    encounters['PATIENT'].isin(self.covid_patients)
                ]
            else:
                covid_encounters = encounters
            
            # Monthly encounter trends
            monthly_encounters = covid_encounters.groupby(
                covid_encounters['START'].dt.to_period('M')
            ).size()
            
            plt.figure(figsize=(12, 6))
            monthly_encounters.plot(kind='line', marker='o', color='purple')
            plt.title('Monthly Encounter Trends')
            plt.xlabel('Month')
            plt.ylabel('Number of Encounters')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'temporal_patterns.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            temporal_analysis['monthly_encounters'] = monthly_encounters.to_dict()
        
        self.eda_results['temporal_patterns'] = temporal_analysis
        logger.info("Temporal analysis completed")
        
        return temporal_analysis
    
    def analyze_data_quality(self):
        """Analyze data quality and missing values"""
        logger.info("Analyzing data quality...")
        
        quality_analysis = {}
        
        if not self.tables:
            logger.warning("No tables loaded. Skipping data quality analysis.")
            self.eda_results['data_quality'] = quality_analysis
            return quality_analysis
        
        for table_name, table_data in self.tables.items():
            missing_data = table_data.isnull().sum()
            quality_analysis[table_name] = {
                'shape': table_data.shape,
                'missing_values': missing_data.to_dict(),
                'missing_percentage': (missing_data / len(table_data) * 100).to_dict()
            }
        
        # Create missing data visualization
        if quality_analysis:
            missing_summary = pd.DataFrame([
                {
                    'table': table_name,
                    'total_rows': info['shape'][0],
                    'total_columns': info['shape'][1],
                    'missing_percentage': sum(info['missing_percentage'].values())
                }
                for table_name, info in quality_analysis.items()
            ])
            
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.bar(missing_summary['table'], missing_summary['total_rows'], color='skyblue')
            plt.title('Number of Rows per Table')
            plt.xticks(rotation=45)
            
            plt.subplot(2, 1, 2)
            plt.bar(missing_summary['table'], missing_summary['missing_percentage'], color='salmon')
            plt.title('Missing Data Percentage per Table')
            plt.xticks(rotation=45)
            plt.ylabel('Missing %')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'data_quality.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        self.eda_results['data_quality'] = quality_analysis
        logger.info("Data quality analysis completed")
        
        return quality_analysis
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        logger.info("Generating summary report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_overview': {
                'total_tables': len(self.tables),
                'table_names': list(self.tables.keys()),
                'covid_patients': len(self.covid_patients) if self.covid_patients is not None else 0
            },
            'eda_results': self.eda_results
        }
        
        # Convert Period objects to strings for JSON serialization
        def convert_periods(obj):
            if isinstance(obj, dict):
                return {str(k): convert_periods(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_periods(item) for item in obj]
            elif hasattr(obj, 'strftime'):  # Handle datetime-like objects
                return str(obj)
            else:
                return obj
        
        # Save report to JSON
        with open(self.output_dir / 'eda_summary_report.json', 'w') as f:
            json.dump(convert_periods(report), f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*60)
        print("SYNTHEA COVID-19 DATASET EDA SUMMARY")
        print("="*60)
        print(f"Total Tables Loaded: {len(self.tables)}")
        print(f"COVID-19 Patients Identified: {len(self.covid_patients) if self.covid_patients is not None else 0}")
        print(f"Analysis Results Saved to: {self.output_dir}")
        print("="*60)
        
        # Print key insights
        if 'demographics' in self.eda_results:
            demo = self.eda_results['demographics']
            print(f"\n📊 DEMOGRAPHICS:")
            print(f"   Total Patients: {demo['total_patients']}")
            print(f"   COVID Patients: {demo['covid_patients']}")
            print(f"   Average Age: {demo['avg_age']:.1f} years")
        
        if 'covid_conditions' in self.eda_results:
            cond = self.eda_results['covid_conditions']
            print(f"\n🏥 COVID-19 CONDITIONS:")
            print(f"   Total Conditions: {cond['total_covid_conditions']}")
            print(f"   Unique Condition Types: {cond['unique_condition_types']}")
        
        if 'medications' in self.eda_results:
            med = self.eda_results['medications']
            print(f"\n💊 MEDICATIONS:")
            print(f"   Total Medications: {med['total_medications']}")
            print(f"   Unique Medications: {med['unique_medications']}")
        
        if 'observations' in self.eda_results:
            obs = self.eda_results['observations']
            print(f"\n🔬 CLINICAL OBSERVATIONS:")
            print(f"   Total Observations: {obs['total_observations']}")
            print(f"   Unique Observation Types: {obs['unique_observation_types']}")
        
        print("\n" + "="*60)
        logger.info("Summary report generated")
        
        return report
    
    def run_complete_eda(self):
        """Run complete EDA pipeline"""
        logger.info("Starting complete EDA pipeline...")
        
        # Load data
        self.load_all_tables()
        
        # Identify COVID patients
        self.identify_covid_patients()
        
        # Run analyses
        self.analyze_patient_demographics()
        self.analyze_covid_conditions()
        self.analyze_medications()
        self.analyze_observations()
        self.analyze_temporal_patterns()
        self.analyze_data_quality()
        
        # Generate summary
        self.generate_summary_report()
        
        logger.info("Complete EDA pipeline finished")
        
        return self.eda_results


def main():
    """Main function to run EDA"""
    print("🚀 Starting Synthea COVID-19 Dataset EDA...")
    
    # Initialize EDA
    eda = SyntheaEDA()
    
    # Run complete analysis
    results = eda.run_complete_eda()
    
    print("\n✅ EDA completed successfully!")
    print(f"📁 Results saved to: {eda.output_dir}")
    print("📊 Check the generated plots and summary report for insights")


if __name__ == "__main__":
    main() 