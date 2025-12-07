#!/usr/bin/env python3
"""
Task 3 Data Collector: Detect Rare and Unexpected Drug-Event Relationships
Collects structured drug-event association data for anomaly detection

Objectives:
- Collect adverse event data for oncology drugs
- Extract key fields: drug name, adverse event, severity, patient info, time
- Generate structured dataset suitable for anomaly detection
"""

import requests
import pandas as pd
import json
import time
from collections import defaultdict

# Oncology drugs including bispecific antibodies and competitors
# Added Glofitamab and Mosunetuzumab as requested (competitors to Epcoritamab)
ONCOLOGY_DRUGS = [
    # Bispecific antibodies (CD20xCD3) - KEY FOCUS
    "Epcoritamab",      # AbbVie/Genmab - approved 2023
    "Glofitamab",       # Roche - competitor, approved 2023
    "Mosunetuzumab",    # Genentech - competitor, approved 2022
    
    # Checkpoint inhibitors
    "Pembrolizumab", "Nivolumab", "Atezolizumab", "Durvalumab", "Ipilimumab",
    
    # Monoclonal antibodies
    "Trastuzumab", "Bevacizumab", "Cetuximab", "Rituximab",
    
    # Tyrosine kinase inhibitors
    "Imatinib", "Erlotinib", "Gefitinib", "Osimertinib", "Crizotinib",
    
    # Chemotherapy
    "Paclitaxel", "Docetaxel", "Doxorubicin", "Carboplatin", "Cisplatin",
    
    # Multiple myeloma drugs
    "Lenalidomide", "Pomalidomide", "Bortezomib", "Carfilzomib", "Venetoclax",
    
    # BTK inhibitors
    "Ibrutinib",
    
    # PARP inhibitors
    "Olaparib", "Rucaparib", "Niraparib", "Talazoparib",
    
    # CDK4/6 inhibitors
    "Palbociclib", "Ribociclib", "Abemaciclib",
    
    # BRAF inhibitors
    "Vemurafenib", "Dabrafenib"
]

BASE_URL = "https://api.fda.gov/drug/event.json"

def collect_drug_events(drug_name, limit=500):
    """
    Collect adverse event data for a specific drug
    """
    print(f"Collecting data for {drug_name}...")
    
    all_records = []
    skip = 0
    
    while len(all_records) < limit:
        try:
            # Build query
            params = {
                'search': f'patient.drug.openfda.generic_name:"{drug_name}"',
                'limit': min(100, limit - len(all_records)),
                'skip': skip
            }
            
            response = requests.get(BASE_URL, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                if not results:
                    print(f"  ✓ {drug_name}: No more data found, collected {len(all_records)} records")
                    break
                
                # Process each record
                for record in results:
                    processed = process_record(record, drug_name)
                    if processed:
                        all_records.append(processed)
                
                print(f"  Progress: {len(all_records)}/{limit}")
                skip += len(results)
                
                # API rate limiting
                time.sleep(0.3)
                
            elif response.status_code == 404:
                print(f"  ⚠ {drug_name}: No data found")
                break
            else:
                print(f"  ✗ {drug_name}: HTTP {response.status_code}")
                break
                
        except Exception as e:
            print(f"  ✗ {drug_name}: Error - {str(e)}")
            break
    
    print(f"  ✓ {drug_name}: Complete, {len(all_records)} records\n")
    return all_records

def process_record(record, target_drug):
    """
    Extract key fields and generate structured records
    """
    try:
        # Basic information
        safety_id = record.get('safetyreportid', '')
        receive_date = record.get('receivedate', '')
        
        # Patient information
        patient = record.get('patient', {})
        age = patient.get('patientonsetage', '')
        age_unit = patient.get('patientonsetageunit', '')
        sex = patient.get('patientsex', '')
        
        # Seriousness indicators
        serious = record.get('serious', 0)
        seriousness_death = record.get('seriousnessdeath', 0)
        seriousness_hosp = record.get('seriousnesshospitalization', 0)
        seriousness_life = record.get('seriousnesslifethreatening', 0)
        seriousness_disable = record.get('seriousnessdisabling', 0)
        
        # Extract all drugs
        drugs = patient.get('drug', [])
        drug_names = []
        drug_indications = []
        for drug in drugs:
            openfda = drug.get('openfda', {})
            generic_names = openfda.get('generic_name', [])
            drug_names.extend(generic_names)
            indication = drug.get('drugindication', '')
            if indication:
                drug_indications.append(indication)
        
        # Extract all adverse events (MedDRA terms)
        reactions = patient.get('reaction', [])
        adverse_events = []
        for reaction in reactions:
            ae_term = reaction.get('reactionmeddrapt', '')
            if ae_term:
                adverse_events.append(ae_term)
        
        # Create one record for each adverse event (drug-event pair)
        processed_records = []
        for ae in adverse_events:
            processed_records.append({
                'safety_report_id': safety_id,
                'receive_date': receive_date,
                'target_drug': target_drug,
                'all_drugs': '|'.join(drug_names),
                'drug_count': len(drugs),
                'adverse_event': ae,
                'event_count': len(adverse_events),
                'patient_age': age,
                'patient_age_unit': age_unit,
                'patient_sex': sex,
                'is_serious': serious,
                'is_death': seriousness_death,
                'is_hospitalization': seriousness_hosp,
                'is_lifethreatening': seriousness_life,
                'is_disabling': seriousness_disable,
                'indication': '|'.join(drug_indications)
            })
        
        return processed_records
        
    except Exception as e:
        print(f"    Record processing error: {str(e)}")
        return None

def main():
    """
    Main function: Collect data for all oncology drugs
    """
    print("=" * 80)
    print("Task 3: Detect Rare and Unexpected Drug-Event Relationships - Data Collector")
    print("=" * 80)
    print(f"Target drug count: {len(ONCOLOGY_DRUGS)}")
    print(f"Records per drug: 500\n")
    
    all_data = []
    
    for i, drug in enumerate(ONCOLOGY_DRUGS, 1):
        print(f"[{i}/{len(ONCOLOGY_DRUGS)}] {drug}")
        records = collect_drug_events(drug, limit=500)
        
        # Flatten nested lists
        for record_list in records:
            if isinstance(record_list, list):
                all_data.extend(record_list)
            else:
                all_data.append(record_list)
        
        # Save intermediate results every 10 drugs
        if i % 10 == 0:
            temp_df = pd.DataFrame(all_data)
            temp_df.to_csv(f'task3_data_temp_{i}.csv', index=False)
            print(f"💾 Intermediate results saved: {len(all_data)} records\n")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Data cleaning
    df = df.drop_duplicates()
    df = df.dropna(subset=['adverse_event'])  # Must have adverse event
    
    # Convert numeric columns to proper types
    numeric_cols = ['is_serious', 'is_death', 'is_hospitalization', 'is_lifethreatening', 'is_disabling']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Save results
    output_file = 'task3_oncology_drug_event_pairs.csv'
    df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 80)
    print("Data Collection Complete!")
    print("=" * 80)
    print(f"Total records: {len(df)}")
    print(f"Unique drugs: {df['target_drug'].nunique()}")
    print(f"Unique adverse events: {df['adverse_event'].nunique()}")
    print(f"Serious event rate: {df['is_serious'].sum() / len(df) * 100:.1f}%")
    print(f"Death events: {df['is_death'].sum()}")
    print(f"Output file: {output_file}")
    
    # Display sample rows
    print("\nData Sample (first 5 rows):")
    print(df[['target_drug', 'adverse_event', 'is_serious', 'is_death', 'patient_age', 'patient_sex']].head())
    
    # Statistics
    print("\nAdverse Event Frequency (Top 20):")
    print(df['adverse_event'].value_counts().head(20))
    
    print("\nDrug Event Count Statistics:")
    print(df['target_drug'].value_counts())

if __name__ == "__main__":
    main()


