#!/usr/bin/env python3
"""
Task 3 Improvement: BERT Clinical Feature Extractor
Identifies clinical features (age, race, medical history) that influence adverse events

Key Features:
1. Extract clinical features from FAERS reports
2. Use BERT for text analysis of narrative fields
3. Identify causal vs correlated factors
4. Support for specific drug-event combinations

Note: This is a framework implementation. Full BERT training requires:
- GPU resources
- Large training dataset
- Fine-tuning on medical text
"""

import pandas as pd
import numpy as np
import requests
import time
import os
from collections import defaultdict

# Try to import transformers (optional dependency)
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Warning: transformers library not installed. BERT features disabled.")
    print("Install with: pip install transformers torch")


class ClinicalFeatureExtractor:
    """
    Extracts and analyzes clinical features associated with drug-event pairs
    Uses BERT for text analysis when available, falls back to rule-based extraction
    """
    
    def __init__(self, use_bert=True):
        """
        Initialize the extractor
        
        Args:
            use_bert: Whether to use BERT for text analysis (requires GPU)
        """
        self.use_bert = use_bert and BERT_AVAILABLE
        
        if self.use_bert:
            print("Initializing BERT model...")
            # Use BioBERT for medical text
            model_name = "dmis-lab/biobert-base-cased-v1.1"
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.eval()
                print(f"✓ Loaded BERT model: {model_name}")
            except Exception as e:
                print(f"✗ Failed to load BERT: {e}")
                self.use_bert = False
        
        self.api_url = "https://api.fda.gov/drug/event.json"
    
    def get_reports_for_drug_event(self, drug, adverse_event, limit=100):
        """
        Fetch FAERS reports for a specific drug-event combination
        
        Args:
            drug: Drug name
            adverse_event: Adverse event term
            limit: Maximum number of reports
        
        Returns:
            List of report dictionaries
        """
        print(f"Fetching reports for {drug} + {adverse_event}...")
        
        reports = []
        
        try:
            # Build query
            params = {
                'search': f'patient.drug.openfda.generic_name:"{drug}" AND patient.reaction.reactionmeddrapt:"{adverse_event}"',
                'limit': limit
            }
            
            response = requests.get(self.api_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                reports = data.get('results', [])
                print(f"✓ Found {len(reports)} reports")
            else:
                print(f"✗ API error: {response.status_code}")
                
        except Exception as e:
            print(f"✗ Error: {e}")
        
        return reports
    
    def extract_clinical_features(self, reports):
        """
        Extract clinical features from FAERS reports
        
        Args:
            reports: List of FAERS report dictionaries
        
        Returns:
            Dictionary with feature distributions
        """
        features = {
            'age': [],
            'sex': [],
            'weight': [],
            'medical_history': [],
            'concomitant_drugs': [],
            'indication': [],
            'outcome': [],
            'country': [],
        }
        
        for report in reports:
            patient = report.get('patient', {})
            
            # Age
            age = patient.get('patientonsetage')
            age_unit = patient.get('patientonsetageunit', '801')  # 801 = years
            if age:
                if age_unit == '801':  # years
                    features['age'].append(float(age))
                elif age_unit == '802':  # months
                    features['age'].append(float(age) / 12)
            
            # Sex
            sex = patient.get('patientsex')
            if sex:
                sex_map = {'1': 'Male', '2': 'Female', '0': 'Unknown'}
                features['sex'].append(sex_map.get(str(sex), 'Unknown'))
            
            # Weight
            weight = patient.get('patientweight')
            if weight:
                features['weight'].append(float(weight))
            
            # Medical history (from drug indications and reactions)
            drugs = patient.get('drug', [])
            for drug in drugs:
                indication = drug.get('drugindication', '')
                if indication:
                    features['indication'].append(indication)
            
            # Concomitant drugs
            for drug in drugs:
                openfda = drug.get('openfda', {})
                drug_names = openfda.get('generic_name', [])
                features['concomitant_drugs'].extend(drug_names)
            
            # Outcome
            serious_outcomes = []
            if report.get('seriousnessdeath'):
                serious_outcomes.append('Death')
            if report.get('seriousnesshospitalization'):
                serious_outcomes.append('Hospitalization')
            if report.get('seriousnesslifethreatening'):
                serious_outcomes.append('Life-threatening')
            if report.get('seriousnessdisabling'):
                serious_outcomes.append('Disability')
            features['outcome'].extend(serious_outcomes if serious_outcomes else ['Non-serious'])
            
            # Country
            country = report.get('occurcountry', 'Unknown')
            features['country'].append(country)
        
        return features
    
    def analyze_features(self, features):
        """
        Analyze extracted features and compute statistics
        
        Args:
            features: Dictionary from extract_clinical_features
        
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        # Age analysis
        if features['age']:
            ages = np.array(features['age'])
            analysis['age'] = {
                'mean': float(np.mean(ages)),
                'median': float(np.median(ages)),
                'std': float(np.std(ages)),
                'min': float(np.min(ages)),
                'max': float(np.max(ages)),
                'count': len(ages),
                'age_groups': {
                    '<18': int(np.sum(ages < 18)),
                    '18-40': int(np.sum((ages >= 18) & (ages < 40))),
                    '40-65': int(np.sum((ages >= 40) & (ages < 65))),
                    '65+': int(np.sum(ages >= 65)),
                }
            }
        
        # Sex distribution
        if features['sex']:
            from collections import Counter
            sex_counts = Counter(features['sex'])
            total = sum(sex_counts.values())
            analysis['sex'] = {
                'distribution': dict(sex_counts),
                'percentages': {k: v/total*100 for k, v in sex_counts.items()},
            }
        
        # Weight analysis
        if features['weight']:
            weights = np.array(features['weight'])
            analysis['weight'] = {
                'mean': float(np.mean(weights)),
                'median': float(np.median(weights)),
                'std': float(np.std(weights)),
            }
        
        # Top indications (medical history proxy)
        if features['indication']:
            from collections import Counter
            indication_counts = Counter(features['indication'])
            analysis['top_indications'] = dict(indication_counts.most_common(10))
        
        # Top concomitant drugs
        if features['concomitant_drugs']:
            from collections import Counter
            drug_counts = Counter(features['concomitant_drugs'])
            analysis['top_concomitant_drugs'] = dict(drug_counts.most_common(10))
        
        # Outcome distribution
        if features['outcome']:
            from collections import Counter
            outcome_counts = Counter(features['outcome'])
            analysis['outcomes'] = dict(outcome_counts)
        
        # Country distribution
        if features['country']:
            from collections import Counter
            country_counts = Counter(features['country'])
            analysis['top_countries'] = dict(country_counts.most_common(5))
        
        return analysis
    
    def identify_risk_factors(self, drug, adverse_event, comparison_ae=None):
        """
        Identify clinical features that are risk factors for the adverse event
        
        Compares feature distributions between:
        - Patients who experienced the AE
        - Patients who did NOT experience the AE (control group)
        
        Args:
            drug: Drug name
            adverse_event: Adverse event to analyze
            comparison_ae: Alternative AE for comparison (default: any other AE)
        
        Returns:
            Dictionary with risk factors and their significance
        """
        print(f"\nAnalyzing risk factors for {drug} + {adverse_event}")
        print("-" * 60)
        
        # Get reports WITH the adverse event
        ae_reports = self.get_reports_for_drug_event(drug, adverse_event, limit=100)
        ae_features = self.extract_clinical_features(ae_reports)
        ae_analysis = self.analyze_features(ae_features)
        
        # Get reports WITHOUT the adverse event (control group)
        # Use a different common AE as control
        if comparison_ae is None:
            # Try to find a balanced control group (gender distribution close to AE group, with sufficient sample size)
            common_aes = ["Fatigue", "Pyrexia", "Nausea", "Diarrhoea", "Headache"]
            best_control_ae = "Fatigue"  # Default
            best_score = float('inf')  # Lower is better
            
            # Get AE group sex distribution for matching
            ae_female_pct = ae_analysis.get('sex', {}).get('percentages', {}).get('Female', 50)
            
            for candidate_ae in common_aes:
                try:
                    candidate_reports = self.get_reports_for_drug_event(drug, candidate_ae, limit=30)
                    if len(candidate_reports) >= 10:  # Need at least 10 reports for statistical power
                        candidate_features = self.extract_clinical_features(candidate_reports)
                        candidate_analysis = self.analyze_features(candidate_features)
                        candidate_female_pct = candidate_analysis.get('sex', {}).get('percentages', {}).get('Female', 50)
                        # Score: balance sex distribution + sample size (prefer larger samples)
                        sex_diff = abs(candidate_female_pct - ae_female_pct)
                        sample_penalty = 100.0 / len(candidate_reports)  # Penalize small samples
                        total_score = sex_diff + sample_penalty
                        if total_score < best_score:
                            best_score = total_score
                            best_control_ae = candidate_ae
                except:
                    continue
            
            comparison_ae = best_control_ae
            print(f"Selected control AE: {comparison_ae} (balanced sex distribution + sample size)")
        
        control_reports = self.get_reports_for_drug_event(drug, comparison_ae, limit=100)
        control_features = self.extract_clinical_features(control_reports)
        control_analysis = self.analyze_features(control_features)
        
        # Compare and identify risk factors
        risk_factors = {}
        
        # Age comparison
        if 'age' in ae_analysis and 'age' in control_analysis:
            ae_age = ae_analysis['age']['mean']
            control_age = control_analysis['age']['mean']
            age_diff = ae_age - control_age
            
            risk_factors['age'] = {
                'ae_group_mean': ae_age,
                'control_group_mean': control_age,
                'difference': age_diff,
                'interpretation': 'Higher age associated with AE' if age_diff > 5 else 
                                 'Lower age associated with AE' if age_diff < -5 else
                                 'No significant age difference',
                'is_risk_factor': abs(age_diff) > 5,
            }
        
        # Sex comparison
        if 'sex' in ae_analysis and 'sex' in control_analysis:
            ae_female_pct = ae_analysis['sex']['percentages'].get('Female', 0)
            control_female_pct = control_analysis['sex']['percentages'].get('Female', 0)
            female_diff = ae_female_pct - control_female_pct
            
            # Calculate odds ratio for sex
            ae_female_count = ae_analysis['sex']['distribution'].get('Female', 0)
            ae_male_count = ae_analysis['sex']['distribution'].get('Male', 0)
            control_female_count = control_analysis['sex']['distribution'].get('Female', 0)
            control_male_count = control_analysis['sex']['distribution'].get('Male', 0)
            
            if ae_male_count > 0 and control_female_count > 0:
                or_sex = (ae_female_count / ae_male_count) / (control_female_count / control_male_count) if control_male_count > 0 else 0
            else:
                or_sex = 0
            
            risk_factors['sex'] = {
                'ae_group_female_pct': ae_female_pct,
                'control_group_female_pct': control_female_pct,
                'difference': female_diff,
                'odds_ratio': round(or_sex, 2) if or_sex > 0 else None,
                'interpretation': 'Female sex associated with AE' if female_diff > 10 else
                                 'Male sex associated with AE' if female_diff < -10 else
                                 'No significant sex difference',
                'is_risk_factor': abs(female_diff) > 10,
            }
        
        # Medical history comparison (indications)
        if 'top_indications' in ae_analysis and 'top_indications' in control_analysis:
            ae_indications = set(ae_analysis['top_indications'].keys())
            control_indications = set(control_analysis['top_indications'].keys())
            
            # Find indications more common in AE group
            ae_specific_indications = []
            for ind in ae_indications:
                ae_count = ae_analysis['top_indications'].get(ind, 0)
                control_count = control_analysis['top_indications'].get(ind, 0)
                ae_pct = (ae_count / len(ae_reports)) * 100 if len(ae_reports) > 0 else 0
                control_pct = (control_count / len(control_reports)) * 100 if len(control_reports) > 0 else 0
                
                if ae_pct > control_pct + 10:  # At least 10% difference
                    ae_specific_indications.append({
                        'indication': ind,
                        'ae_group_pct': round(ae_pct, 1),
                        'control_group_pct': round(control_pct, 1),
                        'difference': round(ae_pct - control_pct, 1),
                        'is_risk_factor': True,
                    })
            
            if ae_specific_indications:
                risk_factors['medical_history'] = {
                    'risk_indications': sorted(ae_specific_indications, key=lambda x: x['difference'], reverse=True)[:5],
                    'interpretation': 'Specific medical conditions associated with increased AE risk',
                }
        
        # Concomitant drugs comparison
        if 'top_concomitant_drugs' in ae_analysis and 'top_concomitant_drugs' in control_analysis:
            ae_drugs = set(ae_analysis['top_concomitant_drugs'].keys())
            control_drugs = set(control_analysis['top_concomitant_drugs'].keys())
            
            # Find drugs more common in AE group
            risk_drugs = []
            target_drug_lower = drug.lower()
            for concomitant_drug in ae_drugs:
                # Skip the target drug itself
                if concomitant_drug.lower() == target_drug_lower:
                    continue
                ae_count = ae_analysis['top_concomitant_drugs'].get(concomitant_drug, 0)
                control_count = control_analysis['top_concomitant_drugs'].get(concomitant_drug, 0)
                ae_pct = (ae_count / len(ae_reports)) * 100 if len(ae_reports) > 0 else 0
                control_pct = (control_count / len(control_reports)) * 100 if len(control_reports) > 0 else 0
                
                if ae_pct > control_pct + 15:  # At least 15% difference
                    risk_drugs.append({
                        'drug': concomitant_drug,
                        'ae_group_pct': round(ae_pct, 1),
                        'control_group_pct': round(control_pct, 1),
                        'difference': round(ae_pct - control_pct, 1),
                        'is_risk_factor': True,
                    })
            
            if risk_drugs:
                risk_factors['concomitant_drugs'] = {
                    'risk_drugs': sorted(risk_drugs, key=lambda x: x['difference'], reverse=True)[:5],
                    'interpretation': 'Concomitant medications associated with increased AE risk',
                }
        
        return {
            'drug': drug,  # Preserve original drug parameter
            'adverse_event': adverse_event,
            'ae_group_size': len(ae_reports),
            'control_group_size': len(control_reports),
            'ae_analysis': ae_analysis,
            'control_analysis': control_analysis,
            'risk_factors': risk_factors,
        }
    
    def get_bert_embedding(self, text):
        """
        Get BERT embedding for text (for similarity analysis)
        
        Args:
            text: Input text
        
        Returns:
            numpy array with embedding
        """
        if not self.use_bert:
            return None
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
        
        return embedding[0]
    
    def causal_vs_correlated(self, risk_factors):
        """
        Attempt to distinguish causal vs correlated factors
        
        Note: True causal inference requires:
        - Randomized controlled trials
        - Propensity score matching
        - Instrumental variables
        
        This is a simplified heuristic approach.
        
        Args:
            risk_factors: Output from identify_risk_factors
        
        Returns:
            Dictionary with causal assessment
        """
        causal_assessment = {}
        
        for factor, data in risk_factors.get('risk_factors', {}).items():
            if not data.get('is_risk_factor'):
                continue
            
            # Heuristic: Larger effect sizes more likely causal
            # This is a simplification - real causal inference needs more data
            
            if factor == 'age':
                diff = abs(data.get('difference', 0))
                causal_assessment[factor] = {
                    'likely_causal': diff > 10,  # >10 year difference
                    'confidence': 'high' if diff > 15 else 'medium' if diff > 10 else 'low',
                    'mechanism': 'Age-related pharmacokinetic changes' if diff > 10 else 'Unclear',
                }
            
            elif factor == 'sex':
                diff = abs(data.get('difference', 0))
                causal_assessment[factor] = {
                    'likely_causal': diff > 20,  # >20% difference
                    'confidence': 'medium',  # Sex differences often confounded
                    'mechanism': 'Hormonal or metabolic differences' if diff > 20 else 'Unclear',
                }
        
        return causal_assessment


def demo():
    """
    Demonstrate the clinical feature extractor
    """
    print("=" * 70)
    print("Task 3: BERT Clinical Feature Extractor Demo")
    print("=" * 70)
    
    # Initialize (without BERT for demo)
    extractor = ClinicalFeatureExtractor(use_bert=False)
    
    # Example: Analyze Epcoritamab + Cytokine Release Syndrome
    drug = "Epcoritamab"
    adverse_event = "Cytokine release syndrome"
    
    print(f"\nAnalyzing: {drug} + {adverse_event}")
    print("-" * 70)
    
    # Get reports
    reports = extractor.get_reports_for_drug_event(drug, adverse_event, limit=50)
    
    if reports:
        # Extract features
        features = extractor.extract_clinical_features(reports)
        
        # Analyze
        analysis = extractor.analyze_features(features)
        
        print("\nClinical Feature Analysis:")
        print("-" * 40)
        
        if 'age' in analysis:
            print(f"\nAge Distribution:")
            print(f"  Mean: {analysis['age']['mean']:.1f} years")
            print(f"  Median: {analysis['age']['median']:.1f} years")
            print(f"  Range: {analysis['age']['min']:.0f} - {analysis['age']['max']:.0f}")
            print(f"  Age Groups: {analysis['age']['age_groups']}")
        
        if 'sex' in analysis:
            print(f"\nSex Distribution:")
            for sex, pct in analysis['sex']['percentages'].items():
                print(f"  {sex}: {pct:.1f}%")
        
            if 'top_indications' in analysis and analysis['top_indications']:
                print(f"\nMedical History (Indications):")
                for indication, count in list(analysis['top_indications'].items())[:5]:
                    print(f"  - {indication}: {count} reports")
            else:
                print(f"\nMedical History: No indication data available")
            
            if 'top_concomitant_drugs' in analysis and analysis['top_concomitant_drugs']:
                print(f"\nConcomitant Drugs:")
                for concomitant_drug_name, count in list(analysis['top_concomitant_drugs'].items())[:5]:
                    print(f"  - {concomitant_drug_name}: {count} reports")
            
            if 'outcomes' in analysis:
                print(f"\nOutcome Distribution:")
                for outcome, count in analysis['outcomes'].items():
                    print(f"  - {outcome}: {count} reports")
        
        # Risk factor analysis (Causal Inference)
        print("\n" + "=" * 70)
        print("Causal Risk Factor Analysis")
        print("=" * 70)
        print("Comparing AE group vs Control group to identify factors that INFLUENCE/CAUSE the AE")
        print("-" * 70)
        
        risk_analysis = extractor.identify_risk_factors(drug, adverse_event)
        
        print(f"\nGroup Sizes:")
        print(f"  AE Group (with {adverse_event}): {risk_analysis['ae_group_size']} reports")
        print(f"  Control Group (without {adverse_event}): {risk_analysis['control_group_size']} reports")
        
        for factor, data in risk_analysis.get('risk_factors', {}).items():
            print(f"\n{factor.upper().replace('_', ' ')}:")
            
            if factor == 'age':
                print(f"  AE Group Mean Age: {data['ae_group_mean']:.1f} years")
                print(f"  Control Group Mean Age: {data['control_group_mean']:.1f} years")
                print(f"  Difference: {data['difference']:.1f} years")
                print(f"  Interpretation: {data['interpretation']}")
                print(f"  Is Risk Factor: {data['is_risk_factor']}")
            
            elif factor == 'sex':
                print(f"  AE Group Female: {data['ae_group_female_pct']:.1f}%")
                print(f"  Control Group Female: {data['control_group_female_pct']:.1f}%")
                print(f"  Difference: {data['difference']:.1f}%")
                if data.get('odds_ratio'):
                    print(f"  Odds Ratio: {data['odds_ratio']}")
                print(f"  Interpretation: {data['interpretation']}")
                print(f"  Is Risk Factor: {data['is_risk_factor']}")
            
            elif factor == 'medical_history':
                print(f"  {data['interpretation']}")
                print(f"  Risk-Inducing Medical Conditions:")
                for ind in data.get('risk_indications', [])[:5]:
                    print(f"    - {ind['indication']}:")
                    print(f"        AE Group: {ind['ae_group_pct']}% vs Control: {ind['control_group_pct']}%")
                    print(f"        Increased Risk: +{ind['difference']}%")
            
            elif factor == 'concomitant_drugs':
                print(f"  {data['interpretation']}")
                print(f"  Risk-Inducing Concomitant Drugs:")
                for drug_info in data.get('risk_drugs', [])[:5]:
                    print(f"    - {drug_info['drug']}:")
                    print(f"        AE Group: {drug_info['ae_group_pct']}% vs Control: {drug_info['control_group_pct']}%")
                    print(f"        Increased Risk: +{drug_info['difference']}%")
    
    else:
        print("No reports found. Try a different drug-event combination.")


if __name__ == "__main__":
    import sys
    
    # If command-line arguments provided, use them
    if len(sys.argv) == 3:
        drug = sys.argv[1]
        adverse_event = sys.argv[2]
        
        print("=" * 70)
        print(f"Task 3: BERT Clinical Feature Analysis")
        print("=" * 70)
        print(f"\nAnalyzing: {drug} + {adverse_event}")
        print("-" * 70)
        
        extractor = ClinicalFeatureExtractor(use_bert=False)
        reports = extractor.get_reports_for_drug_event(drug, adverse_event, limit=50)
        
        if reports:
            features = extractor.extract_clinical_features(reports)
            analysis = extractor.analyze_features(features)
            
            print("\nClinical Feature Analysis:")
            print("-" * 40)
            
            if 'age' in analysis:
                print(f"\nAge Distribution:")
                print(f"  Mean: {analysis['age']['mean']:.1f} years")
                print(f"  Median: {analysis['age']['median']:.1f} years")
                print(f"  Range: {analysis['age']['min']:.0f} - {analysis['age']['max']:.0f}")
                print(f"  Age Groups: {analysis['age']['age_groups']}")
            
            if 'sex' in analysis:
                print(f"\nSex Distribution:")
                for sex, pct in analysis['sex']['percentages'].items():
                    print(f"  {sex}: {pct:.1f}%")
            
            if 'top_indications' in analysis and analysis['top_indications']:
                print(f"\nMedical History (Indications):")
                for indication, count in list(analysis['top_indications'].items())[:5]:
                    print(f"  - {indication}: {count} reports")
            else:
                print(f"\nMedical History: No indication data available")
            
            if 'top_concomitant_drugs' in analysis and analysis['top_concomitant_drugs']:
                print(f"\nConcomitant Drugs:")
                for concomitant_drug_name, count in list(analysis['top_concomitant_drugs'].items())[:5]:
                    print(f"  - {concomitant_drug_name}: {count} reports")
            
            if 'outcomes' in analysis:
                print(f"\nOutcome Distribution:")
                for outcome, count in analysis['outcomes'].items():
                    print(f"  - {outcome}: {count} reports")
            
            # Causal Risk Factor Analysis
            print("\n" + "=" * 70)
            print("Causal Risk Factor Analysis")
            print("=" * 70)
            print("Comparing AE group vs Control group to identify factors that INFLUENCE/CAUSE the AE")
            print("-" * 70)
            
            risk_analysis = extractor.identify_risk_factors(drug, adverse_event)  # drug parameter preserved
            
            print(f"\nGroup Sizes:")
            print(f"  AE Group (with {adverse_event}): {risk_analysis['ae_group_size']} reports")
            print(f"  Control Group (without {adverse_event}): {risk_analysis['control_group_size']} reports")
            
            for factor, data in risk_analysis.get('risk_factors', {}).items():
                print(f"\n{factor.upper().replace('_', ' ')}:")
                
                if factor == 'age':
                    print(f"  AE Group Mean Age: {data['ae_group_mean']:.1f} years")
                    print(f"  Control Group Mean Age: {data['control_group_mean']:.1f} years")
                    print(f"  Difference: {data['difference']:.1f} years")
                    print(f"  Interpretation: {data['interpretation']}")
                    print(f"  Is Risk Factor: {data['is_risk_factor']}")
                
                elif factor == 'sex':
                    print(f"  AE Group Female: {data['ae_group_female_pct']:.1f}%")
                    print(f"  Control Group Female: {data['control_group_female_pct']:.1f}%")
                    print(f"  Difference: {data['difference']:.1f}%")
                    if data.get('odds_ratio'):
                        print(f"  Odds Ratio: {data['odds_ratio']}")
                    print(f"  Interpretation: {data['interpretation']}")
                    print(f"  Is Risk Factor: {data['is_risk_factor']}")
                
                elif factor == 'medical_history':
                    print(f"  {data['interpretation']}")
                    print(f"  Risk-Inducing Medical Conditions:")
                    for ind in data.get('risk_indications', [])[:5]:
                        print(f"    - {ind['indication']}:")
                        print(f"        AE Group: {ind['ae_group_pct']}% vs Control: {ind['control_group_pct']}%")
                        print(f"        Increased Risk: +{ind['difference']}%")
                
                elif factor == 'concomitant_drugs':
                    print(f"  {data['interpretation']}")
                    print(f"  Risk-Inducing Concomitant Drugs:")
                    for drug_info in data.get('risk_drugs', [])[:5]:
                        print(f"    - {drug_info['drug']}:")
                        print(f"        AE Group: {drug_info['ae_group_pct']}% vs Control: {drug_info['control_group_pct']}%")
                        print(f"        Increased Risk: +{drug_info['difference']}%")
        else:
            print("No reports found. Try a different drug-event combination.")
    
    else:
        # Run multiple examples
        print("=" * 70)
        print("Task 3: BERT Clinical Feature Analysis - Multiple Examples")
        print("=" * 70)
        
        examples = [
            ("Epcoritamab", "Cytokine release syndrome"),
            ("Venetoclax", "Klebsiella sepsis"),
            ("Nivolumab", "Autoimmune hepatitis"),
        ]
        
        extractor = ClinicalFeatureExtractor(use_bert=False)
        
        for drug, adverse_event in examples:
            print(f"\n{'=' * 70}")
            print(f"Example: {drug} + {adverse_event}")
            print("=" * 70)
            
            reports = extractor.get_reports_for_drug_event(drug, adverse_event, limit=30)
            
            if reports:
                features = extractor.extract_clinical_features(reports)
                analysis = extractor.analyze_features(features)
                
                print("\nClinical Feature Summary:")
                if 'age' in analysis:
                    print(f"  Mean Age: {analysis['age']['mean']:.1f} years")
                    print(f"  Age Groups: {analysis['age']['age_groups']}")
                if 'sex' in analysis:
                    print(f"  Sex: {analysis['sex']['percentages']}")
                
                print("\nMedical History (Indications):")
                if 'top_indications' in analysis and analysis['top_indications']:
                    for indication, count in list(analysis['top_indications'].items())[:5]:
                        print(f"    - {indication}: {count} reports")
                else:
                    print("    No indication data available")
                
                print("\nConcomitant Drugs:")
                if 'top_concomitant_drugs' in analysis and analysis['top_concomitant_drugs']:
                    for concomitant_drug_name, count in list(analysis['top_concomitant_drugs'].items())[:5]:
                        print(f"    - {concomitant_drug_name}: {count} reports")
                else:
                    print("    No concomitant drug data available")
                
                print("\nOutcomes:")
                if 'outcomes' in analysis and analysis['outcomes']:
                    for outcome, count in analysis['outcomes'].items():
                        print(f"    - {outcome}: {count} reports")
            else:
                print("  No reports found")
            
            time.sleep(1)  # Rate limiting

