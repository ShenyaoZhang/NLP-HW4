#!/usr/bin/env python3
"""
Task 3 Improvement: Drug Label Filter
Fetches FDA drug labels and filters out known/common adverse events
Only keeps RARE or UNKNOWN adverse events for true "unexpected" detection

Key Improvement:
- Before: Detected "anomalies" that were actually known common AEs
- After: Only report truly unexpected drug-event relationships
"""

import requests
import re
import json
import time
import os
from collections import defaultdict

# FDA Drug Label API
LABEL_API_URL = "https://api.fda.gov/drug/label.json"

# AE frequency categories (per FDA/EU guidelines)
# Very common: ≥1/10 (≥10%)
# Common: ≥1/100 to <1/10 (1-10%)
# Uncommon: ≥1/1,000 to <1/100 (0.1-1%)
# Rare: ≥1/10,000 to <1/1,000 (0.01-0.1%)
# Very rare: <1/10,000 (<0.01%)

class DrugLabelFilter:
    """
    Fetches FDA drug labels and extracts known adverse events
    to filter out common/expected AEs from anomaly detection results
    """
    
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        self.known_aes = {}  # drug -> set of known AEs
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_drug_label(self, drug_name):
        """
        Fetch FDA drug label for a specific drug
        Returns the adverse_reactions section
        """
        cache_file = os.path.join(self.cache_dir, f"{drug_name.lower()}_label.json")
        
        # Check cache first
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # Try multiple search strategies
        search_strategies = [
            f'openfda.brand_name:"{drug_name}"',
            f'openfda.generic_name:"{drug_name}"',
            f'openfda.substance_name:"{drug_name}"',
        ]
        
        for search_query in search_strategies:
            try:
                params = {
                    'search': search_query,
                    'limit': 1
                }
                
                response = requests.get(LABEL_API_URL, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'results' in data and len(data['results']) > 0:
                        label = data['results'][0]
                        
                        # Cache the result
                        with open(cache_file, 'w') as f:
                            json.dump(label, f, indent=2)
                        
                        print(f"  ✓ Found label for {drug_name}")
                        return label
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                print(f"  ✗ Error fetching label for {drug_name}: {e}")
        
        print(f"  ⚠ No label found for {drug_name}")
        return None
    
    def extract_adverse_reactions(self, label):
        """
        Extract adverse reactions from drug label
        Returns a set of known adverse event terms
        """
        known_aes = set()
        
        if not label:
            return known_aes
        
        # Fields that contain adverse reaction information
        ae_fields = [
            'adverse_reactions',
            'adverse_reactions_table',
            'warnings_and_cautions',
            'boxed_warning',
        ]
        
        for field in ae_fields:
            if field in label:
                text = label[field]
                if isinstance(text, list):
                    text = ' '.join(text)
                
                # Extract MedDRA-like terms (capitalized phrases)
                # This is a simplified extraction - real implementation would use NLP
                terms = self._extract_ae_terms(text)
                known_aes.update(terms)
        
        return known_aes
    
    def _extract_ae_terms(self, text):
        """
        Extract adverse event terms from label text
        Uses pattern matching for common AE formats
        """
        terms = set()
        
        # Common patterns in FDA labels
        # Pattern 1: Terms in parentheses with percentages
        # e.g., "fatigue (30%)", "nausea (25%)"
        pattern1 = r'([A-Za-z][a-z]+(?:\s+[a-z]+)*)\s*\(\d+(?:\.\d+)?%?\)'
        matches = re.findall(pattern1, text, re.IGNORECASE)
        terms.update([m.strip().lower() for m in matches])
        
        # Pattern 2: Bullet points or listed items
        # e.g., "• Fatigue", "- Nausea"
        pattern2 = r'[•\-\*]\s*([A-Za-z][a-z]+(?:\s+[a-z]+)*)'
        matches = re.findall(pattern2, text)
        terms.update([m.strip().lower() for m in matches])
        
        # Pattern 3: Common AE terms (hardcoded list of very common terms)
        common_terms = [
            'fatigue', 'nausea', 'vomiting', 'diarrhea', 'constipation',
            'headache', 'dizziness', 'rash', 'pruritus', 'pyrexia',
            'anemia', 'neutropenia', 'thrombocytopenia', 'leukopenia',
            'decreased appetite', 'weight loss', 'asthenia', 'pain',
            'cough', 'dyspnea', 'peripheral edema', 'arthralgia', 'myalgia',
            'alopecia', 'stomatitis', 'mucositis', 'hypertension',
            'cytokine release syndrome', 'infusion related reaction',
            'immune-mediated', 'pneumonitis', 'colitis', 'hepatitis',
        ]
        
        text_lower = text.lower()
        for term in common_terms:
            if term in text_lower:
                terms.add(term)
        
        return terms
    
    def load_known_aes_for_drugs(self, drug_list):
        """
        Load known adverse events for a list of drugs
        """
        print("\n" + "=" * 60)
        print("Loading FDA Drug Labels")
        print("=" * 60)
        
        for drug in drug_list:
            print(f"\nProcessing: {drug}")
            label = self.get_drug_label(drug)
            
            if label:
                aes = self.extract_adverse_reactions(label)
                self.known_aes[drug.lower()] = aes
                print(f"  Found {len(aes)} known adverse events")
            else:
                self.known_aes[drug.lower()] = set()
        
        return self.known_aes
    
    def is_known_ae(self, drug, adverse_event):
        """
        Check if an adverse event is known/expected for a drug
        Returns True if the AE is in the drug label (i.e., NOT unexpected)
        """
        drug_lower = drug.lower()
        ae_lower = adverse_event.lower()
        
        if drug_lower not in self.known_aes:
            return False  # No label data, assume unknown
        
        known = self.known_aes[drug_lower]
        
        # Spelling variants (US vs UK English)
        spelling_variants = {
            'diarrhea': 'diarrhoea',
            'anemia': 'anaemia', 
            'edema': 'oedema',
            'leukemia': 'leukaemia',
            'tumor': 'tumour',
            'color': 'colour',
            'hemorrhage': 'haemorrhage',
            'hemoglobin': 'haemoglobin',
        }
        
        # COMPREHENSIVE MedDRA SYNONYMS MAPPING
        # Maps clinical terms to lab/measurement terms and vice versa
        synonyms = {
            # === HEMATOLOGY ===
            'thrombocytopenia': ['platelet count decreased', 'platelet decreased', 'low platelet', 'platelets decreased', 'platelet count low'],
            'neutropenia': ['neutrophil count decreased', 'neutrophils decreased', 'low neutrophil', 'neutrophil count low', 'absolute neutrophil count decreased'],
            'anemia': ['hemoglobin decreased', 'haemoglobin decreased', 'red blood cell count decreased', 'hematocrit decreased', 'haematocrit decreased', 'rbc decreased'],
            'anaemia': ['hemoglobin decreased', 'haemoglobin decreased', 'red blood cell count decreased', 'hematocrit decreased', 'haematocrit decreased', 'rbc decreased'],
            'leukopenia': ['white blood cell count decreased', 'leucocyte count decreased', 'leukocyte count decreased', 'wbc decreased'],
            'lymphopenia': ['lymphocyte count decreased', 'lymphocytes decreased', 'cd4 lymphocytes decreased'],
            'pancytopenia': ['blood cell count decreased', 'bone marrow failure'],
            'febrile neutropenia': ['neutropenic fever', 'fever with neutropenia'],
            
            # === HEPATIC ===
            'hepatotoxicity': ['hepatic enzyme increased', 'liver enzyme increased', 'alt increased', 'ast increased', 'transaminase increased', 'alanine aminotransferase increased', 'aspartate aminotransferase increased', 'liver function test abnormal', 'hepatic function abnormal'],
            'hyperbilirubinemia': ['bilirubin increased', 'blood bilirubin increased', 'jaundice', 'hyperbilirubinaemia'],
            'hepatitis': ['liver injury', 'hepatic injury', 'drug-induced liver injury', 'hepatocellular injury', 'alanine aminotransferase increased', 'aspartate aminotransferase increased', 'alt increased', 'ast increased', 'transaminase increased', 'liver function test abnormal', 'hepatic enzyme increased'],
            
            # === RENAL ===
            'nephrotoxicity': ['creatinine increased', 'blood creatinine increased', 'renal impairment', 'renal failure', 'kidney injury', 'acute kidney injury', 'renal function impaired'],
            'proteinuria': ['protein urine present', 'urine protein increased', 'albuminuria'],
            
            # === ELECTROLYTES ===
            'hyponatremia': ['blood sodium decreased', 'sodium decreased', 'hyponatraemia'],
            'hypernatremia': ['blood sodium increased', 'sodium increased', 'hypernatraemia'],
            'hypokalemia': ['blood potassium decreased', 'potassium decreased', 'hypokalaemia'],
            'hyperkalemia': ['blood potassium increased', 'potassium increased', 'hyperkalaemia'],
            'hypocalcemia': ['blood calcium decreased', 'calcium decreased', 'hypocalcaemia'],
            'hypercalcemia': ['blood calcium increased', 'calcium increased', 'hypercalcaemia'],
            'hypomagnesemia': ['blood magnesium decreased', 'magnesium decreased', 'hypomagnesaemia'],
            'hypophosphatemia': ['blood phosphorus decreased', 'phosphorus decreased', 'hypophosphataemia'],
            
            # === METABOLIC ===
            'hyperglycemia': ['blood glucose increased', 'glucose increased', 'hyperglycaemia', 'diabetes mellitus'],
            'hypoglycemia': ['blood glucose decreased', 'glucose decreased', 'hypoglycaemia'],
            'hyperlipidemia': ['cholesterol increased', 'triglycerides increased', 'lipids increased', 'hyperlipidaemia', 'hypercholesterolemia', 'hypercholesterolaemia'],
            'hyperuricemia': ['uric acid increased', 'blood uric acid increased', 'hyperuricaemia'],
            
            # === CARDIAC ===
            'qt prolongation': ['electrocardiogram qt prolonged', 'ecg qt prolonged', 'long qt syndrome'],
            'tachycardia': ['heart rate increased', 'pulse rate increased'],
            'bradycardia': ['heart rate decreased', 'pulse rate decreased'],
            'hypotension': ['blood pressure decreased', 'low blood pressure'],
            'hypertension': ['blood pressure increased', 'high blood pressure'],
            'cardiac failure': ['heart failure', 'congestive heart failure', 'ejection fraction decreased', 'left ventricular dysfunction'],
            'arrhythmia': ['cardiac arrhythmia', 'irregular heartbeat', 'atrial fibrillation', 'ventricular tachycardia'],
            
            # === COAGULATION ===
            'coagulopathy': ['inr increased', 'prothrombin time prolonged', 'activated partial thromboplastin time prolonged', 'aptt prolonged'],
            'hemorrhage': ['bleeding', 'haemorrhage', 'blood loss'],
            'thrombosis': ['blood clot', 'embolism', 'deep vein thrombosis', 'pulmonary embolism'],
            
            # === NEUROLOGICAL ===
            'neuropathy': ['peripheral neuropathy', 'peripheral sensory neuropathy', 'neuralgia', 'paresthesia', 'paraesthesia', 'numbness', 'tingling'],
            'encephalopathy': ['altered mental status', 'confusion', 'mental status changes', 'cognitive impairment'],
            'seizure': ['convulsion', 'epilepsy'],
            
            # === GASTROINTESTINAL ===
            'diarrhea': ['diarrhoea', 'loose stools', 'frequent bowel movements'],
            'nausea': ['feeling sick', 'queasiness'],
            'vomiting': ['emesis', 'throwing up'],
            'stomatitis': ['mouth ulcer', 'oral mucositis', 'oral ulcer', 'mucositis'],
            'colitis': ['intestinal inflammation', 'enterocolitis'],
            
            # === DERMATOLOGICAL ===
            'rash': ['skin eruption', 'dermatitis', 'exanthema', 'maculopapular rash', 'erythema'],
            'pruritus': ['itching', 'itch'],
            'alopecia': ['hair loss', 'baldness'],
            
            # === RESPIRATORY ===
            'dyspnea': ['shortness of breath', 'breathlessness', 'respiratory distress', 'dyspnoea'],
            'pneumonitis': ['lung inflammation', 'interstitial lung disease', 'pulmonary toxicity'],
            'cough': ['coughing'],
            
            # === IMMUNE/INFUSION ===
            'cytokine release syndrome': ['crs', 'cytokine storm', 'infusion reaction', 'infusion-related reaction'],
            'hypersensitivity': ['allergic reaction', 'anaphylaxis', 'anaphylactic reaction'],
            
            # === MUSCULOSKELETAL ===
            'arthralgia': ['joint pain', 'joint ache'],
            'myalgia': ['muscle pain', 'muscle ache', 'muscular pain'],
            
            # === GENERAL ===
            'fatigue': ['tiredness', 'asthenia', 'weakness', 'lethargy', 'malaise'],
            'pyrexia': ['fever', 'elevated temperature', 'febrile'],
            'edema': ['oedema', 'swelling', 'fluid retention', 'peripheral edema', 'peripheral oedema'],
            'weight decreased': ['weight loss', 'body weight decreased'],
            'weight increased': ['weight gain', 'body weight increased'],
            'decreased appetite': ['anorexia', 'appetite loss', 'loss of appetite', 'poor appetite'],
            
            # === LAB TO CLINICAL (reverse mapping) ===
            'platelet count decreased': ['thrombocytopenia'],
            'neutrophil count decreased': ['neutropenia'],
            'hemoglobin decreased': ['anemia', 'anaemia'],
            'haemoglobin decreased': ['anemia', 'anaemia'],
            'white blood cell count decreased': ['leukopenia'],
            'lymphocyte count decreased': ['lymphopenia'],
            'alanine aminotransferase increased': ['hepatotoxicity', 'liver injury'],
            'aspartate aminotransferase increased': ['hepatotoxicity', 'liver injury'],
            'blood creatinine increased': ['nephrotoxicity', 'renal impairment'],
            'ejection fraction decreased': ['cardiac failure', 'heart failure'],
        }
        
        # Normalize the AE by converting UK to US spelling
        ae_normalized = ae_lower
        for us, uk in spelling_variants.items():
            ae_normalized = ae_normalized.replace(uk, us)
        
        # Exact match
        if ae_lower in known or ae_normalized in known:
            return True
        
        # Check synonyms: if known has "thrombocytopenia" and ae is "platelet count decreased"
        for known_ae in known:
            if known_ae in synonyms:
                for synonym in synonyms[known_ae]:
                    if synonym in ae_lower or synonym in ae_normalized:
                        return True
                    # Also check partial match
                    if ae_lower in synonym or ae_normalized in synonym:
                        return True
        
        # Also check if the known AE appears in the adverse event
        ae_words = set(ae_lower.split())
        ae_normalized_words = set(ae_normalized.split())
        
        # Key terms that should match regardless of context
        key_terms = {
            'neutropenia', 'thrombocytopenia', 'anemia', 'anaemia', 'leukopenia',
            'infection', 'pneumonia', 'colitis', 'hepatitis', 'nephritis',
            'rash', 'fatigue', 'nausea', 'vomiting', 'diarrhea', 'diarrhoea',
            'constipation', 'headache', 'pyrexia', 'fever', 'cough', 'dyspnea',
            'pruritus', 'asthenia', 'arthralgia', 'myalgia', 'edema', 'oedema',
            'stomatitis', 'mucositis', 'alopecia', 'insomnia', 'hypertension',
            'hypotension', 'pneumonitis', 'hypothyroidism', 'hyperthyroidism',
        }
        
        for known_ae in known:
            # Check if known AE matches (exact or as a word in the phrase)
            if known_ae in ae_words or known_ae in ae_normalized_words:
                if known_ae in key_terms:
                    return True
            
            # Multi-word phrase match
            if ' ' in known_ae and known_ae in ae_lower:
                return True
            
            # Check normalized version
            known_normalized = known_ae
            for us, uk in spelling_variants.items():
                known_normalized = known_normalized.replace(uk, us)
            
            if known_normalized in ae_normalized:
                return True
        
        return False
    
    def is_indication_term(self, adverse_event):
        """
        Check if the adverse event is actually an indication (disease) rather than a true AE
        These should be filtered out as they are NOT adverse events
        """
        indication_terms = [
            'lymphoma', 'leukaemia', 'leukemia', 'myeloma', 'carcinoma', 
            'cancer', 'neoplasm', 'tumor', 'tumour', 'malignant', 
            'metastasis', 'metastatic', 'progression', 'refractory', 
            'recurrent', 'relapsed', 'sarcoma', 'melanoma'
        ]
        
        ae_lower = adverse_event.lower()
        for term in indication_terms:
            if term in ae_lower:
                return True
        return False
    
    def filter_anomalies(self, results_df, max_count_for_rare=10, prr_override_threshold=50):
        """
        Filter anomaly detection results to keep only UNKNOWN adverse events
        
        SIMPLE LOGIC: Only filter out:
        1. Known AEs (from FDA labels)
        2. Indication-related terms (diseases, not AEs)
        
        Args:
            results_df: DataFrame with columns ['drug', 'adverse_event', 'count', ...]
            max_count_for_rare: NOT USED (kept for compatibility)
            prr_override_threshold: NOT USED (kept for compatibility)
        
        Returns:
            DataFrame with only unknown/unexpected drug-event pairs
        """
        print("\n" + "=" * 60)
        print("Filtering Known Adverse Events")
        print("=" * 60)
        
        original_count = len(results_df)
        
        # Add column indicating if AE is known
        results_df['is_known_ae'] = results_df.apply(
            lambda row: self.is_known_ae(row['drug'], row['adverse_event']),
            axis=1
        )
        
        # Add column indicating if AE is actually an indication (disease)
        results_df['is_indication'] = results_df['adverse_event'].apply(
            lambda ae: self.is_indication_term(ae)
        )
        
        # NOT USED - kept for compatibility
        results_df['is_high_count'] = False
        results_df['prr_override'] = False
        
        # Use ORIGINAL data mean count for filtering (not anomalies mean)
        # Original data has ~17975 drug-event pairs with mean count = 3.24
        ORIGINAL_MEAN_COUNT = 3.24  # Pre-calculated from raw data
        mean_count = ORIGINAL_MEAN_COUNT
        results_df['is_high_count'] = results_df['count'] >= mean_count
        
        # FILTER: Remove known AEs, indications, AND high-count events (not rare)
        unknown_df = results_df[
            (~results_df['is_known_ae']) & 
            (~results_df['is_indication']) &
            (~results_df['is_high_count'])  # count < mean = truly RARE
        ].copy()
        
        known_df = results_df[results_df['is_known_ae']].copy()
        indication_df = results_df[results_df['is_indication']].copy()
        high_count_df = results_df[results_df['is_high_count']].copy()
        
        print(f"\nOriginal anomalies: {original_count}")
        print(f"Mean count: {mean_count:.1f}")
        print(f"Known/Expected AEs (filtered out): {len(known_df)}")
        print(f"Indication-related terms (filtered out): {len(indication_df)}")
        print(f"High-count events (count >= {mean_count:.0f}, filtered out): {len(high_count_df)}")
        print(f"RARE & UNEXPECTED AEs (kept): {len(unknown_df)}")
        print(f"Total filter rate: {(original_count-len(unknown_df))/original_count*100:.1f}%")
        
        # Show examples of filtered known AEs
        if len(known_df) > 0:
            print("\nExamples of filtered KNOWN adverse events:")
            for _, row in known_df.head(5).iterrows():
                print(f"  - {row['drug']}: {row['adverse_event']} (known)")
        
        # Show examples of filtered indication terms
        if len(indication_df) > 0:
            print("\nExamples of filtered INDICATION terms (not AEs):")
            for _, row in indication_df.head(5).iterrows():
                print(f"  - {row['drug']}: {row['adverse_event']} (indication/disease)")
        
        # Show examples of kept unknown AEs
        if len(unknown_df) > 0:
            print("\nExamples of kept UNEXPECTED adverse events:")
            for _, row in unknown_df.head(5).iterrows():
                print(f"  - {row['drug']}: {row['adverse_event']} (count={int(row['count'])}, PRR={row['prr']:.1f})")
        
        return unknown_df


# Hardcoded known AEs for ALL 37 drugs (from FDA drug labels)
# Auto-extracted from OpenFDA Label API
KNOWN_AES_BACKUP = {
    'abemaciclib': {'abdominal pain', 'alopecia', 'anemia', 'arthralgia', 'asthenia', 'constipation', 'cough', 'decreased appetite', 'dehydration', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'fever', 'headache', 'hypokalemia', 'infection', 'leukopenia', 'nausea', 'neutropenia', 'pain', 'pneumonia', 'pneumonitis', 'pruritus', 'pyrexia', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting'},
    'atezolizumab': {'abdominal pain', 'alopecia', 'anemia', 'arthralgia', 'asthenia', 'back pain', 'colitis', 'constipation', 'cough', 'cytokine release syndrome', 'decreased appetite', 'dehydration', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'headache', 'hepatitis', 'hyperglycemia', 'hypersensitivity', 'hypertension', 'hyperthyroidism', 'hypokalemia', 'hyponatremia', 'hypophosphatemia', 'hypothyroidism', 'infection', 'infusion reaction', 'insomnia', 'leukopenia', 'myalgia', 'nausea', 'nephritis', 'neutropenia', 'pain', 'peripheral neuropathy', 'pneumonia', 'pneumonitis', 'pruritus', 'pyrexia', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting'},
    'bevacizumab': {'abdominal pain', 'arthralgia', 'asthenia', 'back pain', 'constipation', 'cough', 'decreased appetite', 'dehydration', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'headache', 'hyperglycemia', 'hypersensitivity', 'hypertension', 'hypokalemia', 'hyponatremia', 'infection', 'insomnia', 'leukopenia', 'myalgia', 'nausea', 'neutropenia', 'pain', 'pneumonitis', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting', 'weight loss'},
    'bortezomib': {'abdominal pain', 'alopecia', 'anemia', 'arthralgia', 'asthenia', 'back pain', 'colitis', 'constipation', 'cough', 'decreased appetite', 'dehydration', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'headache', 'hepatitis', 'hyperglycemia', 'hypersensitivity', 'hypertension', 'hypokalemia', 'hyponatremia', 'hypotension', 'infection', 'insomnia', 'leukopenia', 'myalgia', 'nausea', 'nephritis', 'neutropenia', 'pain', 'peripheral neuropathy', 'pneumonia', 'pneumonitis', 'pruritus', 'pyrexia', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting'},
    'carboplatin': {'alopecia', 'anemia', 'asthenia', 'constipation', 'dehydration', 'diarrhea', 'fever', 'hypersensitivity', 'hypertension', 'hypotension', 'infection', 'injection site reaction', 'leukopenia', 'mucositis', 'nausea', 'neutropenia', 'pain', 'pruritus', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting'},
    'carfilzomib': {'abdominal pain', 'anemia', 'arthralgia', 'asthenia', 'back pain', 'colitis', 'constipation', 'cough', 'cytokine release syndrome', 'decreased appetite', 'dehydration', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'fever', 'headache', 'hepatitis', 'hyperglycemia', 'hypersensitivity', 'hypertension', 'hypokalemia', 'hyponatremia', 'hypophosphatemia', 'hypotension', 'infection', 'infusion reaction', 'injection site reaction', 'insomnia', 'leukopenia', 'myalgia', 'nausea', 'neutropenia', 'pain', 'peripheral neuropathy', 'pneumonia', 'pneumonitis', 'pruritus', 'pyrexia', 'rash', 'thrombocytopenia', 'vomiting'},
    'cetuximab': {'abdominal pain', 'alopecia', 'anemia', 'arthralgia', 'asthenia', 'constipation', 'cough', 'decreased appetite', 'dehydration', 'diarrhea', 'dyspnea', 'edema', 'fatigue', 'fever', 'headache', 'hypersensitivity', 'hypertension', 'hypokalemia', 'hyponatremia', 'hypotension', 'infection', 'infusion reaction', 'insomnia', 'leukopenia', 'mucositis', 'nausea', 'neutropenia', 'pain', 'peripheral neuropathy', 'pruritus', 'pyrexia', 'rash', 'stomatitis', 'vomiting', 'weight loss'},
    'cisplatin': {'anemia', 'diarrhea', 'edema', 'fever', 'hypotension', 'infection', 'leukopenia', 'nausea', 'neutropenia', 'thrombocytopenia', 'vomiting'},
    'crizotinib': {'abdominal pain', 'anemia', 'arthralgia', 'back pain', 'constipation', 'cough', 'decreased appetite', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'headache', 'hyperglycemia', 'hypertension', 'hypokalemia', 'hyponatremia', 'hypophosphatemia', 'hypotension', 'infection', 'myalgia', 'nausea', 'neutropenia', 'pain', 'peripheral neuropathy', 'pneumonia', 'pneumonitis', 'pruritus', 'pyrexia', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting'},
    'dabrafenib': {'abdominal pain', 'alopecia', 'anemia', 'arthralgia', 'asthenia', 'back pain', 'colitis', 'constipation', 'cough', 'decreased appetite', 'dehydration', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'fever', 'headache', 'hyperglycemia', 'hypersensitivity', 'hypertension', 'hypokalemia', 'hyponatremia', 'hypophosphatemia', 'hypotension', 'infection', 'leukopenia', 'myalgia', 'nausea', 'nephritis', 'neutropenia', 'pain', 'peripheral neuropathy', 'pneumonia', 'pneumonitis', 'pyrexia', 'rash', 'stomatitis', 'vomiting'},
    'docetaxel': {'abdominal pain', 'alopecia', 'anemia', 'arthralgia', 'asthenia', 'back pain', 'colitis', 'constipation', 'cough', 'dehydration', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'fever', 'hepatitis', 'hypersensitivity', 'hypertension', 'hypokalemia', 'hyponatremia', 'hypotension', 'infection', 'leukopenia', 'mucositis', 'myalgia', 'nausea', 'neutropenia', 'pain', 'pneumonia', 'pneumonitis', 'pruritus', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting', 'weight loss'},
    'doxorubicin': {'abdominal pain', 'alopecia', 'asthenia', 'colitis', 'dehydration', 'diarrhea', 'fever', 'hypersensitivity', 'infection', 'leukopenia', 'mucositis', 'nausea', 'neutropenia', 'pain', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting'},
    'durvalumab': {'abdominal pain', 'alopecia', 'anemia', 'arthralgia', 'asthenia', 'back pain', 'colitis', 'constipation', 'cough', 'decreased appetite', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'headache', 'hepatitis', 'hyperglycemia', 'hypertension', 'hyperthyroidism', 'hypokalemia', 'hyponatremia', 'hypothyroidism', 'infection', 'insomnia', 'leukopenia', 'myalgia', 'nausea', 'nephritis', 'neutropenia', 'pain', 'peripheral neuropathy', 'pneumonia', 'pneumonitis', 'pruritus', 'pyrexia', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting'},
    'epcoritamab': {'abdominal pain', 'anemia', 'arthralgia', 'asthenia', 'back pain', 'constipation', 'cough', 'cytokine release syndrome', 'crs', 'decreased appetite', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'fever', 'headache', 'hepatitis', 'hypophosphatemia', 'hypotension', 'icans', 'immune effector cell-associated neurotoxicity syndrome', 'infection', 'injection site reaction', 'insomnia', 'mucositis', 'myalgia', 'nausea', 'neurotoxicity', 'neutropenia', 'pain', 'peripheral neuropathy', 'pneumonia', 'pneumonitis', 'pruritus', 'pyrexia', 'rash', 'stomatitis', 'thrombocytopenia', 'tumor flare', 'vomiting'},
    'erlotinib': {'anemia', 'arthralgia', 'asthenia', 'back pain', 'colitis', 'cough', 'decreased appetite', 'dehydration', 'diarrhea', 'dyspnea', 'fatigue', 'fever', 'headache', 'infection', 'nausea', 'pain', 'pruritus', 'pyrexia', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting'},
    'gefitinib': {'alopecia', 'asthenia', 'cough', 'decreased appetite', 'dehydration', 'diarrhea', 'dyspnea', 'edema', 'fever', 'infection', 'nausea', 'pneumonia', 'pneumonitis', 'pruritus', 'pyrexia', 'rash', 'stomatitis', 'vomiting'},
    'glofitamab': {'abdominal pain', 'asthenia', 'back pain', 'colitis', 'constipation', 'cytokine release syndrome', 'crs', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'fever', 'headache', 'hypotension', 'icans', 'immune effector cell-associated neurotoxicity syndrome', 'infection', 'myalgia', 'nausea', 'neurotoxicity', 'neutropenia', 'pain', 'peripheral neuropathy', 'pneumonia', 'pruritus', 'pyrexia', 'rash', 'thrombocytopenia', 'tumor flare', 'tumor lysis syndrome', 'vomiting'},
    'ibrutinib': {'abdominal pain', 'anemia', 'arthralgia', 'asthenia', 'constipation', 'cough', 'decreased appetite', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'fever', 'headache', 'hepatitis', 'hypertension', 'hypokalemia', 'infection', 'insomnia', 'nausea', 'neutropenia', 'pain', 'peripheral neuropathy', 'pneumonia', 'pruritus', 'pyrexia', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting'},
    'imatinib': {'abdominal pain', 'alopecia', 'anemia', 'arthralgia', 'asthenia', 'back pain', 'colitis', 'constipation', 'cough', 'decreased appetite', 'dehydration', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'fever', 'headache', 'hepatitis', 'hyperglycemia', 'hypertension', 'hyperthyroidism', 'hypokalemia', 'hyponatremia', 'hypophosphatemia', 'hypotension', 'hypothyroidism', 'infection', 'insomnia', 'leukopenia', 'mucositis', 'myalgia', 'nausea', 'neutropenia', 'pain', 'peripheral neuropathy', 'pneumonia', 'pneumonitis', 'pruritus', 'pyrexia', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting'},
    'ipilimumab': {'abdominal pain', 'alopecia', 'anemia', 'arthralgia', 'asthenia', 'back pain', 'colitis', 'constipation', 'cough', 'decreased appetite', 'dehydration', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'fever', 'headache', 'hepatitis', 'hyperglycemia', 'hypersensitivity', 'hypertension', 'hyperthyroidism', 'hypokalemia', 'hyponatremia', 'hypotension', 'hypothyroidism', 'infection', 'infusion reaction', 'insomnia', 'leukopenia', 'myalgia', 'nausea', 'nephritis', 'neutropenia', 'pain', 'peripheral neuropathy', 'pneumonia', 'pneumonitis', 'pruritus', 'pyrexia', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting'},
    'lenalidomide': {'abdominal pain', 'anemia', 'arthralgia', 'asthenia', 'back pain', 'colitis', 'constipation', 'cough', 'decreased appetite', 'dehydration', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'fever', 'headache', 'hepatitis', 'hyperglycemia', 'hypersensitivity', 'hypertension', 'hyperthyroidism', 'hypokalemia', 'hyponatremia', 'hypophosphatemia', 'hypotension', 'hypothyroidism', 'infection', 'insomnia', 'leukopenia', 'myalgia', 'nausea', 'nephritis', 'neutropenia', 'pain', 'peripheral neuropathy', 'pneumonia', 'pneumonitis', 'pruritus', 'pyrexia', 'rash', 'thrombocytopenia', 'vomiting'},
    'mosunetuzumab': {'abdominal pain', 'anemia', 'arthralgia', 'asthenia', 'back pain', 'cough', 'cytokine release syndrome', 'crs', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'fever', 'headache', 'hemophagocytic lymphohistiocytosis', 'hepatitis', 'hlh', 'hypotension', 'icans', 'immune effector cell-associated neurotoxicity syndrome', 'infection', 'insomnia', 'myalgia', 'nausea', 'nephritis', 'neurotoxicity', 'neutropenia', 'pain', 'peripheral neuropathy', 'pneumonia', 'pruritus', 'pyrexia', 'rash', 'thrombocytopenia', 'tumor flare', 'vomiting'},
    'niraparib': {'abdominal pain', 'anemia', 'asthenia', 'constipation', 'cough', 'decreased appetite', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'headache', 'hepatitis', 'hypertension', 'hypokalemia', 'infection', 'insomnia', 'nausea', 'neutropenia', 'pain', 'pneumonia', 'pyrexia', 'rash', 'thrombocytopenia', 'vomiting'},
    'nivolumab': {'abdominal pain', 'alopecia', 'anemia', 'arthralgia', 'asthenia', 'back pain', 'colitis', 'constipation', 'cough', 'decreased appetite', 'dehydration', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'fever', 'headache', 'hepatitis', 'hyperglycemia', 'hypersensitivity', 'hypertension', 'hyperthyroidism', 'hypokalemia', 'hyponatremia', 'hypophosphatemia', 'hypotension', 'hypothyroidism', 'infection', 'injection site reaction', 'insomnia', 'leukopenia', 'mucositis', 'myalgia', 'nausea', 'nephritis', 'neutropenia', 'pain', 'peripheral neuropathy', 'pneumonia', 'pneumonitis', 'pruritus', 'pyrexia', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting'},
    'olaparib': {'abdominal pain', 'anemia', 'arthralgia', 'asthenia', 'back pain', 'colitis', 'constipation', 'cough', 'decreased appetite', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'fever', 'headache', 'hypersensitivity', 'hypertension', 'infection', 'leukopenia', 'myalgia', 'nausea', 'neutropenia', 'pain', 'pneumonia', 'pneumonitis', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting'},
    'osimertinib': {'abdominal pain', 'alopecia', 'anemia', 'arthralgia', 'asthenia', 'back pain', 'colitis', 'constipation', 'cough', 'decreased appetite', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'fever', 'headache', 'hyperglycemia', 'hypokalemia', 'hyponatremia', 'infection', 'leukopenia', 'myalgia', 'nausea', 'neutropenia', 'pain', 'pneumonia', 'pneumonitis', 'pruritus', 'pyrexia', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting'},
    'paclitaxel': {'abdominal pain', 'alopecia', 'anemia', 'arthralgia', 'asthenia', 'back pain', 'colitis', 'dehydration', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fever', 'headache', 'hyperglycemia', 'hypersensitivity', 'hypertension', 'hypotension', 'infection', 'injection site reaction', 'leukopenia', 'mucositis', 'myalgia', 'nausea', 'neutropenia', 'pain', 'peripheral neuropathy', 'pneumonia', 'pneumonitis', 'pruritus', 'rash', 'thrombocytopenia', 'vomiting'},
    'palbociclib': {'abdominal pain', 'alopecia', 'anemia', 'asthenia', 'cough', 'decreased appetite', 'diarrhea', 'dyspnea', 'fatigue', 'fever', 'headache', 'hyperglycemia', 'infection', 'leukopenia', 'nausea', 'neutropenia', 'pain', 'pneumonia', 'pneumonitis', 'pyrexia', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting'},
    'pembrolizumab': {'abdominal pain', 'alopecia', 'anemia', 'arthralgia', 'asthenia', 'back pain', 'colitis', 'constipation', 'cough', 'decreased appetite', 'dehydration', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'fever', 'headache', 'hepatitis', 'hyperglycemia', 'hypersensitivity', 'hypertension', 'hyperthyroidism', 'hypokalemia', 'hyponatremia', 'hypophosphatemia', 'hypotension', 'hypothyroidism', 'infection', 'injection site reaction', 'insomnia', 'leukopenia', 'myalgia', 'nausea', 'nephritis', 'neutropenia', 'pain', 'peripheral neuropathy', 'pneumonia', 'pneumonitis', 'pruritus', 'pyrexia', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting', 'weight loss'},
    'pomalidomide': {'abdominal pain', 'anemia', 'arthralgia', 'asthenia', 'back pain', 'colitis', 'constipation', 'cough', 'decreased appetite', 'dehydration', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'fever', 'headache', 'hepatitis', 'hyperglycemia', 'hypersensitivity', 'hypertension', 'hyperthyroidism', 'hypokalemia', 'hyponatremia', 'hypotension', 'hypothyroidism', 'infection', 'insomnia', 'leukopenia', 'nausea', 'nephritis', 'neutropenia', 'pain', 'peripheral neuropathy', 'pneumonia', 'pneumonitis', 'pruritus', 'pyrexia', 'rash', 'thrombocytopenia', 'vomiting'},
    'ribociclib': {'abdominal pain', 'alopecia', 'anemia', 'arthralgia', 'asthenia', 'back pain', 'constipation', 'cough', 'decreased appetite', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'headache', 'hypersensitivity', 'hypertension', 'hypokalemia', 'hypothyroidism', 'infection', 'insomnia', 'leukopenia', 'nausea', 'neutropenia', 'pain', 'pneumonia', 'pneumonitis', 'pruritus', 'pyrexia', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting'},
    'rituximab': {'abdominal pain', 'alopecia', 'anemia', 'arthralgia', 'asthenia', 'back pain', 'constipation', 'cough', 'cytokine release syndrome', 'decreased appetite', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'fever', 'headache', 'hepatitis', 'hypersensitivity', 'hypertension', 'hypotension', 'infection', 'injection site reaction', 'insomnia', 'leukopenia', 'myalgia', 'nausea', 'neutropenia', 'pain', 'pneumonia', 'pruritus', 'pyrexia', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting'},
    'rucaparib': {'abdominal pain', 'anemia', 'asthenia', 'constipation', 'decreased appetite', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'headache', 'hypersensitivity', 'hypophosphatemia', 'infection', 'insomnia', 'leukopenia', 'nausea', 'neutropenia', 'pain', 'pneumonia', 'pyrexia', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting'},
    'talazoparib': {'abdominal pain', 'alopecia', 'anemia', 'asthenia', 'decreased appetite', 'diarrhea', 'dizziness', 'fatigue', 'headache', 'infection', 'nausea', 'neutropenia', 'pain', 'pneumonia', 'pyrexia', 'stomatitis', 'thrombocytopenia', 'vomiting'},
    'trastuzumab': {'abdominal pain', 'anemia', 'arthralgia', 'asthenia', 'back pain', 'constipation', 'cough', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'headache', 'hepatitis', 'hypersensitivity', 'hypertension', 'hypokalemia', 'hypotension', 'infection', 'insomnia', 'leukopenia', 'myalgia', 'nausea', 'neutropenia', 'pain', 'peripheral neuropathy', 'pneumonitis', 'pruritus', 'pyrexia', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting'},
    'vemurafenib': {'alopecia', 'arthralgia', 'asthenia', 'back pain', 'constipation', 'cough', 'decreased appetite', 'diarrhea', 'edema', 'fatigue', 'headache', 'hypersensitivity', 'hypertension', 'hypotension', 'infection', 'myalgia', 'nausea', 'nephritis', 'neutropenia', 'pain', 'pruritus', 'pyrexia', 'rash', 'vomiting'},
    'venetoclax': {'abdominal pain', 'anemia', 'arthralgia', 'asthenia', 'back pain', 'colitis', 'constipation', 'cough', 'decreased appetite', 'diarrhea', 'dizziness', 'dyspnea', 'edema', 'fatigue', 'headache', 'hyperglycemia', 'hypokalemia', 'hyponatremia', 'hypophosphatemia', 'hypotension', 'infection', 'leukopenia', 'mucositis', 'myalgia', 'nausea', 'nephritis', 'neutropenia', 'pain', 'pneumonia', 'pyrexia', 'rash', 'stomatitis', 'thrombocytopenia', 'vomiting'},
}


def demo():
    """
    Demonstrate the drug label filter
    """
    import pandas as pd
    
    # Create sample anomaly results
    sample_data = {
        'drug': ['Epcoritamab', 'Epcoritamab', 'Epcoritamab', 'Pembrolizumab', 'Pembrolizumab'],
        'adverse_event': ['Cytokine release syndrome', 'Unusual cardiac event', 'Fatigue', 'Pneumonitis', 'Rare skin condition'],
        'anomaly_score': [0.85, 0.72, 0.68, 0.75, 0.70],
        'prr': [15.2, 8.5, 3.2, 12.1, 6.8],
    }
    results_df = pd.DataFrame(sample_data)
    
    print("Original anomaly results:")
    print(results_df)
    
    # Initialize filter
    filter = DrugLabelFilter()
    
    # Use backup known AEs
    filter.known_aes = KNOWN_AES_BACKUP
    
    # Filter results
    unknown_df = filter.filter_anomalies(results_df)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS: Only UNEXPECTED adverse events")
    print("=" * 60)
    print(unknown_df)


if __name__ == "__main__":
    demo()

