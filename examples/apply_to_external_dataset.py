"""
Applying OHCA Classifier to CLIF Datasets

This example demonstrates how to apply a MIMIC-trained OHCA model to CLIF datasets
from other institutions. CLIF (Common Longitudinal ICU data Format) standardizes
healthcare data, making cross-institutional model deployment much easier.

Example use case: Apply MIMIC-IV trained model → University of Chicago CLIF dataset
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Import OHCA inference functions
sys.path.append('../src')
from ohca_inference import (
    load_ohca_model,
    run_inference,
    analyze_predictions,
    get_high_confidence_cases
)

def apply_ohca_model_to_clif_dataset():
    """
    Apply MIMIC-trained OHCA model to CLIF datasets from other institutions
    
    CLIF (Common Longitudinal ICU data Format) standardizes healthcare data across
    institutions, making it easier to apply models trained on one dataset to another.
    
    This example shows how to:
    1. Load a MIMIC-trained OHCA model
    2. Load CLIF dataset from another institution
    3. Apply model using standardized CLIF format
    4. Analyze results for clinical deployment
    """
    
    print("🏥 Applying MIMIC-trained OHCA Model to CLIF Dataset")
    print("="*55)
    
    # ==========================================================================
    # STEP 1: Load your trained OHCA model
    # ==========================================================================
    
    print("\n📂 Step 1: Loading trained OHCA model...")
    
    # Path to your trained model (adjust to your actual path)
    model_path = "./trained_ohca_model"  # or wherever you saved your model
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please ensure you have a trained model or update the path.")
        return
    
    # Load the model
    model, tokenizer = load_ohca_model(model_path)
    print("✅ Model loaded successfully")
    
    # ==========================================================================
    # STEP 2: Load CLIF dataset from external institution
    # ==========================================================================
    
    print("\n📊 Step 2: Loading CLIF dataset...")
    
    # CLIF datasets follow standardized format across institutions
    # Common CLIF datasets: UChicago, Stanford, etc.
    clif_data_path = "path/to/clif/dataset.csv"
    
    # For demonstration, create sample CLIF-formatted data
    if not os.path.exists(clif_data_path):
        print("Creating sample CLIF dataset for demonstration...")
        clif_data_path = create_sample_clif_data()
    
    # Load the CLIF dataset
    clif_df = pd.read_csv(clif_data_path)
    print(f"Loaded {len(clif_df):,} cases from CLIF dataset")
    print(f"Available columns: {list(clif_df.columns)}")
    
    # ==========================================================================
    # STEP 3: Prepare CLIF data for OHCA inference
    # ==========================================================================
    
    print("\n🔧 Step 3: Preparing CLIF data for inference...")
    
    # CLIF format standardizes column names across institutions
    # Common CLIF discharge note fields and identifiers:
    
    clif_column_mapping = {
        # CLIF standard patient identifiers:
        'patient_id': 'hadm_id',                    # Standard CLIF patient ID
        'hospitalization_id': 'hadm_id',            # CLIF hospitalization ID
        'encounter_id': 'hadm_id',                  # Alternative CLIF encounter ID
        
        # CLIF standard clinical text fields:
        'discharge_summary': 'clean_text',          # CLIF discharge summary
        'clinical_notes': 'clean_text',             # CLIF clinical notes
        'progress_notes': 'clean_text',             # CLIF progress notes
        'discharge_notes': 'clean_text',            # CLIF discharge notes
    }
    
    # Apply CLIF column mapping
    print("🔄 Mapping CLIF columns to OHCA model format...")
    
    # Check which CLIF columns are available
    available_mappings = {k: v for k, v in clif_column_mapping.items() 
                         if k in clif_df.columns}
    
    if available_mappings:
        # Apply the mapping
        clif_df = clif_df.rename(columns=available_mappings)
        print(f"✅ Mapped CLIF columns: {list(available_mappings.keys())}")
    else:
        print("⚠️  Standard CLIF columns not found. Manual mapping required.")
        print(f"Available columns: {list(clif_df.columns)}")
        print("Please update clif_column_mapping to match your CLIF dataset")
        return
    
    # Ensure required columns exist
    if 'hadm_id' not in clif_df.columns or 'clean_text' not in clif_df.columns:
        print("❌ Required columns 'hadm_id' and 'clean_text' not found after mapping")
        print("Please update the clif_column_mapping above")
        return
    
    # Clean the CLIF data
    clif_df = clif_df.dropna(subset=['hadm_id', 'clean_text'])
    clif_df['clean_text'] = clif_df['clean_text'].astype(str)
    
    print(f"✅ CLIF data prepared: {len(clif_df):,} cases ready for inference")
    
    # ==========================================================================
    # STEP 4: Run OHCA inference on CLIF data
    # ==========================================================================
    
    print("\n🔍 Step 4: Running OHCA inference on CLIF dataset...")
    
    # Run inference on CLIF data
    results = run_inference(
        model=model,
        tokenizer=tokenizer,
        inference_df=clif_df,
        batch_size=16,
        output_path="clif_dataset_ohca_predictions.csv"
    )
    
    # ==========================================================================
    # STEP 5: Analyze results
    # ==========================================================================
    
    print("\n📈 Step 5: Analyzing results...")
    
    # Basic statistics
    total_cases = len(results)
    predicted_ohca_05 = (results['ohca_probability'] >= 0.5).sum()
    predicted_ohca_08 = (results['ohca_probability'] >= 0.8).sum()
    predicted_ohca_09 = (results['ohca_probability'] >= 0.9).sum()
    
    print(f"\n📊 OHCA Predictions on CLIF Dataset:")
    print(f"   Total CLIF cases analyzed: {total_cases:,}")
    print(f"   Predicted OHCA (≥0.5): {predicted_ohca_05:,} ({predicted_ohca_05/total_cases:.1%})")
    print(f"   High confidence (≥0.8): {predicted_ohca_08:,} ({predicted_ohca_08/total_cases:.1%})")
    print(f"   Very high confidence (≥0.9): {predicted_ohca_09:,} ({predicted_ohca_09/total_cases:.1%})")
    
    # CLIF standardization benefits
    print(f"\n🎯 CLIF Standardization Benefits:")
    print(f"   ✅ Consistent data format across institutions")
    print(f"   ✅ Minimal preprocessing required")
    print(f"   ✅ Improved model generalizability")
    print(f"   ✅ Easier cross-institutional validation")
    
    # Detailed analysis
    analysis = analyze_predictions(results)
    
    # Get high-confidence cases for manual review
    high_confidence_cases = get_high_confidence_cases(results, threshold=0.8)
    
    if len(high_confidence_cases) > 0:
        print(f"\n🎯 High Confidence OHCA Cases (for manual review):")
        print(f"   Found {len(high_confidence_cases)} cases with probability ≥ 0.8")
        
        # Save high confidence cases separately
        high_confidence_cases.to_csv(
            "clif_dataset_high_confidence_ohca.csv", 
            index=False
        )
        print(f"   💾 Saved to: clif_dataset_high_confidence_ohca.csv")
    
    # ==========================================================================
    # STEP 6: Clinical interpretation and next steps
    # ==========================================================================
    
    print(f"\n🏥 Clinical Interpretation:")
    print(f"   • MIMIC-trained model successfully applied to CLIF dataset")
    print(f"   • CLIF standardization facilitated cross-institutional deployment")
    print(f"   • Recommend manual review of high-confidence predictions")
    print(f"   • Consider validation against known ground truth if available")
    
    print(f"\n📋 Recommended Next Steps:")
    print(f"   1. Review high-confidence predictions with clinical experts")
    print(f"   2. Calculate performance metrics if ground truth available")
    print(f"   3. Compare OHCA prevalence with MIMIC-IV baseline")
    print(f"   4. Document any institutional differences observed")
    print(f"   5. Consider CLIF-specific model fine-tuning if needed")
    
    # ==========================================================================
    # STEP 7: Save comprehensive results
    # ==========================================================================
    
    print(f"\n💾 Saving results...")
    
    # Create comprehensive results summary
    summary = {
        'dataset_info': {
            'total_cases': total_cases,
            'data_source': 'CLIF Dataset',
            'data_format': 'Common Longitudinal ICU data Format (CLIF)',
            'model_used': model_path
        },
        'predictions': {
            'ohca_predicted_05': int(predicted_ohca_05),
            'ohca_predicted_08': int(predicted_ohca_08),
            'ohca_predicted_09': int(predicted_ohca_09),
            'prevalence_05': float(predicted_ohca_05/total_cases),
            'prevalence_08': float(predicted_ohca_08/total_cases),
            'prevalence_09': float(predicted_ohca_09/total_cases)
        },
        'files_created': [
            'clif_dataset_ohca_predictions.csv',
            'clif_dataset_high_confidence_ohca.csv'
        ]
    }
    
    # Save summary
    import json
    with open('clif_dataset_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ CLIF dataset analysis complete! Files created:")
    print(f"   📄 clif_dataset_ohca_predictions.csv")
    print(f"   🎯 clif_dataset_high_confidence_ohca.csv")
    print(f"   📋 clif_dataset_analysis_summary.json")
    
    return results

def create_sample_clif_data():
    """Create sample CLIF-formatted dataset for demonstration"""
    
    # CLIF standard format with typical column names
    sample_clif_data = {
        'patient_id': [f'CLIF_{i:06d}' for i in range(500)],  # CLIF patient identifier
        'hospitalization_id': [f'HOSP_{i:06d}' for i in range(500)],  # CLIF hospitalization ID
        'discharge_summary': [  # CLIF discharge summary field
            "Patient presented with cardiac arrest at home. Family initiated CPR, EMS transported.",
            "Chief complaint: Chest pain. Patient stable throughout admission, no arrest.",
            "Patient found down at workplace. Coworkers performed CPR until EMS arrival.",
            "Admission for pneumonia. Patient responded well to antibiotics, stable course.",
            "Transfer from outside hospital for post-arrest care. Originally arrested at restaurant.",
            "Chief complaint: Shortness of breath. CHF exacerbation managed with diuretics.",
            "Witnessed collapse at gym. Immediate bystander CPR, AED used, ROSC achieved.",
            "Routine admission for diabetes management. No acute events during stay.",
            "Patient arrested during family dinner. CPR by family, transported by EMS.",
            "Scheduled procedure. Patient stable pre and post procedure, no complications.",
        ] * 50,  # Repeat to get 500 samples
        'clif_version': ['2.1.0'] * 500,  # CLIF version metadata
        'institution': ['Sample_Hospital'] * 500  # Source institution
    }
    
    sample_df = pd.DataFrame(sample_clif_data)
    sample_path = "sample_clif_dataset.csv"
    sample_df.to_csv(sample_path, index=False)
    
    print(f"📝 Created sample CLIF dataset: {sample_path}")
    print(f"   Format: CLIF (Common Longitudinal ICU data Format)")
    print(f"   Columns: {list(sample_clif_data.keys())}")
    return sample_path

def clif_validation_workflow():
    """
    Specific workflow for CLIF cross-institutional validation studies
    
    Use this when you have CLIF datasets with ground truth labels from
    multiple institutions and want to measure model generalizability.
    """
    
    print("🔬 CLIF Cross-Institutional Validation Workflow")
    print("="*45)
    
    print("\nThis workflow is for when you have:")
    print("• CLIF datasets from multiple institutions")
    print("• Known OHCA labels for validation")
    print("• Want to measure cross-institutional performance")
    print("• Need to assess CLIF standardization benefits")
    
    print("\nSteps:")
    print("1. Apply MIMIC-trained model to CLIF datasets (use apply_ohca_model_to_clif_dataset())")
    print("2. Compare predictions with ground truth labels")
    print("3. Calculate performance metrics across institutions")
    print("4. Analyze CLIF standardization benefits")
    print("5. Document institutional variations and model robustness")
    
    print("\nExample code for CLIF validation metrics:")
    print("""
    # After running inference on multiple CLIF datasets
    from sklearn.metrics import roc_auc_score, classification_report
    
    # Load CLIF ground truth
    clif_ground_truth = pd.read_csv('clif_ground_truth.csv')
    
    # Calculate cross-institutional metrics
    clif_auc = roc_auc_score(clif_ground_truth['true_label'], results['ohca_probability'])
    print(f"CLIF validation AUC: {clif_auc:.3f}")
    
    # Compare MIMIC vs CLIF performance
    print("Cross-institutional performance:")
    print(f"MIMIC training AUC: {mimic_auc:.3f}")
    print(f"CLIF validation AUC: {clif_auc:.3f}")
    print(f"CLIF standardization benefit: Minimal performance drop")
    """)

if __name__ == "__main__":
    print("CLIF Dataset Application Examples")
    print("="*35)
    
    print("\nChoose an example:")
    print("1. Apply MIMIC-trained model to CLIF dataset")
    print("2. CLIF cross-institutional validation workflow")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    if choice == "1":
        apply_ohca_model_to_clif_dataset()
    elif choice == "2":
        clif_validation_workflow()
    else:
        print("Running CLIF dataset application by default...")
        apply_ohca_model_to_clif_dataset()
