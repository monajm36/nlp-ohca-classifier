"""
Applying OHCA Classifier to External Datasets

This example demonstrates how to apply the trained OHCA model to external datasets


Example use case: Apply MIMIC-trained model to University of Chicago CLIF dataset
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

def apply_ohca_model_to_external_data():
    """
    Complete example of applying OHCA model to external dataset
    
    This example shows how to:
    1. Load a pre-trained OHCA model
    2. Prepare external dataset 
    3. Run inference
    4. Analyze results for clinical use
    """
    
    print("🏥 Applying OHCA Model to External Dataset")
    print("="*50)
    
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
    # STEP 2: Load external dataset
    # ==========================================================================
    
    print("\n📊 Step 2: Loading external dataset...")
    
    # Example: University of Chicago CLIF dataset
    # Replace with your actual data path and format
    external_data_path = "path/to/external/dataset.csv"
    
    # For demonstration, create sample external data
    if not os.path.exists(external_data_path):
        print("Creating sample external dataset for demonstration...")
        external_data_path = create_sample_external_data()
    
    # Load the external dataset
    external_df = pd.read_csv(external_data_path)
    print(f"Loaded {len(external_df):,} cases from external dataset")
    
    # ==========================================================================
    # STEP 3: Prepare data for inference
    # ==========================================================================
    
    print("\n🔧 Step 3: Preparing data for inference...")
    
    # The OHCA model expects columns named 'hadm_id' and 'clean_text'
    # Adapt this section based on your external dataset's column names
    
    # Example column mapping for different datasets:
    column_mapping = {
        # UChicago CLIF example (update with actual column names):
        'patient_id': 'hadm_id',                    # Patient identifier
        'discharge_summary': 'clean_text',          # Clinical text
        
        # Alternative mappings for other datasets:
        # 'encounter_id': 'hadm_id',                # Different ID name
        # 'clinical_notes': 'clean_text',          # Different text column
        # 'admission_id': 'hadm_id',               # Another ID variant
        # 'progress_notes': 'clean_text',          # Different note type
    }
    
    # Apply column mapping
    if any(col in external_df.columns for col in column_mapping.keys()):
        # Rename columns to match model expectations
        external_df = external_df.rename(columns=column_mapping)
        print("✅ Column names mapped successfully")
    else:
        # If columns already have correct names or need manual adjustment
        print("⚠️  Please update column_mapping to match your dataset")
        print(f"Available columns: {list(external_df.columns)}")
        return
    
    # Ensure required columns exist
    if 'hadm_id' not in external_df.columns or 'clean_text' not in external_df.columns:
        print("❌ Required columns 'hadm_id' and 'clean_text' not found")
        print("Please update the column mapping above")
        return
    
    # Clean the data
    external_df = external_df.dropna(subset=['hadm_id', 'clean_text'])
    external_df['clean_text'] = external_df['clean_text'].astype(str)
    
    print(f"✅ Data prepared: {len(external_df):,} cases ready for inference")
    
    # ==========================================================================
    # STEP 4: Run OHCA inference
    # ==========================================================================
    
    print("\n🔍 Step 4: Running OHCA inference...")
    
    # Run inference on external data
    results = run_inference(
        model=model,
        tokenizer=tokenizer,
        inference_df=external_df,
        batch_size=16,
        output_path="external_dataset_ohca_predictions.csv"
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
    
    print(f"\n📊 OHCA Prediction Results:")
    print(f"   Total cases analyzed: {total_cases:,}")
    print(f"   Predicted OHCA (≥0.5): {predicted_ohca_05:,} ({predicted_ohca_05/total_cases:.1%})")
    print(f"   High confidence (≥0.8): {predicted_ohca_08:,} ({predicted_ohca_08/total_cases:.1%})")
    print(f"   Very high confidence (≥0.9): {predicted_ohca_09:,} ({predicted_ohca_09/total_cases:.1%})")
    
    # Detailed analysis
    analysis = analyze_predictions(results)
    
    # Get high-confidence cases for manual review
    high_confidence_cases = get_high_confidence_cases(results, threshold=0.8)
    
    if len(high_confidence_cases) > 0:
        print(f"\n🎯 High Confidence OHCA Cases (for manual review):")
        print(f"   Found {len(high_confidence_cases)} cases with probability ≥ 0.8")
        
        # Save high confidence cases separately
        high_confidence_cases.to_csv(
            "external_dataset_high_confidence_ohca.csv", 
            index=False
        )
        print(f"   💾 Saved to: external_dataset_high_confidence_ohca.csv")
    
    # ==========================================================================
    # STEP 6: Clinical interpretation and next steps
    # ==========================================================================
    
    print(f"\n🏥 Clinical Interpretation:")
    print(f"   • Model identified potential OHCA cases in external dataset")
    print(f"   • Recommend manual review of high-confidence predictions")
    print(f"   • Consider validation against known ground truth if available")
    print(f"   • Monitor for domain shift between training and external data")
    
    print(f"\n📋 Recommended Next Steps:")
    print(f"   1. Review high-confidence predictions manually")
    print(f"   2. Calculate performance metrics if ground truth available")
    print(f"   3. Consider model recalibration for new institution")
    print(f"   4. Document any systematic differences observed")
    
    # ==========================================================================
    # STEP 7: Save comprehensive results
    # ==========================================================================
    
    print(f"\n💾 Saving results...")
    
    # Create comprehensive results summary
    summary = {
        'dataset_info': {
            'total_cases': total_cases,
            'data_source': 'External Dataset',
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
            'external_dataset_ohca_predictions.csv',
            'external_dataset_high_confidence_ohca.csv'
        ]
    }
    
    # Save summary
    import json
    with open('external_dataset_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Analysis complete! Files created:")
    print(f"   📄 external_dataset_ohca_predictions.csv")
    print(f"   🎯 external_dataset_high_confidence_ohca.csv")
    print(f"   📋 external_dataset_analysis_summary.json")
    
    return results

def create_sample_external_data():
    """Create sample external dataset for demonstration"""
    
    sample_data = {
        'patient_id': [f'EXT_{i:06d}' for i in range(500)],
        'discharge_summary': [
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
        ] * 50  # Repeat to get 500 samples
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_path = "sample_external_dataset.csv"
    sample_df.to_csv(sample_path, index=False)
    
    print(f"📝 Created sample external dataset: {sample_path}")
    return sample_path

def external_validation_workflow():
    """
    Specific workflow for external validation studies
    
    Use this when you have ground truth labels for the external dataset
    and want to measure model performance across institutions.
    """
    
    print("🔬 External Validation Workflow")
    print("="*35)
    
    print("\nThis workflow is for when you have:")
    print("• External dataset with known OHCA labels")
    print("• Want to measure cross-institutional performance")
    print("• Need to assess model generalizability")
    
    print("\nSteps:")
    print("1. Apply model to external data (use apply_ohca_model_to_external_data())")
    print("2. Compare predictions with ground truth labels")
    print("3. Calculate performance metrics (AUC, sensitivity, specificity)")
    print("4. Analyze performance differences vs training data")
    print("5. Document domain shift and model limitations")
    
    print("\nExample code for validation metrics:")
    print("""
    # After running inference
    from sklearn.metrics import roc_auc_score, classification_report
    
    # Load ground truth
    ground_truth = pd.read_csv('external_ground_truth.csv')
    
    # Calculate metrics
    auc = roc_auc_score(ground_truth['true_label'], results['ohca_probability'])
    print(f"External validation AUC: {auc:.3f}")
    
    # Compare with training performance
    print("Performance comparison:")
    print(f"Training AUC: {training_auc:.3f}")
    print(f"External AUC: {auc:.3f}")
    print(f"Performance drop: {training_auc - auc:.3f}")
    """)

if __name__ == "__main__":
    print("External Dataset Application Examples")
    print("="*40)
    
    print("\nChoose an example:")
    print("1. Apply model to external dataset")
    print("2. External validation workflow info")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    if choice == "1":
        apply_ohca_model_to_external_data()
    elif choice == "2":
        external_validation_workflow()
    else:
        print("Running external dataset application by default...")
        apply_ohca_model_to_external_data()
