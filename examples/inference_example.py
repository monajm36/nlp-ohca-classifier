"""
OHCA Inference Example

This example shows how to use a pre-trained OHCA classifier on new datasets.
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.append('../src')

from ohca_inference import (
    load_ohca_model,
    run_inference,
    quick_inference,
    process_large_dataset,
    test_model_on_sample,
    get_high_confidence_cases,
    analyze_predictions
)

def basic_inference_example():
    """Basic example of running inference on new data"""
    
    print("🔍 Basic OHCA Inference Example")
    print("="*40)
    
    # ==========================================================================
    # STEP 1: Prepare your model and data
    # ==========================================================================
    
    model_path = "./trained_ohca_model"  # Path to your trained model
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please train a model first using the training pipeline.")
        return
    
    # ==========================================================================
    # STEP 2: Prepare new data for inference
    # ==========================================================================
    
    # Your new data should be a CSV with columns: hadm_id, clean_text
    new_data_path = "path/to/new_discharge_notes.csv"
    
    # For demonstration, create sample new data
    if not os.path.exists(new_data_path):
        print("Creating sample new data for demonstration...")
        
        new_sample_data = {
            'hadm_id': [f'NEW_{i:06d}' for i in range(100)],
            'clean_text': [
                "Chief complaint: Cardiac arrest at home. Family initiated CPR immediately, EMS transported to hospital.",
                "Chief complaint: Chest pain. Patient has stable angina, no cardiac arrest occurred during admission.",
                "Chief complaint: Found down at work. Witnessed cardiac arrest, coworker CPR, AED used with ROSC.",
                "Chief complaint: Shortness of breath. CHF exacerbation, treated with diuretics, stable course.",
                "Chief complaint: Syncope. Brief loss of consciousness, no arrest, negative cardiac workup.",
                "Chief complaint: Transfer for cardiac catheterization. OHCA at restaurant, bystander CPR given.",
                "Chief complaint: Diabetes management. Routine admission for glucose control, no acute events.",
                "Chief complaint: Cardiac arrest in parking garage. CPR by security, EMS achieved ROSC.",
                "Chief complaint: Pneumonia. Community-acquired pneumonia, treated with antibiotics.",
                "Chief complaint: Collapse at gym. Witnessed cardiac arrest, immediate bystander CPR provided.",
            ] * 10  # Repeat to get 100 samples
        }
        
        new_df = pd.DataFrame(new_sample_data)
        new_df.to_csv(new_data_path, index=False)
        print(f"Sample new data saved to: {new_data_path}")
    
    # ==========================================================================
    # STEP 3: Load the trained model
    # ==========================================================================
    
    print(f"\n📂 STEP 3: Loading Trained Model")
    print("-" * 40)
    
    model, tokenizer = load_ohca_model(model_path)
    
    # ==========================================================================
    # STEP 4: Run inference
    # ==========================================================================
    
    print(f"\n🔍 STEP 4: Running Inference")
    print("-" * 40)
    
    # Load new data
    new_df = pd.read_csv(new_data_path)
    print(f"Loaded {len(new_df):,} new cases for inference")
    
    # Run inference
    results = run_inference(
        model=model,
        tokenizer=tokenizer,
        inference_df=new_df,
        batch_size=16,
        output_path="./inference_results.csv",
        probability_threshold=0.5
    )
    
    # ==========================================================================
    # STEP 5: Analyze results
    # ==========================================================================
    
    print(f"\n📊 STEP 5: Analyzing Results")
    print("-" * 40)
    
    analysis = analyze_predictions(results, new_df)
    
    # Show high-confidence cases
    high_conf_cases = get_high_confidence_cases(results, threshold=0.8)
    
    if len(high_conf_cases) > 0:
        print(f"\n🎯 High Confidence OHCA Cases (≥0.8):")
        print("-" * 40)
        
        for i, (_, row) in enumerate(high_conf_cases.head(5).iterrows(), 1):
            hadm_id = row['hadm_id']
            prob = row['ohca_probability']
            
            # Get original text
            original_text = new_df[new_df['hadm_id'] == hadm_id]['clean_text'].iloc[0]
            
            print(f"{i}. {hadm_id} (Probability: {prob:.3f})")
            print(f"   Text: {original_text[:150]}...")
            print()
    
    # ==========================================================================
    # STEP 6: Save and summarize
    # ==========================================================================
    
    print(f"\n✅ INFERENCE COMPLETE!")
    print("="*40)
    
    print(f"📁 Results saved to: ./inference_results.csv")
    print(f"📊 Total cases processed: {len(results):,}")
    print(f"🎯 Predicted OHCA cases: {results['prediction_050'].sum():,}")
    print(f"⭐ High confidence cases: {(results['ohca_probability'] >= 0.8).sum():,}")
    
    return results

def quick_inference_example():
    """Quick inference using the convenience function"""
    
    print("⚡ Quick Inference Example")
    print("="*30)
    
    model_path = "./trained_ohca_model"
    data_path = "path/to/new_discharge_notes.csv"
    
    # Use the quick inference function
    results = quick_inference(
        model_path=model_path,
        data_path=data_path,
        output_path="./quick_inference_results.csv"
    )
    
    return results

def test_model_example():
    """Test model on specific sample texts"""
    
    print("🧪 Model Testing Example")
    print("="*30)
    
    model_path = "./trained_ohca_model"
    
    # Test specific cases
    test_cases = {
        'TEST_001': "Chief complaint: Cardiac arrest at home. Found down by spouse, immediate CPR.",
        'TEST_002': "Chief complaint: Chest pain. Acute MI, no arrest, successful PCI performed.",
        'TEST_003': "Chief complaint: Found down at work. Cardiac arrest witnessed, bystander CPR given.",
        'TEST_004': "Chief complaint: Syncope. Brief LOC, no arrest, negative workup.",
        'TEST_005': "Chief complaint: Transfer for OHCA. Arrest in restaurant, EMS resuscitation."
    }
    
    # Test the model
    test_results = test_model_on_sample(model_path, test_cases)
    
    return test_results

def large_dataset_example():
    """Example of processing a large dataset in chunks"""
    
    print("📦 Large Dataset Processing Example")
    print("="*40)
    
    model_path = "./trained_ohca_model"
    large_data_path = "path/to/large_dataset.csv"
    output_path = "./large_dataset_results.csv"
    
    # For demonstration, create a large sample dataset
    if not os.path.exists(large_data_path):
        print("Creating large sample dataset...")
        
        # Create 50,000 sample records
        large_sample = {
            'hadm_id': [f'LARGE_{i:08d}' for i in range(50000)],
            'clean_text': [
                "Chief complaint: Cardiac arrest at home.",
                "Chief complaint: Chest pain, no arrest.",
                "Chief complaint: Found down, cardiac arrest.",
                "Chief complaint: Shortness of breath.",
                "Chief complaint: Syncope, no arrest.",
            ] * 10000
        }
        
        large_df = pd.DataFrame(large_sample)
        large_df.to_csv(large_data_path, index=False)
        print(f"Large sample dataset created: {large_data_path}")
    
    # Process in chunks
    result_path = process_large_dataset(
        model_path=model_path,
        data_path=large_data_path,
        output_path=output_path,
        chunk_size=5000,  # Process 5000 rows at a time
        batch_size=32     # Larger batch size for efficiency
    )
    
    print(f"Large dataset processing complete: {result_path}")
    
    return result_path

def clinical_interpretation_example():
    """Example of interpreting results for clinical use"""
    
    print("🏥 Clinical Interpretation Example")
    print("="*40)
    
    # Load results from previous inference
    results_path = "./inference_results.csv"
    
    if not os.path.exists(results_path):
        print("No inference results found. Running basic inference first...")
        results = basic_inference_example()
    else:
        results = pd.read_csv(results_path)
    
    print(f"\n📋 Clinical Decision Support")
    print("-" * 40)
    
    # Categorize cases by clinical priority
    very_high = results[results['ohca_probability'] >= 0.9]
    high = results[(results['ohca_probability'] >= 0.7) & (results['ohca_probability'] < 0.9)]
    medium = results[(results['ohca_probability'] >= 0.3) & (results['ohca_probability'] < 0.7)]
    low = results[results['ohca_probability'] < 0.3]
    
    print(f"🔴 Very High Priority (≥0.9): {len(very_high):,} cases")
    print("   → Immediate manual review recommended")
    print("   → High likelihood of true OHCA")
    
    print(f"\n🟡 High Priority (0.7-0.9): {len(high):,} cases") 
    print("   → Clinical review suggested")
    print("   → Moderate to high OHCA likelihood")
    
    print(f"\n🟠 Medium Priority (0.3-0.7): {len(medium):,} cases")
    print("   → Consider for review if resources allow")
    print("   → Uncertain cases requiring judgment")
    
    print(f"\n🟢 Low Priority (<0.3): {len(low):,} cases")
    print("   → Unlikely to be OHCA")
    print("   → Can focus on other priorities")
    
    # Workflow recommendations
    print(f"\n📋 Recommended Clinical Workflow:")
    print("1. Review all Very High Priority cases first")
    print("2. Validate High Priority cases when possible")
    print("3. Use medium priority for quality improvement")
    print("4. Monitor low priority for false negatives")
    
    # Performance expectations
    print(f"\n📊 Expected Performance:")
    print("• Model catches ~85-95% of true OHCA cases")
    print("• ~10-20% of predictions may be false positives")
    print("• Higher thresholds = fewer false positives")
    print("• Lower thresholds = catch more true cases")
    
    return {
        'very_high_priority': very_high,
        'high_priority': high,
        'medium_priority': medium,
        'low_priority': low
    }

def model_validation_example():
    """Example of validating model performance on new data"""
    
    print("🔬 Model Validation Example")
    print("="*35)
    
    print("\n📋 Model Validation Checklist:")
    print("1. Test on held-out dataset from same institution")
    print("2. Test on data from different institutions") 
    print("3. Analyze performance across patient subgroups")
    print("4. Check for distribution shifts in new data")
    print("5. Monitor prediction confidence distributions")
    
    # Load inference results
    if os.path.exists("./inference_results.csv"):
        results = pd.read_csv("./inference_results.csv")
        
        print(f"\n📊 Current Model Statistics:")
        print(f"   Mean probability: {results['ohca_probability'].mean():.4f}")
        print(f"   Std probability: {results['ohca_probability'].std():.4f}")
        print(f"   Cases >0.8: {(results['ohca_probability'] > 0.8).sum():,}")
        print(f"   Cases >0.5: {(results['ohca_probability'] > 0.5).sum():,}")
        
        # Check for potential issues
        very_low = (results['ohca_probability'] < 0.01).sum()
        very_high = (results['ohca_probability'] > 0.99).sum()
        
        print(f"\n🔍 Quality Checks:")
        if very_low > len(results) * 0.8:
            print("   ⚠️  Many very low probabilities - check data quality")
        if very_high > len(results) * 0.1:
            print("   ⚠️  Many very high probabilities - possible overfitting")
        if results['ohca_probability'].std() < 0.1:
            print("   ⚠️  Low prediction variance - model may be underconfident")
        
        print("   ✅ Use manual review to validate high-confidence predictions")
    
    print(f"\n💡 Validation Tips:")
    print("• Start with high-confidence predictions for validation")
    print("• Track false positive and false negative rates")
    print("• Consider retraining if performance degrades")
    print("• Monitor for changes in data distribution")

if __name__ == "__main__":
    print("OHCA Inference Examples")
    print("="*25)
    
    print("\nChoose an example:")
    print("1. Basic inference on new data")
    print("2. Quick inference (simple)")
    print("3. Test model on sample texts") 
    print("4. Large dataset processing")
    print("5. Clinical interpretation")
    print("6. Model validation")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        basic_inference_example()
    elif choice == "2":
        quick_inference_example()
    elif choice == "3":
        test_model_example()
    elif choice == "4":
        large_dataset_example()
    elif choice == "5":
        clinical_interpretation_example()
    elif choice == "6":
        model_validation_example()
    else:
        print("Running basic inference example by default...")
        basic_inference_example()
