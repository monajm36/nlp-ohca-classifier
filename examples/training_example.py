"""
OHCA Training Pipeline Example

This example shows how to train an OHCA classifier from scratch.
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.append('../src')

from ohca_training_pipeline import (
    create_training_sample,
    prepare_training_data,
    train_ohca_model,
    evaluate_model,
    complete_training_pipeline,
    complete_annotation_and_train
)

def example_training_pipeline():
    """Complete example of training an OHCA classifier"""
    
    print("🚀 OHCA Training Pipeline Example")
    print("="*50)
    
    # ==========================================================================
    # STEP 1: Prepare your data
    # ==========================================================================
    
    # Your discharge notes should be in CSV format with columns:
    # - hadm_id: Unique identifier for each hospital admission  
    # - clean_text: Cleaned discharge note text
    
    data_path = "path/to/your/discharge_notes.csv"
    
    # For demonstration, create sample data
    if not os.path.exists(data_path):
        print("Creating sample data for demonstration...")
        
        sample_data = {
            'hadm_id': [f'HADM_{i:06d}' for i in range(2000)],
            'clean_text': [
                "Chief complaint: Cardiac arrest at home. Patient found down by family members, CPR initiated immediately. EMS called, patient transported to ED.",
                "Chief complaint: Chest pain. Patient presents with acute onset chest pain, no loss of consciousness, no arrest occurred.",
                "Chief complaint: Shortness of breath. Patient has chronic heart failure exacerbation, stable vital signs throughout admission.",
                "Chief complaint: Patient found down, cardiac arrest in parking lot, bystander CPR given, ROSC achieved by EMS in field.",
                "Chief complaint: Syncope. Patient had brief loss of consciousness but no cardiac arrest, workup negative for cardiac causes.",
                "Chief complaint: Transfer from outside hospital. Patient had witnessed cardiac arrest at work, CPR by coworkers, transferred for cardiac catheterization.",
            ] * 334  # Repeat to get 2000+ samples
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(data_path, index=False)
        print(f"Sample data saved to: {data_path}")
    
    # ==========================================================================
    # STEP 2: Create annotation sample
    # ==========================================================================
    
    print("\n📝 STEP 2: Creating Annotation Sample")
    print("-" * 40)
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} discharge notes")
    
    # Create balanced sample for annotation
    annotation_result = create_training_sample(
        df, 
        output_dir="./training_annotation_interface"
    )
    
    print(f"\n✅ Annotation interface created!")
    print(f"📁 Files created:")
    print(f"   - ./training_annotation_interface/ohca_annotation.xlsx")
    print(f"   - ./training_annotation_interface/annotation_guidelines.md")
    
    # ==========================================================================
    # MANUAL ANNOTATION PHASE
    # ==========================================================================
    
    print("\n" + "="*60)
    print("⏸️  MANUAL ANNOTATION REQUIRED")
    print("="*60)
    print("Before continuing, you need to:")
    print("1. Open: ./training_annotation_interface/ohca_annotation.xlsx")
    print("2. Read: ./training_annotation_interface/annotation_guidelines.md") 
    print("3. Manually label each case:")
    print("   - 1 = OHCA (out-of-hospital cardiac arrest)")
    print("   - 0 = Non-OHCA (everything else)")
    print("4. Fill in confidence scores (1-5)")
    print("5. Save the Excel file")
    print("6. Run continue_training_after_annotation()")
    print("="*60)
    
    # For demonstration, create mock annotations
    print("\n🔧 Creating mock annotations for demonstration...")
    
    annotation_df = pd.read_excel("./training_annotation_interface/ohca_annotation.xlsx")
    
    # Simple rule-based mock labeling (in practice, this is done manually)
    def mock_label(text):
        text_lower = str(text).lower()
        if 'cardiac arrest' in text_lower and any(word in text_lower for word in ['home', 'work', 'found down', 'parking lot']):
            return 1  # OHCA
        else:
            return 0  # Non-OHCA
    
    annotation_df['ohca_label'] = annotation_df['clean_text'].apply(mock_label)
    annotation_df['confidence'] = 4  # Mock confidence
    annotation_df['annotator'] = 'demo'
    annotation_df['annotation_date'] = '2025-01-01'
    annotation_df['notes'] = 'Mock annotation for demo'
    
    # Save completed annotations
    completed_file = "./training_annotation_interface/ohca_annotation_completed.xlsx"
    annotation_df.to_excel(completed_file, index=False)
    
    print(f"✅ Mock annotations created: {completed_file}")
    
    # Continue with training
    return continue_training_after_annotation(completed_file)

def continue_training_after_annotation(annotation_file):
    """Continue training after manual annotation is complete"""
    
    print("\n🔄 CONTINUING TRAINING AFTER ANNOTATION")
    print("="*50)
    
    # ==========================================================================
    # STEP 3: Prepare training data
    # ==========================================================================
    
    print("\n📊 STEP 3: Preparing Training Data")
    print("-" * 40)
    
    # Load completed annotations
    labeled_df = pd.read_excel(annotation_file)
    
    # Prepare training datasets
    train_dataset, val_dataset, train_df, tokenizer = prepare_training_data(labeled_df)
    
    # ==========================================================================
    # STEP 4: Train the model
    # ==========================================================================
    
    print("\n🏋️ STEP 4: Training Model")
    print("-" * 40)
    
    model, trained_tokenizer = train_ohca_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset, 
        train_df=train_df,
        tokenizer=tokenizer,
        num_epochs=3,
        save_path="./trained_ohca_model"
    )
    
    # ==========================================================================
    # STEP 5: Evaluate the model
    # ==========================================================================
    
    print("\n📈 STEP 5: Evaluating Model")
    print("-" * 40)
    
    evaluation_results = evaluate_model(
        model=model,
        val_dataset=val_dataset,
        save_results=True,
        results_path="./trained_ohca_model/evaluation_results.txt"
    )
    
    # ==========================================================================
    # STEP 6: Training complete summary
    # ==========================================================================
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    
    print(f"📁 Model saved to: ./trained_ohca_model/")
    print(f"📊 Evaluation results: ./trained_ohca_model/evaluation_results.txt")
    
    print(f"\n📈 Performance Summary:")
    print(f"   AUC-ROC: {evaluation_results['auc']:.3f}")
    print(f"   F1-Score: {evaluation_results['optimal_metrics']['f1']:.3f}")
    print(f"   Sensitivity: {evaluation_results['optimal_metrics']['recall']:.1%}")
    print(f"   Specificity: {evaluation_results['optimal_metrics']['specificity']:.1%}")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Review evaluation results")
    print(f"   2. Test model on new data using inference module")
    print(f"   3. Deploy model for clinical use")
    print(f"   4. Consider retraining with more data if needed")
    
    return {
        'model_path': "./trained_ohca_model/",
        'evaluation_results': evaluation_results,
        'training_data_size': len(train_dataset),
        'validation_data_size': len(val_dataset)
    }

def quick_training_example():
    """Simplified training example using the complete pipeline function"""
    
    print("⚡ Quick Training Pipeline Example")
    print("="*40)
    
    # Use the complete pipeline function
    data_path = "path/to/your/discharge_notes.csv"
    
    # Step 1: Create annotation interface
    result = complete_training_pipeline(
        data_path=data_path,
        annotation_dir="./quick_annotation_interface",
        model_save_path="./quick_trained_model"
    )
    
    print(f"Annotation files created:")
    print(f"  📄 {result['annotation_file']}")
    print(f"  📋 {result['guidelines_file']}")
    
    # After manual annotation, continue with:
    # final_result = complete_annotation_and_train(
    #     annotation_file=result['annotation_file'],
    #     model_save_path="./quick_trained_model",
    #     num_epochs=3
    # )
    
    return result

def training_tips_and_best_practices():
    """Tips for successful OHCA model training"""
    
    print("💡 OHCA Training Tips & Best Practices")
    print("="*45)
    
    print("\n📋 Data Preparation:")
    print("   • Ensure discharge notes are well-cleaned")
    print("   • Include diverse hospital systems if possible")
    print("   • Minimum 200-300 cases for reliable training")
    print("   • Aim for 10-30% OHCA prevalence in sample")
    
    print("\n🏷️  Annotation Guidelines:")
    print("   • Be consistent with OHCA definition")
    print("   • Focus on PRIMARY reason for admission")
    print("   • Use confidence scores to flag uncertain cases")
    print("   • Consider inter-annotator agreement for quality")
    
    print("\n🔧 Model Training:")
    print("   • Start with 3 epochs, increase if underfitting")
    print("   • Monitor for overfitting in small datasets")
    print("   • Consider class balancing for imbalanced data")
    print("   • Use validation set to tune hyperparameters")
    
    print("\n📊 Model Evaluation:")
    print("   • Prioritize sensitivity (catching OHCA cases)")
    print("   • Balance sensitivity vs specificity for use case")
    print("   • AUC > 0.8 indicates good performance")
    print("   • F1-score > 0.7 suggests balanced performance")
    
    print("\n🎯 Model Deployment:")
    print("   • Test on held-out dataset before deployment")
    print("   • Consider probability thresholds for clinical use")
    print("   • Plan for model monitoring and retraining")
    print("   • Document model limitations and scope")

if __name__ == "__main__":
    print("OHCA Training Examples")
    print("="*25)
    
    print("\nChoose an example:")
    print("1. Complete training pipeline")
    print("2. Quick training example") 
    print("3. Training tips and best practices")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        example_training_pipeline()
    elif choice == "2":
        quick_training_example()
    elif choice == "3":
        training_tips_and_best_practices()
    else:
        print("Running complete training pipeline by default...")
        example_training_pipeline()
