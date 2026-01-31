"""
COMPLETE PIPELINE TEST - Cross-platform
========================================

Runs the entire feedback validation pipeline automatically:
1. Generate feedback
2. Simulate revisions  
3. Validate improvements

Usage:
    python test_full_pipeline.py [num_essays] [data_file]
    
Example:
    python test_full_pipeline.py 30 data/train_balanced.csv
"""

import os
import sys
import subprocess
import time

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print('='*80)
    print(f"Running: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed")
        sys.exit(1)
    
    print(f"\n✅ {description} complete!")
    return result.returncode

def main():
    print("\n" + "="*80)
    print("ADAPTIVE FEEDBACK VALIDATION - COMPLETE PIPELINE TEST")
    print("="*80)
    
    # Configuration
    num_essays = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    data_file = sys.argv[2] if len(sys.argv) > 2 else "data/train_balanced.csv"
    
    # Check data file
    if not os.path.exists(data_file):
        print(f"\n❌ Error: {data_file} not found")
        print("   Usage: python test_full_pipeline.py [num_essays] [data_file]")
        sys.exit(1)
    
    print(f"\n📁 Using dataset: {data_file}")
    print(f"📊 Testing with: {num_essays} essays")
    
    # ========================================================================
    # STEP 1: GENERATE FEEDBACK
    # ========================================================================
    input("\nPress Enter to start Step 1: Generate Feedback...")
    
    cmd = f"python validation_pipeline.py generate {data_file} --output test_feedback.json --num {num_essays}"
    run_command(cmd, "STEP 1: GENERATING FEEDBACK PACKAGE")
    
    # ========================================================================
    # STEP 2: SIMULATE REVISIONS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: SIMULATING STUDENT REVISIONS")
    print("="*80)
    print("\nℹ️  In a real scenario, students would revise their essays here.")
    print("   This simulation mimics realistic improvements:")
    print("   • 60-80% compliance with feedback")
    print("   • Adding transitions and better vocabulary")
    print("   • Expanding short essays")
    print("   • Improving paragraph structure")
    
    input("\nPress Enter to simulate revisions...")
    
    cmd = "python simulate_revisions.py test_feedback_original.csv test_feedback.json simulated_revisions.csv"
    run_command(cmd, "STEP 2: SIMULATING REVISIONS")
    
    # ========================================================================
    # STEP 3: VALIDATE IMPROVEMENTS
    # ========================================================================
    input("\nPress Enter to validate improvements...")
    
    cmd = "python validation_pipeline.py validate test_feedback_original.csv simulated_revisions.csv"
    run_command(cmd, "STEP 3: VALIDATING IMPROVEMENTS")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETE! 🎉")
    print("="*80)
    
    print("\n📄 Generated files:")
    print("   • test_feedback.json - Feedback data")
    print("   • test_feedback.txt - Human-readable feedback")
    print("   • test_feedback_original.csv - Original essays")
    print("   • simulated_revisions.csv - Revised essays")
    print("   • validation_results.json - Statistical results")
    print("   • improvement_analysis.png - Visualization")
    
    print("\n📊 What to check:")
    print("   1. Look at test_feedback.txt - Is the feedback helpful?")
    print("   2. Compare original vs revised essays - Are improvements realistic?")
    print("   3. Check validation_results.json - Is p-value < 0.05?")
    print("   4. View improvement_analysis.png - Do most essays improve?")
    
    print("\n💡 Next steps:")
    print("   • If results look good, collect real student revisions")
    print("   • Use: python validation_pipeline.py validate original.csv revised.csv")
    print("   • Real students typically show 0.3-0.8 band improvement")
    print()

if __name__ == "__main__":
    main()