#!/usr/bin/env python3
"""Interactive launcher for parameter studies."""
import subprocess
import sys

def main():
    print("🔬 SD3 Parameter Study Launcher")
    print("=" * 40)
    print()
    print("Choose your experiment:")
    print("1. Quick test (3 experiments)")
    print("2. Switch step study (9 experiments)")
    print("3. Scheduler comparison (5 experiments)")
    print("4. Comprehensive study (25+ experiments)")
    print("5. Custom single experiment")
    print("0. Exit")
    print()
    
    choice = input("Enter your choice (0-5): ").strip()
    
    if choice == "0":
        print("👋 Goodbye!")
        return
    
    elif choice == "1":
        print("🚀 Running quick test...")
        # Just run 3 different switch steps
        for step in [10, 25, 40]:
            cmd = [
                "modal", "run", "src/modal/sd3_modal.py",
                "--prompt-1", "A mountain landscape",
                "--prompt-2", "A city street",
                "--switch-step", str(step),
                "--seed", "42"
            ]
            print(f"Testing switch step {step}...")
            subprocess.run(cmd)
    
    elif choice == "2":
        print("🚀 Running switch step study...")
        subprocess.run(["python", "run_parameter_study.py"])
    
    elif choice == "3":
        print("🚀 Running scheduler comparison...")
        # Run just the scheduler part
        import json
        import datetime
        
        schedulers = [
            "FlowMatchEulerDiscreteScheduler",
            "DPMSolverMultistepScheduler", 
            "EulerDiscreteScheduler",
            "DDIMScheduler",
            "HeunDiscreteScheduler"
        ]
        
        results = []
        for scheduler in schedulers:
            cmd = [
                "modal", "run", "src/modal/sd3_modal.py",
                "--prompt-1", "A peaceful mountain landscape",
                "--prompt-2", "A busy city street",
                "--switch-step", "15",
                "--seed", "42",
                "--scheduler", scheduler
            ]
            print(f"Testing {scheduler}...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            # Parse and save results...
    
    elif choice == "4":
        print("🚀 Running comprehensive study...")
        print("⚠️  This will take a while (25+ experiments)!")
        confirm = input("Continue? (y/N): ").strip().lower()
        if confirm == 'y':
            subprocess.run(["python", "run_parameter_study.py"])
        else:
            print("Cancelled.")
    
    elif choice == "5":
        print("🔧 Custom experiment setup")
        prompt1 = input("Prompt 1: ").strip() or "A mountain landscape"
        prompt2 = input("Prompt 2: ").strip() or "A city street"
        step = input("Switch step (1-49): ").strip() or "25"
        scheduler = input("Scheduler (or press Enter for default): ").strip() or "FlowMatchEulerDiscreteScheduler"
        
        cmd = [
            "modal", "run", "src/modal/sd3_modal.py",
            "--prompt-1", prompt1,
            "--prompt-2", prompt2,
            "--switch-step", step,
            "--seed", "42",
            "--scheduler", scheduler
        ]
        
        print(f"🚀 Running: {' '.join(cmd)}")
        subprocess.run(cmd)
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()