#!/usr/bin/env python3
"""
DeepSight AI - Deployment Readiness Checklist
Final verification before production deployment
"""

import os
import json
import time
import subprocess
import sys
from pathlib import Path

class DeploymentChecker:
    """Comprehensive deployment readiness checker"""
    
    def __init__(self):
        self.checklist = {
            "critical": [],
            "important": [],
            "optional": [],
            "passed": 0,
            "total": 0
        }
    
    def check_item(self, category, name, condition, details=""):
        """Check a deployment item"""
        self.checklist["total"] += 1
        
        if condition:
            self.checklist["passed"] += 1
            status = "✅"
        else:
            status = "❌"
        
        item = {
            "name": name,
            "status": status,
            "passed": condition,
            "details": details
        }
        
        self.checklist[category].append(item)
        return condition
    
    def run_deployment_check(self):
        """Run complete deployment readiness check"""
        print("🚀 DeepSight AI - Deployment Readiness Check")
        print("=" * 60)
        
        # Critical checks
        print("\n🔴 CRITICAL REQUIREMENTS")
        print("-" * 30)
        
        self.check_item("critical", "Python Environment", 
                       sys.version_info >= (3, 8), f"Python {sys.version}")
        
        self.check_item("critical", "Main Application File", 
                       os.path.exists("app.py"), "app.py")
        
        self.check_item("critical", "Model Weights", 
                       os.path.exists("weights/best_model.pth"), "best_model.pth")
        
        self.check_item("critical", "Grad-CAM Module", 
                       os.path.exists("grad_cam.py"), "grad_cam.py")
        
        # Test imports
        try:
            import torch, streamlit, cv2, numpy, plotly
            import_success = True
        except ImportError as e:
            import_success = False
        
        self.check_item("critical", "Core Dependencies", 
                       import_success, "torch, streamlit, cv2, numpy, plotly")
        
        # Important checks
        print("\n🟡 IMPORTANT FEATURES")
        print("-" * 30)
        
        self.check_item("important", "Emotional Intelligence AI", 
                       os.path.exists("emotional_intelligence_ai.py"), "Psychological analysis")
        
        self.check_item("important", "Advanced Training", 
                       os.path.exists("train_advanced.py"), "Model training capabilities")
        
        self.check_item("important", "Test Suite", 
                       os.path.exists("comprehensive_test.py"), "Validation framework")
        
        self.check_item("important", "Documentation", 
                       any(os.path.exists(f) for f in ["README.md", "project_summary.json"]), "Project docs")
        
        # Check data directories
        data_dirs = ["crops", "weights", "__pycache__"]
        data_available = all(os.path.exists(d) for d in data_dirs)
        self.check_item("important", "Data Structure", 
                       data_available, "crops, weights, cache dirs")
        
        # Optional enhancements
        print("\n🟢 OPTIONAL ENHANCEMENTS")
        print("-" * 30)
        
        self.check_item("optional", "Demo Dataset", 
                       os.path.exists("ffpp_data"), "FaceForensics++ samples")
        
        self.check_item("optional", "Enhanced Crops", 
                       os.path.exists("crops_enhanced"), "Improved training data")
        
        self.check_item("optional", "Configuration Files", 
                       os.path.exists("config.json"), "JSON configuration")
        
        self.check_item("optional", "Git Repository", 
                       os.path.exists(".git"), "Version control")
        
        # Test model loading
        try:
            import torch
            checkpoint = torch.load("weights/best_model.pth", map_location="cpu")
            model_accuracy = checkpoint.get("accuracy", 0)
            model_working = model_accuracy > 0.9  # 90%+ accuracy
            accuracy_text = f"{model_accuracy:.2%}" if isinstance(model_accuracy, float) else str(model_accuracy)
        except:
            model_working = False
            accuracy_text = "Failed to load"
        
        self.check_item("critical", "Model Performance", 
                       model_working, f"Accuracy: {accuracy_text}")
        
        # Print results
        self._print_results()
        
        # Generate recommendation
        return self._generate_recommendation()
    
    def _print_results(self):
        """Print detailed results"""
        
        categories = [
            ("🔴 CRITICAL", "critical"),
            ("🟡 IMPORTANT", "important"), 
            ("🟢 OPTIONAL", "optional")
        ]
        
        for title, category in categories:
            print(f"\n{title}")
            items = self.checklist[category]
            
            for item in items:
                print(f"{item['status']} {item['name']}")
                if item['details']:
                    print(f"    └─ {item['details']}")
        
        # Summary
        passed = self.checklist["passed"]
        total = self.checklist["total"]
        score = passed / total if total > 0 else 0
        
        print(f"\n📊 OVERALL SCORE: {passed}/{total} ({score:.1%})")
        
        # Count by category
        critical_passed = sum(1 for item in self.checklist["critical"] if item["passed"])
        critical_total = len(self.checklist["critical"])
        
        important_passed = sum(1 for item in self.checklist["important"] if item["passed"])
        important_total = len(self.checklist["important"])
        
        print(f"🔴 Critical: {critical_passed}/{critical_total}")
        print(f"🟡 Important: {important_passed}/{important_total}")
        print(f"🟢 Optional: {passed - critical_passed - important_passed}/{total - critical_total - important_total}")
    
    def _generate_recommendation(self):
        """Generate deployment recommendation"""
        
        critical_passed = sum(1 for item in self.checklist["critical"] if item["passed"])
        critical_total = len(self.checklist["critical"])
        
        important_passed = sum(1 for item in self.checklist["important"] if item["passed"])
        important_total = len(self.checklist["important"])
        
        total_score = self.checklist["passed"] / self.checklist["total"]
        
        print(f"\n🎯 DEPLOYMENT RECOMMENDATION")
        print("-" * 40)
        
        if critical_passed == critical_total and total_score >= 0.8:
            print("🎉 READY FOR PRODUCTION DEPLOYMENT!")
            print("   ✅ All critical requirements met")
            print("   ✅ High overall score")
            print("   🚀 Launch command: streamlit run app.py")
            recommendation = "DEPLOY"
            
        elif critical_passed == critical_total:
            print("⚠️ READY WITH MINOR ISSUES")
            print("   ✅ All critical requirements met")
            print("   ⚠️ Some optional features missing")
            print("   🚀 Can deploy, consider improvements")
            recommendation = "DEPLOY_WITH_WARNINGS"
            
        else:
            print("❌ NOT READY FOR DEPLOYMENT")
            print("   ❌ Critical requirements missing")
            print("   🔧 Fix critical issues before deployment")
            
            # List critical failures
            failed_critical = [item for item in self.checklist["critical"] if not item["passed"]]
            print("   \n🔧 REQUIRED FIXES:")
            for item in failed_critical:
                print(f"      • {item['name']}")
            
            recommendation = "DO_NOT_DEPLOY"
        
        # Save deployment report
        deployment_report = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "recommendation": recommendation,
            "score": total_score,
            "checklist": self.checklist,
            "critical_score": critical_passed / critical_total if critical_total > 0 else 0,
            "important_score": important_passed / important_total if important_total > 0 else 0
        }
        
        with open("deployment_report.json", "w") as f:
            json.dump(deployment_report, f, indent=2)
        
        print(f"\n📄 Deployment report saved to: deployment_report.json")
        
        return recommendation

def main():
    """Main deployment check function"""
    checker = DeploymentChecker()
    recommendation = checker.run_deployment_check()
    
    # Final message
    print(f"\n{'='*60}")
    if recommendation == "DEPLOY":
        print("🎉 DeepSight AI is production-ready!")
        print("🚀 Run: streamlit run app.py")
        print("🌐 Visit: http://localhost:8501")
        return True
    elif recommendation == "DEPLOY_WITH_WARNINGS":
        print("⚠️ DeepSight AI is deployable with minor issues")
        print("🚀 Run: streamlit run app.py")
        return True
    else:
        print("❌ Fix critical issues before deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
