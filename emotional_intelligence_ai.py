#!/usr/bin/env python3
"""
DeepSight AI - Emotional Intelligence Module
Advanced psychological pattern recognition for deepfake detection
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
import json
from dataclasses import dataclass
from enum import Enum

class EmotionType(Enum):
    """Basic emotion types for analysis"""
    HAPPINESS = "happiness"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"

@dataclass
class MicroExpression:
    """Micro-expression data structure"""
    emotion_type: EmotionType
    intensity: float
    duration: float
    onset_time: float
    authenticity_score: float
    facs_units: List[str]

@dataclass
class PersonalityProfile:
    """Big Five personality traits"""
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float

class EmotionalIntelligenceAI:
    """Advanced psychological pattern recognition for deepfake detection"""
    
    def __init__(self):
        """Initialize the emotional intelligence system"""
        self.facs_database = self._load_facs_database()
        self.personality_models = self._load_personality_models()
        self.cultural_patterns = self._load_cultural_patterns()
        self.micro_expression_detector = self._initialize_micro_expression_detector()
        
    def _load_facs_database(self) -> Dict[str, Any]:
        """Load Facial Action Coding System database"""
        # Simulated FACS database - in real implementation, load from research data
        return {
            "AU01": {"name": "Inner Brow Raiser", "muscle": "Frontalis", "emotion_link": ["surprise", "fear"]},
            "AU02": {"name": "Outer Brow Raiser", "muscle": "Frontalis", "emotion_link": ["surprise"]},
            "AU04": {"name": "Brow Lowerer", "muscle": "Corrugator", "emotion_link": ["anger", "sadness"]},
            "AU05": {"name": "Upper Lid Raiser", "muscle": "Levator palpebrae", "emotion_link": ["surprise", "fear"]},
            "AU06": {"name": "Cheek Raiser", "muscle": "Orbicularis oculi", "emotion_link": ["happiness"]},
            "AU07": {"name": "Lid Tightener", "muscle": "Orbicularis oculi", "emotion_link": ["anger", "disgust"]},
            "AU09": {"name": "Nose Wrinkler", "muscle": "Levator labii", "emotion_link": ["disgust"]},
            "AU10": {"name": "Upper Lip Raiser", "muscle": "Levator labii", "emotion_link": ["disgust"]},
            "AU12": {"name": "Lip Corner Puller", "muscle": "Zygomaticus major", "emotion_link": ["happiness"]},
            "AU15": {"name": "Lip Corner Depressor", "muscle": "Triangularis", "emotion_link": ["sadness"]},
            "AU17": {"name": "Chin Raiser", "muscle": "Mentalis", "emotion_link": ["sadness", "doubt"]},
            "AU20": {"name": "Lip Stretcher", "muscle": "Risorius", "emotion_link": ["fear"]},
            "AU23": {"name": "Lip Tightener", "muscle": "Orbicularis oris", "emotion_link": ["anger"]},
            "AU24": {"name": "Lip Pressor", "muscle": "Orbicularis oris", "emotion_link": ["anger"]},
            "AU25": {"name": "Lips Part", "muscle": "Depressor labii", "emotion_link": ["surprise"]},
            "AU26": {"name": "Jaw Drop", "muscle": "Masseter", "emotion_link": ["surprise", "fear"]},
            "AU27": {"name": "Mouth Stretch", "muscle": "Pterygoid", "emotion_link": ["fear"]}
        }
    
    def _load_personality_models(self) -> Dict[str, Any]:
        """Load Big Five personality models"""
        return {
            "facial_expressions": {
                "openness": {"features": ["eye_openness", "brow_position", "smile_asymmetry"]},
                "conscientiousness": {"features": ["micro_expression_control", "expression_timing"]},
                "extraversion": {"features": ["smile_frequency", "eye_contact_duration", "expression_intensity"]},
                "agreeableness": {"features": ["smile_genuineness", "eye_warmth", "facial_tension"]},
                "neuroticism": {"features": ["stress_markers", "expression_volatility", "micro_tension"]}
            },
            "behavioral_patterns": {
                "openness": {"indicators": ["expression_variety", "novel_expressions"]},
                "conscientiousness": {"indicators": ["expression_consistency", "timing_precision"]},
                "extraversion": {"indicators": ["social_expressions", "attention_seeking"]},
                "agreeableness": {"indicators": ["positive_expressions", "empathy_markers"]},
                "neuroticism": {"indicators": ["anxiety_markers", "stress_responses"]}
            }
        }
    
    def _load_cultural_patterns(self) -> Dict[str, Any]:
        """Load cultural gesture and expression patterns"""
        return {
            "western": {
                "eye_contact": {"normal_duration": 3.5, "variance": 1.2},
                "smile_patterns": {"frequency": 0.7, "intensity": 0.6},
                "personal_space": {"comfortable_distance": 60}
            },
            "east_asian": {
                "eye_contact": {"normal_duration": 2.1, "variance": 0.8},
                "smile_patterns": {"frequency": 0.4, "intensity": 0.4},
                "personal_space": {"comfortable_distance": 80}
            },
            "latin": {
                "eye_contact": {"normal_duration": 4.2, "variance": 1.5},
                "smile_patterns": {"frequency": 0.8, "intensity": 0.8},
                "personal_space": {"comfortable_distance": 45}
            }
        }
    
    def _initialize_micro_expression_detector(self):
        """Initialize micro-expression detection model"""
        # Simplified model - in real implementation, use trained neural network
        class MicroExpressionDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_extractor = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.AdaptiveAvgPool2d(8),
                    nn.Flatten(),
                    nn.Linear(64 * 8 * 8, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 7)  # 7 basic emotions
                )
            
            def forward(self, x):
                return self.feature_extractor(x)
        
        return MicroExpressionDetector()
    
    def analyze_micro_expressions(self, facial_sequence: np.ndarray) -> List[MicroExpression]:
        """Detect and analyze micro-expressions in facial sequence"""
        micro_expressions = []
        
        for i in range(len(facial_sequence) - 1):
            current_frame = facial_sequence[i]
            next_frame = facial_sequence[i + 1]
            
            # Detect expression change
            expression_change = self._detect_expression_change(current_frame, next_frame)
            
            if expression_change['magnitude'] > 0.1:  # Threshold for micro-expression
                # Extract FACS units
                facs_units = self._extract_facs_units(current_frame, next_frame)
                
                # Determine emotion type
                emotion_type = self._classify_emotion(facs_units)
                
                # Calculate authenticity score
                authenticity_score = self._calculate_expression_authenticity(
                    facs_units, emotion_type, expression_change
                )
                
                micro_expression = MicroExpression(
                    emotion_type=emotion_type,
                    intensity=expression_change['magnitude'],
                    duration=expression_change['duration'],
                    onset_time=i * 0.033,  # Assuming 30 FPS
                    authenticity_score=authenticity_score,
                    facs_units=facs_units
                )
                
                micro_expressions.append(micro_expression)
        
        return micro_expressions
    
    def _detect_expression_change(self, frame1: np.ndarray, frame2: np.ndarray) -> Dict[str, float]:
        """Detect expression changes between frames"""
        # Optical flow analysis for muscle movement
        flow = cv2.calcOpticalFlowPyrLK(
            cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY),
            None, None
        )
        
        if flow[0] is not None:
            magnitude = np.mean(np.sqrt(flow[0][:,:,0]**2 + flow[0][:,:,1]**2))
            return {
                'magnitude': magnitude,
                'duration': 0.033,  # Single frame duration
                'direction': np.mean(np.arctan2(flow[0][:,:,1], flow[0][:,:,0]))
            }
        else:
            return {'magnitude': 0.0, 'duration': 0.0, 'direction': 0.0}
    
    def _extract_facs_units(self, frame1: np.ndarray, frame2: np.ndarray) -> List[str]:
        """Extract FACS action units from expression change"""
        # Simplified FACS extraction - in real implementation, use specialized models
        active_facs = []
        
        # Analyze different facial regions
        regions = {
            'upper_face': frame1[0:50, :],  # Brow and forehead
            'eye_region': frame1[25:75, :],  # Eyes and surrounding
            'lower_face': frame1[75:, :]     # Mouth and chin
        }
        
        # Simple heuristics for demonstration
        for region_name, region in regions.items():
            if region_name == 'upper_face':
                if self._detect_brow_movement(region):
                    active_facs.extend(['AU01', 'AU02', 'AU04'])
            elif region_name == 'eye_region':
                if self._detect_eye_movement(region):
                    active_facs.extend(['AU05', 'AU06', 'AU07'])
            elif region_name == 'lower_face':
                if self._detect_mouth_movement(region):
                    active_facs.extend(['AU12', 'AU15', 'AU25'])
        
        return active_facs
    
    def _detect_brow_movement(self, brow_region: np.ndarray) -> bool:
        """Detect brow movement in upper face region"""
        # Simplified detection - in real implementation, use landmark detection
        return np.std(brow_region) > 20
    
    def _detect_eye_movement(self, eye_region: np.ndarray) -> bool:
        """Detect eye movement and changes"""
        return np.std(eye_region) > 15
    
    def _detect_mouth_movement(self, mouth_region: np.ndarray) -> bool:
        """Detect mouth and lip movement"""
        return np.std(mouth_region) > 25
    
    def _classify_emotion(self, facs_units: List[str]) -> EmotionType:
        """Classify emotion based on FACS units"""
        emotion_votes = {emotion: 0 for emotion in EmotionType}
        
        for facs in facs_units:
            if facs in self.facs_database:
                for emotion in self.facs_database[facs]['emotion_link']:
                    if emotion in [e.value for e in EmotionType]:
                        emotion_votes[EmotionType(emotion)] += 1
        
        # Return emotion with highest vote
        if max(emotion_votes.values()) > 0:
            return max(emotion_votes, key=emotion_votes.get)
        else:
            return EmotionType.NEUTRAL
    
    def _calculate_expression_authenticity(
        self, 
        facs_units: List[str], 
        emotion_type: EmotionType, 
        expression_change: Dict[str, float]
    ) -> float:
        """Calculate authenticity score for expression"""
        
        # Base authenticity score
        authenticity = 0.5
        
        # Check FACS consistency with emotion
        consistent_facs = 0
        total_facs = len(facs_units)
        
        for facs in facs_units:
            if facs in self.facs_database:
                if emotion_type.value in self.facs_database[facs]['emotion_link']:
                    consistent_facs += 1
        
        if total_facs > 0:
            facs_consistency = consistent_facs / total_facs
            authenticity += 0.3 * facs_consistency
        
        # Check timing authenticity (micro-expressions should be brief)
        timing_score = min(1.0, 0.5 / max(0.1, expression_change['duration']))
        authenticity += 0.2 * timing_score
        
        # Normalize to [0, 1]
        return min(1.0, max(0.0, authenticity))
    
    def analyze_personality_consistency(
        self, 
        behavioral_patterns: Dict[str, Any], 
        known_personality: PersonalityProfile
    ) -> Dict[str, float]:
        """Verify behavioral patterns match known personality"""
        
        # Extract personality traits from behavioral patterns
        extracted_traits = self._extract_personality_traits(behavioral_patterns)
        
        # Compare with known personality
        consistency_scores = {}
        trait_names = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        
        for trait in trait_names:
            known_value = getattr(known_personality, trait)
            extracted_value = extracted_traits.get(trait, 0.5)
            
            # Calculate consistency (inverse of difference)
            difference = abs(known_value - extracted_value)
            consistency_scores[trait] = 1.0 - min(1.0, difference)
        
        # Overall consistency
        overall_consistency = np.mean(list(consistency_scores.values()))
        
        return {
            'individual_traits': consistency_scores,
            'overall_consistency': overall_consistency,
            'trait_deviations': {
                trait: abs(getattr(known_personality, trait) - extracted_traits.get(trait, 0.5))
                for trait in trait_names
            }
        }
    
    def _extract_personality_traits(self, behavioral_patterns: Dict[str, Any]) -> Dict[str, float]:
        """Extract Big Five personality traits from behavioral patterns"""
        traits = {}
        
        # Simplified trait extraction - in real implementation, use validated models
        
        # Openness: variety in expressions, novelty seeking
        expression_variety = behavioral_patterns.get('expression_variety', 0.5)
        traits['openness'] = min(1.0, expression_variety)
        
        # Conscientiousness: consistency and control
        expression_consistency = behavioral_patterns.get('expression_consistency', 0.5)
        traits['conscientiousness'] = min(1.0, expression_consistency)
        
        # Extraversion: social expressions, intensity
        social_expressions = behavioral_patterns.get('social_expressions', 0.5)
        traits['extraversion'] = min(1.0, social_expressions)
        
        # Agreeableness: positive expressions, warmth
        positive_expressions = behavioral_patterns.get('positive_expressions', 0.5)
        traits['agreeableness'] = min(1.0, positive_expressions)
        
        # Neuroticism: stress markers, volatility
        stress_markers = behavioral_patterns.get('stress_markers', 0.5)
        traits['neuroticism'] = min(1.0, stress_markers)
        
        return traits
    
    def detect_deepfake_psychological_inconsistencies(
        self, 
        video_sequence: np.ndarray
    ) -> Dict[str, Any]:
        """Main method to detect psychological inconsistencies in deepfakes"""
        
        results = {
            'micro_expressions': [],
            'authenticity_scores': [],
            'psychological_flags': [],
            'overall_authenticity': 0.0
        }
        
        # Analyze micro-expressions
        micro_expressions = self.analyze_micro_expressions(video_sequence)
        results['micro_expressions'] = micro_expressions
        
        # Calculate authenticity scores
        authenticity_scores = [me.authenticity_score for me in micro_expressions]
        results['authenticity_scores'] = authenticity_scores
        
        # Identify psychological flags
        if authenticity_scores:
            avg_authenticity = np.mean(authenticity_scores)
            
            if avg_authenticity < 0.3:
                results['psychological_flags'].append("Very low micro-expression authenticity")
            elif avg_authenticity < 0.5:
                results['psychological_flags'].append("Low micro-expression authenticity")
            
            # Check for unnatural expression patterns
            if len(micro_expressions) < 2:
                results['psychological_flags'].append("Insufficient emotional variation")
            
            # Check timing patterns
            durations = [me.duration for me in micro_expressions]
            if durations and np.std(durations) < 0.01:
                results['psychological_flags'].append("Unnatural expression timing uniformity")
            
            results['overall_authenticity'] = avg_authenticity
        else:
            results['psychological_flags'].append("No micro-expressions detected")
            results['overall_authenticity'] = 0.0
        
        return results

# Example usage and testing
if __name__ == "__main__":
    print("🎭 DeepSight AI - Emotional Intelligence Module")
    print("=" * 50)
    
    # Initialize the emotional intelligence system
    emotional_ai = EmotionalIntelligenceAI()
    
    # Create sample data for testing
    sample_video = np.random.randint(0, 255, (10, 100, 100, 3), dtype=np.uint8)
    
    # Known personality profile for testing
    known_personality = PersonalityProfile(
        openness=0.7,
        conscientiousness=0.6,
        extraversion=0.8,
        agreeableness=0.7,
        neuroticism=0.3
    )
    
    # Sample behavioral patterns
    behavioral_patterns = {
        'expression_variety': 0.6,
        'expression_consistency': 0.7,
        'social_expressions': 0.8,
        'positive_expressions': 0.7,
        'stress_markers': 0.2
    }
    
    print("🔍 Analyzing psychological authenticity...")
    
    # Run psychological analysis
    psychological_results = emotional_ai.detect_deepfake_psychological_inconsistencies(sample_video)
    
    print(f"✅ Analysis complete!")
    print(f"📊 Overall authenticity: {psychological_results['overall_authenticity']:.2f}")
    print(f"🚩 Psychological flags: {len(psychological_results['psychological_flags'])}")
    
    for flag in psychological_results['psychological_flags']:
        print(f"   ⚠️  {flag}")
    
    # Test personality consistency
    print("\n🧠 Testing personality consistency...")
    personality_results = emotional_ai.analyze_personality_consistency(
        behavioral_patterns, known_personality
    )
    
    print(f"📈 Overall consistency: {personality_results['overall_consistency']:.2f}")
    print("🎯 Individual trait consistency:")
    for trait, score in personality_results['individual_traits'].items():
        print(f"   {trait}: {score:.2f}")
    
    print("\n🎉 Emotional Intelligence Module ready for integration!")
    print("   This module can now be integrated into your main DeepSight AI app")
    print("   for advanced psychological pattern recognition.")
