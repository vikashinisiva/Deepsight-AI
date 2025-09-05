/**
 * DeepSight AI Service - Main AI Service for Image and Video Analysis
 * Integrates with DeepSight AI backend for comprehensive analysis
 */

import { deepSightAIService, DetectionResult } from './deepfakeService';
import enhancedAIService, { ImageAnalysisResult } from './deepSightAIService';

// Re-export types for convenience
export type { ImageAnalysisResult, VideoAnalysisRequest } from './deepSightAIService';
export type { DetectionResult };

// Define AnalysisResult as a union type for the UI
export type AnalysisResult = DetectionResult | ImageAnalysisResult;

class AIService {
  private deepfakeService = deepSightAIService;
  private enhancedService = enhancedAIService;

  /**
   * Analyze image using the enhanced AI service
   */
  async analyzeImage(request: any) {
    return await this.enhancedService.analyzeImage(request);
  }

  /**
   * Analyze video for deepfake detection
   */
  async analyzeVideo(request: { videoUri: string; detailed?: boolean; onProgress?: (progress: number) => void }): Promise<DetectionResult> {
    try {
      console.log('🚀 Starting deepfake analysis for video:', request.videoUri);

      // Use the deepfake service to analyze the video
      const result = await this.deepfakeService.detectDeepfake({
        videoUri: request.videoUri,
        maxFrames: 30,
        detailed: request.detailed || true,
        onProgress: request.onProgress
      });

      console.log('✅ Deepfake analysis completed:', result);
      return result;

    } catch (error) {
      console.error('❌ Deepfake analysis failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      throw new Error(`Deepfake analysis failed: ${errorMessage}`);
    }
  }

  /**
   * Infer manipulation types from detection result
   */
  private inferManipulationTypes(result: DetectionResult): string[] {
    const types: string[] = [];

    if (result.prediction === 'FAKE') {
      // Based on confidence and other factors, infer manipulation types
      if (result.fake_confidence > 0.8) {
        types.push('Face Manipulation');
        types.push('Deepfake Generation');
      }
      if (result.frames_used > 20) {
        types.push('Video Synthesis');
      }
      if (result.detailed_analysis?.detection_method) {
        types.push(result.detailed_analysis.detection_method);
      }
    }

    return types.length > 0 ? types : ['Unknown Manipulation'];
  }

  /**
   * Calculate risk level based on confidence
   */
  private calculateRiskLevel(confidence: number, confidenceLevel: string): 'low' | 'medium' | 'high' | 'critical' {
    if (confidence > 0.9 || confidenceLevel === 'HIGH') {
      return 'critical';
    } else if (confidence > 0.7 || confidenceLevel === 'MEDIUM') {
      return 'high';
    } else if (confidence > 0.5) {
      return 'medium';
    } else {
      return 'low';
    }
  }

  /**
   * Generate human-readable explanation
   */
  private generateExplanation(result: DetectionResult): string {
    if (result.prediction === 'REAL') {
      return `This video appears to be authentic with ${Math.round((1 - result.fake_confidence) * 100)}% confidence. No signs of manipulation were detected.`;
    } else if (result.prediction === 'FAKE') {
      return `This video shows signs of manipulation with ${Math.round(result.fake_confidence * 100)}% confidence. ${result.frames_used} frames were analyzed using ${result.detailed_analysis?.detection_method || 'advanced AI detection'}.`;
    } else {
      return 'Unable to determine the authenticity of this video. Further analysis may be required.';
    }
  }

  /**
   * Health check for AI services
   */
  async healthCheck(): Promise<boolean> {
    try {
      const deepfakeHealth = await this.deepfakeService.healthCheck();
      return deepfakeHealth;
    } catch (error) {
      console.error('AI service health check failed:', error);
      return false;
    }
  }

  /**
   * Get analysis history
   */
  async getAnalysisHistory(sessionId: string) {
    return await this.deepfakeService.getAnalysisHistory(sessionId);
  }

  /**
   * Get API statistics
   */
  async getApiStats() {
    return await this.deepfakeService.getApiStats();
  }

  /**
   * Clear cache
   */
  async clearCache() {
    await this.deepfakeService.clearCache();
    await this.enhancedService.clearCache();
  }

  /**
   * Update API configuration
   */
  updateApiConfig(config: { baseUrl?: string; timeout?: number }) {
    if (config.baseUrl) {
      this.deepfakeService.updateBaseUrl(config.baseUrl);
      this.enhancedService.updateBaseUrl(config.baseUrl);
    }
  }
}

// Export singleton instance
export const aiService = new AIService();
export default aiService;