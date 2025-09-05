import React, { useState, useEffect } from 'react';
import { View, Text, Image, ScrollView, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import { Ionicons } from '@expo/vector-icons';
import { router, useLocalSearchParams } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { Video, ResizeMode } from 'expo-av';
import {
  aiService,
  ImageAnalysisResult,
  DetectionResult,
  AnalysisResult,
} from '@/services/aiService';
import { storageService, StoredAnalysis } from '@/services/storageService';

interface AnalysisProgress {
  step: string;
  progress: number;
  message: string;
}

export default function AnalysisScreen() {
  const { imageUri, mediaType, isVideo } = useLocalSearchParams<{ 
    imageUri: string; 
    mediaType?: string; 
    isVideo?: string;
  }>();
  const [isAnalyzing, setIsAnalyzing] = useState(true);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState<AnalysisProgress>({
    step: 'initializing',
    progress: 0,
    message: isVideo === 'true' ? 'Preparing video for deepfake analysis...' : 'Preparing image for analysis...'
  });
  const [error, setError] = useState<string | null>(null);

  // Type guards for the result
  const isDetectionResult = (res: AnalysisResult | null): res is DetectionResult => res !== null && 'prediction' in res;
  const isImageResult = (res: AnalysisResult | null): res is ImageAnalysisResult => res !== null && 'objects' in res;

  const isVideoAnalysis = isVideo === 'true';

  useEffect(() => {
    if (imageUri) {
      performAnalysis();
    }
  }, [imageUri]);

  const performAnalysis = async () => {
    try {
      setIsAnalyzing(true);
      setError(null);
      
      if (isVideoAnalysis) {
        // Video deepfake analysis workflow
        setAnalysisProgress({
          step: 'uploading',
          progress: 10,
          message: 'Uploading video to AI service...'
        });

        setAnalysisProgress({
          step: 'processing',
          progress: 30,
          message: 'AI is analyzing video for deepfake detection...'
        });

        const result = await aiService.analyzeVideo({
          videoUri: imageUri,
          detailed: true,
          onProgress: (progress) => {
            setAnalysisProgress(prev => ({ ...prev, progress }));
          }
        });

        setAnalysisProgress({
          step: 'finalizing',
          progress: 90,
          message: 'Finalizing deepfake analysis...'
        });
        
        setAnalysisResult(result);
        
        await storageService.saveAnalysis({
          id: result.session_id,
          imageUri: imageUri,
          result: result as any, // Cast for storage, UI will handle it
          createdAt: new Date().toISOString(),
          isFavorite: false,
          tags: result.prediction === 'FAKE' ? ['deepfake', 'video'] : ['real', 'video']
        });

      } else {
        // Regular image analysis workflow
        setAnalysisProgress({
          step: 'uploading',
          progress: 10,
          message: 'Uploading image to AI service...'
        });

        // Get user preferences for analysis types
        const preferences = await storageService.getUserPreferences();
        
        setAnalysisProgress({
          step: 'processing',
          progress: 30,
          message: 'AI models are analyzing your image...'
        });

        // Perform the actual AI analysis
        const result = await aiService.analyzeImage({
          imageUri: imageUri,
          analysisTypes: preferences.defaultAnalysisTypes as any[],
          options: {
            confidence_threshold: preferences.confidenceThreshold,
            include_bounding_boxes: true,
            include_colors: true,
            include_metadata: true,
            max_objects: 20
          }
        });

        setAnalysisProgress({
          step: 'finalizing',
          progress: 90,
          message: 'Finalizing results...'
        });

        // Save the analysis result
        await storageService.saveAnalysis({
          id: result.id,
          imageUri: imageUri,
          result: result,
          createdAt: new Date().toISOString(),
          isFavorite: false,
          tags: result.tags
        });

        setAnalysisResult(result);
      }

      setAnalysisProgress({
        step: 'complete',
        progress: 100,
        message: isVideoAnalysis ? 'Deepfake analysis complete!' : 'Analysis complete!'
      });

      setIsAnalyzing(false);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      console.error('Analysis failed:', error);
      setError(errorMessage);
      setIsAnalyzing(false);
    }
  };

  const saveResult = () => {
    Alert.alert('Saved', 'Analysis result has been saved to your history.');
  };

  const shareResult = () => {
    Alert.alert('Share', 'Sharing functionality would be implemented here.');
  };

  const retakePhoto = () => {
    router.back();
  };

  const retryAnalysis = () => {
    setError(null);
    performAnalysis();
  };

  const calculateRiskLevel = (confidence: number, confidenceLevel: string): 'low' | 'medium' | 'high' | 'critical' => {
    if (confidence > 0.9 || confidenceLevel === 'HIGH') {
      return 'critical';
    } else if (confidence > 0.7 || confidenceLevel === 'MEDIUM') {
      return 'high';
    } else if (confidence > 0.5) {
      return 'medium';
    }
    return 'low';
  };

  const getRiskColor = (riskLevel: 'low' | 'medium' | 'high' | 'critical' | undefined): string => {
    switch (riskLevel) {
      case 'low':
        return '#10b981'; // Green
      case 'medium':
        return '#f59e0b'; // Yellow
      case 'high':
        return '#f97316'; // Orange
      case 'critical':
        return '#ef4444'; // Red
      default:
        return '#6b7280';      // Gray
    }
  };

  const generateExplanation = (result: DetectionResult): string => {
    if (result.prediction === 'REAL') {
      return `This video appears to be authentic with ${Math.round((1 - result.fake_confidence) * 100)}% confidence. No signs of manipulation were detected.`;
    } else if (result.prediction === 'FAKE') {
      return `This video shows signs of manipulation with ${Math.round(result.fake_confidence * 100)}% confidence. ${result.frames_used} frames were analyzed using ${result.detailed_analysis?.detection_method || 'advanced AI detection'}.`;
    } else {
      return 'Unable to determine the authenticity of this video. Further analysis may be required.';
    }
  };

  const inferManipulationTypes = (result: DetectionResult): string[] => {
    const types: string[] = [];
    if (result.prediction === 'FAKE') {
      if (result.fake_confidence > 0.8) types.push('Face Swap');
      if (result.detailed_analysis?.detection_method) types.push(result.detailed_analysis.detection_method);
      if (types.length === 0) types.push('General Manipulation');
    }
    return types;
  };

  if (error) {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar style="dark" />
        
        <LinearGradient
          colors={['#1e3a8a', '#3b82f6']}
          style={styles.header}
        >
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Ionicons name="arrow-back" size={24} color="white" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Analysis Failed</Text>
        </LinearGradient>

        <View style={styles.errorContainer}>
          <Ionicons name="alert-circle" size={64} color="#ef4444" />
          <Text style={styles.errorTitle}>Analysis Failed</Text>
          <Text style={styles.errorText}>{error}</Text>
          <View style={styles.errorActions}>
            <TouchableOpacity style={styles.primaryRetryButton} onPress={retryAnalysis}>
              <Ionicons name="refresh" size={20} color="white" />
              <Text style={styles.primaryRetryButtonText}>Retry Analysis</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.secondaryButton} onPress={retakePhoto}>
              <Ionicons name="camera" size={20} color="#3b82f6" />
              <Text style={styles.secondaryButtonText}>Take New Photo</Text>
            </TouchableOpacity>
          </View>
        </View>
      </SafeAreaView>
    );
  }

  if (isAnalyzing) {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar style="dark" />
        
        <LinearGradient
          colors={['#1e3a8a', '#3b82f6']}
          style={styles.header}
        >
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Ionicons name="arrow-back" size={24} color="white" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>
            {isVideoAnalysis ? 'Analyzing Video' : 'Analyzing Image'}
          </Text>
        </LinearGradient>

        <View style={styles.loadingContainer}>
          {imageUri && (
            isVideoAnalysis ? (
              <Video 
                source={{ uri: imageUri }} 
                style={styles.analysisImage}
                resizeMode={ResizeMode.COVER}
                shouldPlay={false}
                isLooping={false}
                useNativeControls={false}
              />
            ) : (
              <Image source={{ uri: imageUri }} style={styles.analysisImage} />
            )
          )}
          
          <View style={styles.loadingContent}>
            <View style={styles.loadingSpinner}>
              <Ionicons name="analytics" size={48} color="#3b82f6" />
            </View>
            <Text style={styles.loadingTitle}>AI Analysis in Progress</Text>
            <Text style={styles.loadingText}>
              {analysisProgress.message}
            </Text>
            
            {/* Progress Bar */}
            <View style={styles.progressBarContainer}>
              <View style={[styles.progressBar, { width: `${analysisProgress.progress}%` }]} />
            </View>
            <Text style={styles.progressText}>{analysisProgress.progress}% Complete</Text>
            
            <View style={styles.progressSteps}>
              <View style={styles.progressStep}>
                <View style={[
                  styles.progressDot, 
                  analysisProgress.progress >= 10 && styles.progressDotActive
                ]} />
                <Text style={styles.progressStepText}>Processing Image</Text>
              </View>
              <View style={styles.progressStep}>
                <View style={[
                  styles.progressDot, 
                  analysisProgress.progress >= 30 && styles.progressDotActive
                ]} />
                <Text style={styles.progressStepText}>Object Detection</Text>
              </View>
              <View style={styles.progressStep}>
                <View style={[
                  styles.progressDot, 
                  analysisProgress.progress >= 60 && styles.progressDotActive
                ]} />
                <Text style={styles.progressStepText}>Color Analysis</Text>
              </View>
              <View style={styles.progressStep}>
                <View style={[
                  styles.progressDot, 
                  analysisProgress.progress >= 90 && styles.progressDotActive
                ]} />
                <Text style={styles.progressStepText}>Finalizing</Text>
              </View>
            </View>
          </View>
        </View>
      </SafeAreaView>
    );
  }

  if (!analysisResult) {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar style="dark" />
        <View style={styles.errorContainer}>
          <Ionicons name="alert-circle" size={64} color="#ef4444" />
          <Text style={styles.errorTitle}>Analysis Failed</Text>
          <Text style={styles.errorText}>
            Unable to analyze the image. Please try again.
          </Text>
          <TouchableOpacity style={styles.retryButton} onPress={retakePhoto}>
            <Text style={styles.retryButtonText}>Try Again</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />
      
      <LinearGradient
        colors={['#1e3a8a', '#3b82f6']}
        style={styles.header}
      >
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="white" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Analysis Results</Text>
        <TouchableOpacity onPress={shareResult} style={styles.shareButton}>
          <Ionicons name="share" size={24} color="white" />
        </TouchableOpacity>
      </LinearGradient>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {/* Media Display */}
        {imageUri && (
          <View style={styles.imageContainer}>
            {isVideoAnalysis ? (
              <Video 
                source={{ uri: imageUri }} 
                style={styles.resultImage}
                resizeMode={ResizeMode.COVER}
                shouldPlay={false}
                isLooping={false}
                useNativeControls={true}
              />
            ) : (
              <Image source={{ uri: imageUri }} style={styles.resultImage} />
            )}
          </View>
        )}

        {/* Deepfake Detection Results */}
        {isDetectionResult(analysisResult) && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>
              <Ionicons name="shield-checkmark" size={20} color="#10b981" />  Deepfake Detection
            </Text>
            <View style={[
              styles.deepfakeCard, 
              { borderColor: analysisResult.prediction === 'FAKE' ? '#ef4444' : '#10b981' }
            ]}>
              <View style={styles.deepfakeHeader}>
                <View style={[
                  styles.deepfakeStatus,
                  { backgroundColor: analysisResult.prediction === 'FAKE' ? '#fef2f2' : '#f0fdf4' }
                ]}>
                  <Ionicons 
                    name={analysisResult.prediction === 'FAKE' ? "warning" : "checkmark-circle"} 
                    size={24} 
                    color={analysisResult.prediction === 'FAKE' ? '#ef4444' : '#10b981'} 
                  />
                  <Text style={[
                    styles.deepfakeResult,
                    { color: analysisResult.prediction === 'FAKE' ? '#ef4444' : '#10b981' }
                  ]}>
                    {analysisResult.prediction === 'FAKE' ? 'DEEPFAKE DETECTED' : 'AUTHENTIC VIDEO'}
                  </Text>
                </View>
                <View style={[
                  styles.riskBadge,
                  { backgroundColor: getRiskColor(calculateRiskLevel(analysisResult.fake_confidence, analysisResult.confidence_level)) }
                ]}>
                  <Text style={styles.riskText}>
                    {calculateRiskLevel(analysisResult.fake_confidence, analysisResult.confidence_level).toUpperCase()} RISK
                  </Text>
                </View>
              </View>
              
              <View style={styles.deepfakeMetrics}>
                <View style={styles.metric}>
                  <Text style={styles.metricLabel}>Fake Confidence</Text>
                  <Text style={styles.metricValue}>
                    {Math.round(analysisResult.fake_confidence * 100)}%
                  </Text>
                </View>
                <View style={styles.metric}>
                  <Text style={styles.metricLabel}>Real</Text>
                  <Text style={[styles.metricValue, { color: '#10b981' }]}>
                    {Math.round((1 - analysisResult.fake_confidence) * 100)}%
                  </Text>
                </View>
                <View style={styles.metric}>
                  <Text style={styles.metricLabel}>Frames Used</Text>
                  <Text style={styles.metricValue}>{analysisResult.frames_used}</Text>
                </View>
              </View>
              
              <View style={styles.deepfakeExplanation}>
                <Text style={styles.explanationTitle}>Analysis Details:</Text>
                <Text style={styles.explanationText}>
                  {generateExplanation(analysisResult)}
                </Text>
              </View>
              
              {inferManipulationTypes(analysisResult).length > 0 && (
                <View style={styles.manipulationTypes}>
                  <Text style={styles.explanationTitle}>Detected Manipulations:</Text>
                  <View style={styles.manipulationList}>
                    {inferManipulationTypes(analysisResult).map((type, index) => (
                      <View key={index} style={styles.manipulationTag}>
                        <Text style={styles.manipulationText}>{type}</Text>
                      </View>
                    ))}
                  </View>
                </View>
              )}
            </View>
          </View>
        )}

        {/* Objects Detected (for Image Analysis) */}
        {isImageResult(analysisResult) && analysisResult.objects.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Objects Detected</Text>
          <View style={styles.objectsGrid}>
            {analysisResult.objects.map((object, index) => (
              <View key={index} style={styles.objectCard}>
                <Text style={styles.objectName}>{object.name}</Text>
                <Text style={styles.objectConfidence}>
                  {Math.round(object.confidence * 100)}% confidence
                </Text>
              </View>
            ))}
          </View>
        </View>
        )}

        {isImageResult(analysisResult) && analysisResult.colors.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Color Palette</Text>
          <View style={styles.colorsContainer}>
            {analysisResult.colors.map((color, index) => (
              <View key={index} style={styles.colorItem}>
                <View 
                  style={[styles.colorSwatch, { backgroundColor: color.hex }]} 
                />
                <View style={styles.colorInfo}>
                  <Text style={styles.colorName}>{color.name}</Text>
                  <Text style={styles.colorPercentage}>{color.percentage}%</Text>
                </View>
              </View>
            ))}
          </View>
        </View>
        )}

        {isImageResult(analysisResult) && analysisResult.tags.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Tags</Text>
          <View style={styles.tagsContainer}>
            {analysisResult.tags.map((tag, index) => (
              <View key={index} style={styles.tag}>
                <Text style={styles.tagText}>#{tag}</Text>
              </View>
            ))}
          </View>
        </View>
        )}

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Analysis Details</Text>
          <View style={styles.metadataContainer}>
            {isImageResult(analysisResult) && (
              <>
                <View style={styles.metadataRow}>
                  <Text style={styles.metadataLabel}>Dimensions:</Text>
                  <Text style={styles.metadataValue}>
                    {analysisResult.imageInfo.dimensions.width} × {analysisResult.imageInfo.dimensions.height}
                  </Text>
                </View>
                <View style={styles.metadataRow}>
                  <Text style={styles.metadataLabel}>File Size:</Text>
                  <Text style={styles.metadataValue}>
                    {analysisResult.imageInfo.fileSize 
                      ? `${(analysisResult.imageInfo.fileSize / (1024 * 1024)).toFixed(1)} MB`
                      : 'Unknown'
                    }
                  </Text>
                </View>
                <View style={styles.metadataRow}>
                  <Text style={styles.metadataLabel}>Format:</Text>
                  <Text style={styles.metadataValue}>{analysisResult.imageInfo.format || 'Unknown'}</Text>
                </View>
              </>
            )}
            <View style={styles.metadataRow}>
              <Text style={styles.metadataLabel}>Processing Time:</Text>
              <Text style={styles.metadataValue}>{isImageResult(analysisResult) ? analysisResult.metadata.processing_time : analysisResult.processing_time}ms</Text>
            </View>
            <View style={styles.metadataRow}>
              <Text style={styles.metadataLabel}>Model Version:</Text>
              <Text style={styles.metadataValue}>{isImageResult(analysisResult) ? analysisResult.metadata.model_version : (analysisResult.detailed_analysis?.model_info || 'unknown')}</Text>
            </View>
          </View>
        </View>

        {/* Action Buttons */}
        <View style={styles.actionsContainer}>
          <TouchableOpacity style={styles.primaryButton} onPress={saveResult}>
            <Ionicons name="bookmark" size={20} color="white" />
            <Text style={styles.primaryButtonText}>Save Result</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.secondaryButton} onPress={retakePhoto}>
            <Ionicons name="camera" size={20} color="#3b82f6" />
            <Text style={styles.secondaryButtonText}>Analyze Another</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8fafc',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 24,
    paddingVertical: 16,
  },
  backButton: {
    padding: 8,
  },
  shareButton: {
    padding: 8,
  },
  headerTitle: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  loadingContainer: {
    flex: 1,
    padding: 24,
  },
  analysisImage: {
    width: '100%',
    height: 200,
    borderRadius: 12,
    marginBottom: 32,
  },
  loadingContent: {
    alignItems: 'center',
  },
  loadingSpinner: {
    marginBottom: 24,
  },
  loadingTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1e293b',
    marginBottom: 12,
  },
  loadingText: {
    fontSize: 16,
    color: '#64748b',
    textAlign: 'center',
    lineHeight: 24,
    marginBottom: 32,
  },
  progressSteps: {
    width: '100%',
  },
  progressStep: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  progressDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#e2e8f0',
    marginRight: 12,
  },
  progressDotActive: {
    backgroundColor: '#3b82f6',
  },
  progressStepText: {
    fontSize: 16,
    color: '#374151',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
  },
  errorTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1e293b',
    marginTop: 16,
    marginBottom: 8,
  },
  errorText: {
    fontSize: 16,
    color: '#64748b',
    textAlign: 'center',
    lineHeight: 24,
    marginBottom: 32,
  },
  retryButton: {
    backgroundColor: '#3b82f6',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  retryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  content: {
    flex: 1,
    paddingHorizontal: 24,
  },
  imageContainer: {
    marginVertical: 16,
  },
  resultImage: {
    width: '100%',
    height: 200,
    borderRadius: 12,
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1e293b',
    marginBottom: 12,
  },
  objectsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  objectCard: {
    backgroundColor: 'white',
    padding: 12,
    borderRadius: 8,
    minWidth: '45%',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  objectName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1e293b',
    marginBottom: 4,
  },
  objectConfidence: {
    fontSize: 14,
    color: '#3b82f6',
  },
  colorsContainer: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  colorItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  colorSwatch: {
    width: 32,
    height: 32,
    borderRadius: 16,
    marginRight: 12,
  },
  colorInfo: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  colorName: {
    fontSize: 16,
    color: '#1e293b',
  },
  colorPercentage: {
    fontSize: 14,
    color: '#64748b',
  },
  tagsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  tag: {
    backgroundColor: '#eff6ff',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  tagText: {
    color: '#3b82f6',
    fontSize: 14,
    fontWeight: '500',
  },
  metadataContainer: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  metadataRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  metadataLabel: {
    fontSize: 14,
    color: '#64748b',
  },
  metadataValue: {
    fontSize: 14,
    color: '#1e293b',
    fontWeight: '500',
  },
  actionsContainer: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 32,
  },
  primaryButton: {
    flex: 1,
    backgroundColor: '#3b82f6',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    borderRadius: 12,
    gap: 8,
  },
  primaryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  secondaryButton: {
    flex: 1,
    backgroundColor: 'white',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#3b82f6',
    gap: 8,
  },
  secondaryButtonText: {
    color: '#3b82f6',
    fontSize: 16,
    fontWeight: '600',
  },
  progressBarContainer: {
    width: '100%',
    height: 6,
    backgroundColor: '#e2e8f0',
    borderRadius: 3,
    marginBottom: 8,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#3b82f6',
    borderRadius: 3,
  },
  progressText: {
    fontSize: 14,
    color: '#64748b',
    marginBottom: 24,
    fontWeight: '500',
  },
  errorActions: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 24,
  },
  primaryRetryButton: {
    flex: 1,
    backgroundColor: '#3b82f6',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    borderRadius: 8,
    gap: 8,
  },
  primaryRetryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },

  // Deepfake Detection Styles
  deepfakeCard: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 20,
    marginTop: 16,
    borderWidth: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  deepfakeHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  deepfakeStatus: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    borderRadius: 12,
    flex: 1,
    marginRight: 12,
  },
  deepfakeResult: {
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 8,
  },
  riskBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
  },
  riskText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  deepfakeMetrics: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 16,
    paddingVertical: 12,
    backgroundColor: '#f8fafc',
    borderRadius: 12,
  },
  metric: {
    alignItems: 'center',
  },
  metricLabel: {
    fontSize: 14,
    color: '#64748b',
    marginBottom: 4,
  },
  metricValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#374151',
  },
  deepfakeExplanation: {
    marginTop: 12,
    padding: 12,
    backgroundColor: '#f1f5f9',
    borderRadius: 8,
  },
  explanationTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#374151',
    marginBottom: 8,
  },
  explanationText: {
    fontSize: 14,
    color: '#64748b',
    lineHeight: 20,
  },
  manipulationTypes: {
    marginTop: 12,
  },
  manipulationList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 8,
  },
  manipulationTag: {
    backgroundColor: '#fef2f2',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#fecaca',
  },
  manipulationText: {
    fontSize: 12,
    color: '#dc2626',
    fontWeight: '500',
  },
});
