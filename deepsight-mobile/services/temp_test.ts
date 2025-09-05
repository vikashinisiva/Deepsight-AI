import deepSightAIService from './deepSightAIService';

const testAnalyzeVideo = async () => {
  try {
    const result = await deepSightAIService.analyzeVideo({
      videoUri: 'file:///c:/Users/visha/DeepSight_AI/Deepsight-AI/deepsight-mobile/assets/mock.mp4',
      onProgress: (progress) => {
        console.log(`Progress: ${progress}%`);
      },
    });
    console.log('Analysis result:', result);
  } catch (error) {
    console.error('Analysis failed:', error);
  }
};

testAnalyzeVideo();
