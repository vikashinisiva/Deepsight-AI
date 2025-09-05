/**
 * Storage Service for DeepSight AI
 * Handles local data storage, caching, and offline capabilities
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { ImageAnalysisResult } from './aiService';

export interface StoredAnalysis {
  id: string;
  imageUri: string;
  result: ImageAnalysisResult;
  createdAt: string;
  isFavorite: boolean;
  tags: string[];
}

export interface UserPreferences {
  defaultAnalysisTypes: string[];
  confidenceThreshold: number;
  enableNotifications: boolean;
  autoSaveResults: boolean;
  cacheImagesPermanently: boolean;
  maxStoredResults: number;
}

export interface AppStatistics {
  totalAnalyses: number;
  successfulAnalyses: number;
  averageProcessingTime: number;
  mostDetectedObjects: { [key: string]: number };
  favoriteCategories: { [key: string]: number };
  lastAnalysisDate: string;
}

class StorageService {
  private readonly KEYS = {
    ANALYSES: 'deepsight_analyses',
    PREFERENCES: 'deepsight_preferences',
    STATISTICS: 'deepsight_statistics',
    CACHE: 'deepsight_cache',
    USER_DATA: 'deepsight_user_data'
  };

  private defaultPreferences: UserPreferences = {
    defaultAnalysisTypes: ['deepfake_detection', 'object_detection', 'color_analysis', 'scene_classification'],
    confidenceThreshold: 0.5,
    enableNotifications: true,
    autoSaveResults: true,
    cacheImagesPermanently: false,
    maxStoredResults: 100
  };

  /**
   * Analysis Results Management
   */
  async saveAnalysis(analysis: StoredAnalysis): Promise<void> {
    try {
      const analyses = await this.getAnalyses();
      const updatedAnalyses = [analysis, ...analyses];
      
      // Limit the number of stored results based on user preferences
      const preferences = await this.getUserPreferences();
      const limitedAnalyses = updatedAnalyses.slice(0, preferences.maxStoredResults);
      
      await AsyncStorage.setItem(this.KEYS.ANALYSES, JSON.stringify(limitedAnalyses));
      
      // Update statistics
      await this.updateStatistics(analysis.result);
      
      console.log('✅ Analysis saved successfully:', analysis.id);
    } catch (error) {
      console.error('❌ Failed to save analysis:', error);
      throw error;
    }
  }

  async getAnalyses(): Promise<StoredAnalysis[]> {
    try {
      const stored = await AsyncStorage.getItem(this.KEYS.ANALYSES);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error('❌ Failed to get analyses:', error);
      return [];
    }
  }

  async getAnalysisById(id: string): Promise<StoredAnalysis | null> {
    try {
      const analyses = await this.getAnalyses();
      return analyses.find(analysis => analysis.id === id) || null;
    } catch (error) {
      console.error('❌ Failed to get analysis by ID:', error);
      return null;
    }
  }

  async deleteAnalysis(id: string): Promise<void> {
    try {
      const analyses = await this.getAnalyses();
      const filteredAnalyses = analyses.filter(analysis => analysis.id !== id);
      await AsyncStorage.setItem(this.KEYS.ANALYSES, JSON.stringify(filteredAnalyses));
      console.log('✅ Analysis deleted successfully:', id);
    } catch (error) {
      console.error('❌ Failed to delete analysis:', error);
      throw error;
    }
  }

  async toggleFavorite(id: string): Promise<void> {
    try {
      const analyses = await this.getAnalyses();
      const updatedAnalyses = analyses.map(analysis => 
        analysis.id === id 
          ? { ...analysis, isFavorite: !analysis.isFavorite }
          : analysis
      );
      await AsyncStorage.setItem(this.KEYS.ANALYSES, JSON.stringify(updatedAnalyses));
      console.log('✅ Favorite status toggled for:', id);
    } catch (error) {
      console.error('❌ Failed to toggle favorite:', error);
      throw error;
    }
  }

  async getFavoriteAnalyses(): Promise<StoredAnalysis[]> {
    try {
      const analyses = await this.getAnalyses();
      return analyses.filter(analysis => analysis.isFavorite);
    } catch (error) {
      console.error('❌ Failed to get favorite analyses:', error);
      return [];
    }
  }

  async searchAnalyses(query: string): Promise<StoredAnalysis[]> {
    try {
      const analyses = await this.getAnalyses();
      const lowercaseQuery = query.toLowerCase();
      
      return analyses.filter(analysis => 
        analysis.tags.some(tag => tag.toLowerCase().includes(lowercaseQuery)) ||
        analysis.result.objects.some(obj => obj.name.toLowerCase().includes(lowercaseQuery)) ||
        analysis.result.scenes.some(scene => scene.name.toLowerCase().includes(lowercaseQuery)) ||
        (analysis.result.caption && analysis.result.caption.toLowerCase().includes(lowercaseQuery))
      );
    } catch (error) {
      console.error('❌ Failed to search analyses:', error);
      return [];
    }
  }

  /**
   * User Preferences Management
   */
  async getUserPreferences(): Promise<UserPreferences> {
    try {
      const stored = await AsyncStorage.getItem(this.KEYS.PREFERENCES);
      return stored ? { ...this.defaultPreferences, ...JSON.parse(stored) } : this.defaultPreferences;
    } catch (error) {
      console.error('❌ Failed to get user preferences:', error);
      return this.defaultPreferences;
    }
  }

  async saveUserPreferences(preferences: Partial<UserPreferences>): Promise<void> {
    try {
      const currentPreferences = await this.getUserPreferences();
      const updatedPreferences = { ...currentPreferences, ...preferences };
      await AsyncStorage.setItem(this.KEYS.PREFERENCES, JSON.stringify(updatedPreferences));
      console.log('✅ User preferences saved successfully');
    } catch (error) {
      console.error('❌ Failed to save user preferences:', error);
      throw error;
    }
  }

  /**
   * Statistics Management
   */
  async getStatistics(): Promise<AppStatistics> {
    try {
      const stored = await AsyncStorage.getItem(this.KEYS.STATISTICS);
      if (stored) {
        return JSON.parse(stored);
      }
      
      // Calculate statistics from existing data
      return await this.calculateStatistics();
    } catch (error) {
      console.error('❌ Failed to get statistics:', error);
      return this.getDefaultStatistics();
    }
  }

  private async calculateStatistics(): Promise<AppStatistics> {
    const analyses = await this.getAnalyses();
    
    if (analyses.length === 0) {
      return this.getDefaultStatistics();
    }

    const mostDetectedObjects: { [key: string]: number } = {};
    const favoriteCategories: { [key: string]: number } = {};
    let totalProcessingTime = 0;
    let successfulAnalyses = 0;

    analyses.forEach(analysis => {
      const result = analysis.result as any;
      
      // Check if this is a deepfake result or image analysis result
      const isDeepfakeResult = result.hasOwnProperty('prediction');
      
      // Count successful analyses
      if (isDeepfakeResult) {
        if (result.success) {
          successfulAnalyses++;
        }
        // Track processing time for deepfake results
        totalProcessingTime += result.processing_time || 0;
      } else {
        if (result.objects?.length > 0 || result.colors?.length > 0) {
          successfulAnalyses++;
        }
        // Track processing time for image analysis
        totalProcessingTime += result.metadata?.processing_time || 0;
        
        // Count detected objects (only for image analysis)
        result.objects?.forEach((obj: any) => {
          mostDetectedObjects[obj.name] = (mostDetectedObjects[obj.name] || 0) + 1;
          if (obj.category) {
            favoriteCategories[obj.category] = (favoriteCategories[obj.category] || 0) + 1;
          }
        });
      }
    });

    return {
      totalAnalyses: analyses.length,
      successfulAnalyses,
      averageProcessingTime: totalProcessingTime / analyses.length,
      mostDetectedObjects,
      favoriteCategories,
      lastAnalysisDate: analyses[0]?.createdAt || new Date().toISOString()
    };
  }

  private async updateStatistics(result: ImageAnalysisResult | any): Promise<void> {
    try {
      const stats = await this.getStatistics();
      
      // Update basic counters
      stats.totalAnalyses += 1;
      
      // Check if this is a deepfake result or image analysis result
      const isDeepfakeResult = result.hasOwnProperty('prediction');
      
      if (isDeepfakeResult) {
        // Handle deepfake results
        if (result.success) {
          stats.successfulAnalyses += 1;
        }
        
        // Update average processing time using deepfake processing_time
        const processingTime = result.processing_time || 0;
        const totalTime = stats.averageProcessingTime * (stats.totalAnalyses - 1) + processingTime;
        stats.averageProcessingTime = totalTime / stats.totalAnalyses;
      } else {
        // Handle image analysis results
        if (result.objects?.length > 0 || result.colors?.length > 0) {
          stats.successfulAnalyses += 1;
        }
        
        // Update average processing time
        const processingTime = result.metadata?.processing_time || 0;
        const totalTime = stats.averageProcessingTime * (stats.totalAnalyses - 1) + processingTime;
        stats.averageProcessingTime = totalTime / stats.totalAnalyses;
        
        // Update object counts (only for image analysis)
        result.objects?.forEach((obj: any) => {
          stats.mostDetectedObjects[obj.name] = (stats.mostDetectedObjects[obj.name] || 0) + 1;
          if (obj.category) {
            stats.favoriteCategories[obj.category] = (stats.favoriteCategories[obj.category] || 0) + 1;
          }
        });
      }
      
      // Update last analysis date
      stats.lastAnalysisDate = new Date().toISOString();
      
      await AsyncStorage.setItem(this.KEYS.STATISTICS, JSON.stringify(stats));
    } catch (error) {
      console.error('❌ Failed to update statistics:', error);
    }
  }

  private getDefaultStatistics(): AppStatistics {
    return {
      totalAnalyses: 0,
      successfulAnalyses: 0,
      averageProcessingTime: 0,
      mostDetectedObjects: {},
      favoriteCategories: {},
      lastAnalysisDate: new Date().toISOString()
    };
  }

  /**
   * Cache Management
   */
  async cacheData(key: string, data: any, ttl?: number): Promise<void> {
    try {
      const cacheEntry = {
        data,
        timestamp: Date.now(),
        ttl: ttl || 24 * 60 * 60 * 1000 // 24 hours default
      };
      
      const cache = await this.getCache();
      cache[key] = cacheEntry;
      
      await AsyncStorage.setItem(this.KEYS.CACHE, JSON.stringify(cache));
    } catch (error) {
      console.error('❌ Failed to cache data:', error);
    }
  }

  async getCachedData(key: string): Promise<any | null> {
    try {
      const cache = await this.getCache();
      const entry = cache[key];
      
      if (!entry) return null;
      
      // Check if cache has expired
      if (Date.now() - entry.timestamp > entry.ttl) {
        delete cache[key];
        await AsyncStorage.setItem(this.KEYS.CACHE, JSON.stringify(cache));
        return null;
      }
      
      return entry.data;
    } catch (error) {
      console.error('❌ Failed to get cached data:', error);
      return null;
    }
  }

  private async getCache(): Promise<{ [key: string]: any }> {
    try {
      const stored = await AsyncStorage.getItem(this.KEYS.CACHE);
      return stored ? JSON.parse(stored) : {};
    } catch (error) {
      console.error('❌ Failed to get cache:', error);
      return {};
    }
  }

  async clearCache(): Promise<void> {
    try {
      await AsyncStorage.setItem(this.KEYS.CACHE, JSON.stringify({}));
      console.log('✅ Cache cleared successfully');
    } catch (error) {
      console.error('❌ Failed to clear cache:', error);
    }
  }

  /**
   * Export/Import Data
   */
  async exportData(): Promise<string> {
    try {
      const analyses = await this.getAnalyses();
      const preferences = await this.getUserPreferences();
      const statistics = await this.getStatistics();
      
      const exportData = {
        analyses,
        preferences,
        statistics,
        exportDate: new Date().toISOString(),
        version: '1.0'
      };
      
      return JSON.stringify(exportData, null, 2);
    } catch (error) {
      console.error('❌ Failed to export data:', error);
      throw error;
    }
  }

  async importData(jsonData: string): Promise<void> {
    try {
      const data = JSON.parse(jsonData);
      
      // Validate data structure
      if (!data.analyses || !Array.isArray(data.analyses)) {
        throw new Error('Invalid data format');
      }
      
      // Import analyses
      await AsyncStorage.setItem(this.KEYS.ANALYSES, JSON.stringify(data.analyses));
      
      // Import preferences if available
      if (data.preferences) {
        await AsyncStorage.setItem(this.KEYS.PREFERENCES, JSON.stringify(data.preferences));
      }
      
      // Import statistics if available
      if (data.statistics) {
        await AsyncStorage.setItem(this.KEYS.STATISTICS, JSON.stringify(data.statistics));
      }
      
      console.log('✅ Data imported successfully');
    } catch (error) {
      console.error('❌ Failed to import data:', error);
      throw error;
    }
  }

  /**
   * Clear All Data
   */
  async clearAllData(): Promise<void> {
    try {
      await AsyncStorage.multiRemove([
        this.KEYS.ANALYSES,
        this.KEYS.PREFERENCES,
        this.KEYS.STATISTICS,
        this.KEYS.CACHE
      ]);
      console.log('✅ All data cleared successfully');
    } catch (error) {
      console.error('❌ Failed to clear all data:', error);
      throw error;
    }
  }

  /**
   * Get Storage Usage
   */
  async getStorageUsage(): Promise<{
    totalItems: number;
    totalSize: number;
    breakdown: { [key: string]: number };
  }> {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const deepsightKeys = keys.filter(key => key.startsWith('deepsight_'));
      
      let totalSize = 0;
      const breakdown: { [key: string]: number } = {};
      
      for (const key of deepsightKeys) {
        const value = await AsyncStorage.getItem(key);
        const size = value ? new Blob([value]).size : 0;
        totalSize += size;
        breakdown[key] = size;
      }
      
      return {
        totalItems: deepsightKeys.length,
        totalSize,
        breakdown
      };
    } catch (error) {
      console.error('❌ Failed to get storage usage:', error);
      return { totalItems: 0, totalSize: 0, breakdown: {} };
    }
  }
}

// Export singleton instance
export const storageService = new StorageService();
export default storageService;
