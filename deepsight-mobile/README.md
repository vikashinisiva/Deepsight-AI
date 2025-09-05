# DeepSight AI Mobile App

A React Native Expo mobile application for AI-powered image analysis. DeepSight AI allows users to capture or upload images and receive intelligent analysis results using advanced machine learning models.

## Features

- 📱 **Image Capture**: Use device camera to capture images for analysis
- 🖼️ **Gallery Upload**: Select images from device gallery
- 🤖 **AI Analysis**: Powered by advanced computer vision models
- 📊 **Results History**: View and manage previous analysis results
- 🌓 **Dark/Light Mode**: Automatic theme switching
- 💾 **Offline Support**: Local storage for analysis history
- 🔒 **Secure**: Privacy-focused design

## Tech Stack

- **React Native** with **Expo SDK 51**
- **TypeScript** for type safety
- **Expo Router** for navigation
- **NativeWind** for styling (Tailwind CSS for React Native)
- **Expo Camera** for image capture
- **Expo Image Picker** for gallery access
- **AsyncStorage** for local data persistence
- **React Native Reanimated** for smooth animations

## Project Structure

```
├── app/                    # App routes and screens
│   ├── (tabs)/            # Tab-based navigation
│   │   ├── index.tsx      # Home screen
│   │   ├── capture.tsx    # Camera/capture screen
│   │   ├── results.tsx    # Results history screen
│   │   └── profile.tsx    # Profile/settings screen
│   └── _layout.tsx        # Root layout
├── components/            # Reusable UI components
├── constants/             # App constants and colors
├── hooks/                 # Custom React hooks
├── services/              # API and service layers
├── utils/                 # Utility functions
└── assets/                # Images, fonts, and other assets
```

## Getting Started

### Prerequisites

- Node.js (v18 or higher)
- npm or yarn
- Expo CLI (`npm install -g @expo/cli`)
- iOS Simulator (Mac) or Android Emulator

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd deepsight-mobile
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Run on your preferred platform:
   - **iOS**: `npm run ios` (Mac only)
   - **Android**: `npm run android`
   - **Web**: `npm run web`

### Development

- The app uses Expo's managed workflow for easier development and deployment
- Hot reloading is enabled for rapid development
- TypeScript provides compile-time type checking

## Configuration

### Camera Permissions

The app requires camera and photo library permissions. These are configured in `app.json`:

```json
{
  "ios": {
    "infoPlist": {
      "NSCameraUsageDescription": "This app uses the camera to capture images for AI analysis.",
      "NSPhotoLibraryUsageDescription": "This app accesses your photo library to select images for AI analysis."
    }
  },
  "android": {
    "permissions": [
      "CAMERA",
      "READ_EXTERNAL_STORAGE",
      "WRITE_EXTERNAL_STORAGE"
    ]
  }
}
```

### Environment Variables

Create a `.env` file in the root directory:

```env
API_BASE_URL=https://your-api-endpoint.com
API_KEY=your-api-key
```

## Building for Production

### Android

```bash
expo build:android
```

### iOS

```bash
expo build:ios
```

## Testing

Run tests with:

```bash
npm test
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## API Integration

The app is designed to integrate with a backend AI service. Update the service layer in `services/` to connect to your specific AI API endpoints.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, email support@deepsight.ai or join our Discord community.

---

Built with ❤️ by the DeepSight AI Team
