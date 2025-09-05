import { Ionicons } from '@expo/vector-icons';
import { StyleProp, TextStyle } from 'react-native';

export function IconSymbol({
  name,
  size = 24,
  color,
  style,
}: {
  name: keyof typeof Ionicons.glyphMap;
  size?: number;
  color: string;
  style?: StyleProp<TextStyle>;
}) {
  return (
    <Ionicons
      name={name}
      size={size}
      color={color}
      style={style}
    />
  );
}
