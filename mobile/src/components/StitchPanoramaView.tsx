import React, { useState } from "react";
import { Button, Image, ScrollView, StyleSheet, Text, View } from "react-native";

type Props = {
  imageUri: string;
  imageWidth: number;
  imageHeight: number;
};

export function StitchPanoramaView({ imageUri, imageWidth, imageHeight }: Props) {
  const [zoomed, setZoomed] = useState(false);
  const displayHeight = zoomed ? 420 : 260;
  const displayWidth = Math.max(320, displayHeight * imageWidth / imageHeight);
  return (
    <View style={styles.container}>
      <ScrollView horizontal contentContainerStyle={styles.scrollContent}>
        <Image
          source={{ uri: imageUri }}
          style={{ width: displayWidth, height: displayHeight }}
          resizeMode="contain"
        />
      </ScrollView>
      <Text style={styles.meta}>{imageWidth}×{imageHeight}px · kéo ngang để xem toàn ảnh</Text>
      <Button title={zoomed ? "Thu nhỏ" : "Phóng to"} onPress={() => setZoomed((value) => !value)} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { gap: 6 },
  scrollContent: { minWidth: "100%", backgroundColor: "#05070a" },
  meta: { color: "#9fb0c5", fontSize: 12 },
});
