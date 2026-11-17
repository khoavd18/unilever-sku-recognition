import React, { useMemo, useState } from "react";
import { Image, LayoutChangeEvent, StyleSheet, Text, View } from "react-native";
import { formatDetectionOverlayLabel } from "../recognitionDisplay";
import type { Detection } from "../types";

type Props = {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  detections: Detection[];
};

export function PanoramaResult({ imageUrl, imageWidth, imageHeight, detections }: Props) {
  const [layout, setLayout] = useState({ width: 0, height: 280 });
  const geometry = useMemo(() => {
    const scale = Math.min(layout.width / imageWidth, layout.height / imageHeight);
    const renderedWidth = imageWidth * scale;
    const renderedHeight = imageHeight * scale;
    return {
      scale,
      offsetX: (layout.width - renderedWidth) / 2,
      offsetY: (layout.height - renderedHeight) / 2,
    };
  }, [imageHeight, imageWidth, layout]);
  const onLayout = (event: LayoutChangeEvent) => setLayout(event.nativeEvent.layout);
  return (
    <View style={styles.frame} onLayout={onLayout}>
      <Image source={{ uri: imageUrl }} style={StyleSheet.absoluteFill} resizeMode="contain" />
      {detections.map((detection) => {
        const [x1, y1, x2, y2] = detection.bbox;
        return (
          <View
            key={detection.detection_id}
            style={[
              styles.box,
              detection.resolved ? styles.resolved : styles.unresolved,
              {
                left: geometry.offsetX + x1 * geometry.scale,
                top: geometry.offsetY + y1 * geometry.scale,
                width: (x2 - x1) * geometry.scale,
                height: (y2 - y1) * geometry.scale,
              },
            ]}
          >
            <Text style={styles.boxLabel} numberOfLines={2}>
              {formatDetectionOverlayLabel(detection)}
            </Text>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  frame: { width: "100%", height: 280, backgroundColor: "#05070a", overflow: "hidden" },
  box: { position: "absolute", borderWidth: 2 },
  resolved: { borderColor: "#45e6a5" },
  unresolved: { borderColor: "#ffb84d" },
  boxLabel: { color: "white", backgroundColor: "rgba(0,0,0,0.7)", fontSize: 9 },
});
