import type { StitchResult } from "@notchip/expo-panoramic-stitcher";

export type NativeStitchProgress = { progress: number; stage: string };

export async function nativeStitcherAvailability(): Promise<{ available: boolean; reason: string }> {
  try {
    const module = await import("@notchip/expo-panoramic-stitcher");
    if (!module.isStitchingAvailable()) {
      return {
        available: false,
        reason: "Native stitcher không có trong runtime này. Expo Go không chứa module; hãy dùng custom development/preview build.",
      };
    }
    return { available: true, reason: "Native OpenCV stitcher sẵn sàng." };
  } catch (error) {
    return {
      available: false,
      reason: `Không nạp được native stitcher: ${String(error)}. Expo Go chỉ hỗ trợ chế độ máy chủ.`,
    };
  }
}

export async function stitchOnDevice(
  uris: readonly string[],
  onProgress: (value: NativeStitchProgress) => void,
): Promise<StitchResult> {
  if (uris.length < 2 || uris.length > 6) {
    throw new Error("Ghép trên điện thoại cần từ 2 đến 6 ảnh");
  }
  const module = await import("@notchip/expo-panoramic-stitcher");
  if (!module.isStitchingAvailable()) {
    throw new Error("Native stitcher không có trong runtime này; Expo Go cần dùng chế độ máy chủ");
  }
  const subscription = module.default.addListener("onStitchProgress", onProgress);
  try {
    const result = await module.stitchImagePaths([...uris], {
      warpMode: "cylindrical",
      blendStrength: 5,
      matchConf: 0.3,
      panoConfidence: 0.7,
      matchNeighbors: 1,
      matchWrap: false,
      outputWidth: 4096,
      autoResize: false,
      jpegQuality: 95,
    });
    if (!result.success) {
      throw new Error(result.errorMessage ?? "Native stitcher báo thất bại");
    }
    if (result.usedCount !== uris.length) {
      throw new Error(
        `Panorama chưa đầy đủ: chỉ dùng ${result.usedCount}/${uris.length} ảnh (index ${result.usedIndices.join(", ")})`,
      );
    }
    return result;
  } finally {
    subscription.remove();
  }
}
