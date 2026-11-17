import { File, Paths } from "expo-file-system";
import * as MediaLibrary from "expo-media-library/legacy";
import * as Sharing from "expo-sharing";

async function localPanoramaUri(imageUri: string): Promise<string> {
  if (/^https?:\/\//i.test(imageUri)) {
    const destination = new File(Paths.cache, `panorama_${Date.now()}.jpg`);
    const downloaded = await File.downloadFileAsync(imageUri, destination);
    return downloaded.uri;
  }
  return imageUri.startsWith("file://") ? imageUri : `file://${imageUri}`;
}

export async function sharePanorama(imageUri: string): Promise<void> {
  if (!await Sharing.isAvailableAsync()) {
    throw new Error("Thiết bị không hỗ trợ bảng chia sẻ tệp");
  }
  const localUri = await localPanoramaUri(imageUri);
  await Sharing.shareAsync(localUri, {
    mimeType: "image/jpeg",
    dialogTitle: "Chia sẻ panorama",
  });
}

export async function savePanorama(imageUri: string): Promise<void> {
  const permission = await MediaLibrary.requestPermissionsAsync(true, ["photo"]);
  if (!permission.granted) {
    throw new Error("Chưa được cấp quyền lưu ảnh vào thư viện");
  }
  await MediaLibrary.saveToLibraryAsync(await localPanoramaUri(imageUri));
}
