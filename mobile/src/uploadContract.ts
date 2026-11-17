export type UploadPhotoFile = Blob & {
  readonly exists: boolean;
  readonly name: string;
  readonly uri: string;
  bytes(): Promise<Uint8Array>;
};

export function assertReadablePhotoFile(
  file: Pick<UploadPhotoFile, "exists" | "name" | "size" | "uri">,
  index: number,
): void {
  const label = `Ảnh #${index + 1}`;
  if (!file.exists) {
    throw new Error(`${label} không còn tồn tại tại ${file.uri}`);
  }
  if (!Number.isFinite(file.size) || file.size <= 0) {
    throw new Error(`${label} rỗng hoặc không đọc được tại ${file.uri}`);
  }
}

export function buildCaptureUploadForm(
  files: readonly UploadPhotoFile[],
  captureMetadata: unknown,
): FormData {
  return buildOrderedPhotoUploadForm(files, captureMetadata, 2, 2);
}

export function buildOrderedPhotoUploadForm(
  files: readonly UploadPhotoFile[],
  captureMetadata: unknown,
  minimumPhotos: number,
  maximumPhotos: number,
): FormData {
  if (files.length < minimumPhotos || files.length > maximumPhotos) {
    throw new Error(`Cần từ ${minimumPhotos} đến ${maximumPhotos} ảnh theo thứ tự chụp`);
  }

  const form = new FormData();
  files.forEach((file, index) => {
    assertReadablePhotoFile(file, index);
    form.append("photos", file, `capture_${index + 1}.jpg`);
  });
  form.append("capture_metadata", JSON.stringify(captureMetadata));
  return form;
}
