export type CaptureWorkflow = "stitch" | "recognition";
export type SweepDirection = "left" | "right";
export type CapturePrimaryAction = "capture" | "process_recognition" | "process_stitch";

export type CaptureActionState = {
  action: CapturePrimaryAction;
  title: string;
  disabled: boolean;
};

export function captureInstruction(
  workflow: CaptureWorkflow,
  photoCount: number,
  sweepDirection: SweepDirection = "right",
  maximumPhotos = 6,
): string {
  if (photoCount === 0) {
    return "Căn mặt kệ vào khung rồi chụp; giữ máy ổn định khi bấm nút.";
  }
  if (photoCount >= maximumPhotos) {
    return workflow === "recognition"
      ? `Đã đủ ${maximumPhotos} ảnh. Bắt đầu ghép và nhận diện.`
      : `Đã đủ ${maximumPhotos} ảnh. Bắt đầu ghép panorama.`;
  }
  const direction = sweepDirection === "right" ? "phải" : "trái";
  if (workflow === "recognition" && photoCount >= 2) {
    return `Có thể nhận diện ngay, hoặc tiếp tục sang ${direction} và giữ phần ảnh cũ trong vùng xanh.`;
  }
  return `Di chuyển sang ${direction}, đưa phần ảnh cũ vào vùng xanh rồi chụp tiếp.`;
}

export function captureActionState(
  workflow: CaptureWorkflow,
  photoCount: number,
  guidanceReady: boolean,
  maximumPhotos = 6,
): CaptureActionState {
  if (!Number.isInteger(photoCount) || photoCount < 0) {
    throw new Error("photoCount must be a non-negative integer");
  }
  if (workflow === "recognition") {
    if (photoCount >= 2) {
      return {
        action: "process_recognition",
        title: `Ghép và nhận diện · ${photoCount} ảnh`,
        disabled: photoCount > maximumPhotos,
      };
    }
    return {
      action: "capture",
      title: `Chụp ảnh ${photoCount + 1}/${maximumPhotos}`,
      disabled: !guidanceReady,
    };
  }
  if (photoCount >= 2) {
    return {
      action: "process_stitch",
      title: `Ghép panorama · ${photoCount} ảnh`,
      disabled: photoCount > maximumPhotos,
    };
  }
  return {
    action: "capture",
    title: `Chụp ảnh ${photoCount + 1}`,
    disabled: !guidanceReady,
  };
}
