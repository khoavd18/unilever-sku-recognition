import React, { useRef, useState } from "react";
import {
  ActivityIndicator, Alert, Button, Image, Platform, Pressable,
  ScrollView, StyleSheet, Text, View,
} from "react-native";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";
import { CameraView, useCameraPermissions } from "expo-camera";
import { File } from "expo-file-system";
import { fetch as expoFetch } from "expo/fetch";
import * as ScreenOrientation from "expo-screen-orientation";
import { StatusBar } from "expo-status-bar";
import { isCaptureGuidanceReady, sensorStaleAfterMs } from "./src/captureGuidance";
import { captureActionState, captureInstruction } from "./src/captureWorkflow";
import type { SweepDirection } from "./src/captureWorkflow";
import { PanoramaResult } from "./src/components/PanoramaResult";
import { StitchPanoramaView } from "./src/components/StitchPanoramaView";
import {
  formatDetectionBBox,
  formatDetectionReason,
  summarizeDetections,
} from "./src/recognitionDisplay";
import { useCaptureGuidance } from "./src/hooks/useCaptureGuidance";
import { savePanorama, sharePanorama } from "./src/sharePanorama";
import { nativeStitcherAvailability, stitchOnDevice } from "./src/stitching/nativeStitcher";
import {
  assertReadablePhotoFile,
  buildOrderedPhotoUploadForm,
} from "./src/uploadContract";
import type { CapturePhoto, ProcessResult, StitchOnlyResult } from "./src/types";

const API_BASE = (process.env.EXPO_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const GUIDANCE = {
  maxMotionDelta: 0.18,
  stabilityWindowMs: 350,
  updateIntervalMs: 100,
  gravityFilterAlpha: 0.2,
};
const MIN_STITCH_PHOTOS = 2;
const MAX_STITCH_PHOTOS = 6;

type Workflow = "stitch" | "recognition";
type StitchExecution = "server" | "native";
type Phase = "capture" | "processing" | "recognition_ready" | "recognition_result" | "stitch_result" | "error";
type RetryAction = "recognition" | "stitch" | null;

const absoluteUrl = (path?: string | null) => path ? `${API_BASE}${path}` : "";

function ShelfCaptureApp() {
  const camera = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [photos, setPhotos] = useState<CapturePhoto[]>([]);
  const [sweepDirection, setSweepDirection] = useState<SweepDirection>("right");
  const [workflow, setWorkflow] = useState<Workflow>("recognition");
  const [stitchExecution, setStitchExecution] = useState<StitchExecution>("server");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [nativeStatus, setNativeStatus] = useState<string>("Custom build bắt buộc cho chế độ trên điện thoại.");
  const [phase, setPhase] = useState<Phase>("capture");
  const [progressText, setProgressText] = useState("");
  const [recognitionResult, setRecognitionResult] = useState<ProcessResult | null>(null);
  const [stitchResult, setStitchResult] = useState<StitchOnlyResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retryAction, setRetryAction] = useState<RetryAction>(null);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const guidance = useCaptureGuidance(GUIDANCE);
  const staleAfter = sensorStaleAfterMs(GUIDANCE);
  const guidanceReady = isCaptureGuidanceReady(guidance, Date.now(), staleAfter);
  const captureLimit = MAX_STITCH_PHOTOS;
  const captureEnabled = photos.length < captureLimit;
  const previous = photos.at(-1);
  const primaryAction = captureActionState(
    workflow,
    photos.length,
    captureEnabled,
    MAX_STITCH_PHOTOS,
  );
  const instruction = captureInstruction(workflow, photos.length, sweepDirection, captureLimit);

  const guidanceText = guidanceReady
    ? "Máy ổn định · sẵn sàng"
    : guidance.available && guidance.status === "moving"
      ? "Di chuyển chậm lại"
      : "Căn kệ trong khung";

  function captureMetadata() {
    return {
      capture_order: photos.map((_, index) => index),
      sweep_direction: sweepDirection,
      coverage_complete: false,
      platform: Platform.OS,
      photos,
      guidance_config: GUIDANCE,
    };
  }

  async function capture() {
    if (!camera.current || photos.length >= captureLimit) return;
    const picture = await camera.current.takePictureAsync({
      quality: 1,
      exif: true,
      skipProcessing: false,
      shutterSound: false,
    });
    if (!picture) return;
    const capturedFile = new File(picture.uri);
    try {
      assertReadablePhotoFile(capturedFile, photos.length);
    } catch (caught) {
      Alert.alert("Không lưu được ảnh", String(caught));
      return;
    }
    const screenOrientation = await ScreenOrientation.getOrientationAsync();
    setPhotos((current) => [...current, {
      uri: picture.uri,
      width: picture.width,
      height: picture.height,
      exif: picture.exif as Record<string, unknown> | undefined,
      capturedAt: new Date().toISOString(),
      screenOrientation,
      guidance,
    }]);
  }

  function runPrimaryAction() {
    if (primaryAction.action === "capture") {
      void capture();
    } else if (primaryAction.action === "process_recognition") {
      void processRecognition();
    } else {
      void processStitch();
    }
  }

  function beginProcessing(action: RetryAction, message: string) {
    setRetryAction(action);
    setProgressText(message);
    setError(null);
    setPhase("processing");
  }

  async function processRecognition() {
    if (photos.length < MIN_STITCH_PHOTOS || photos.length > MAX_STITCH_PHOTOS) return;
    beginProcessing("recognition", `Đang tải ${photos.length} ảnh, ghép và nhận diện trên máy chủ…`);
    try {
      const files = photos.map((photo) => new File(photo.uri));
      const form = buildOrderedPhotoUploadForm(
        files,
        captureMetadata(),
        MIN_STITCH_PHOTOS,
        MAX_STITCH_PHOTOS,
      );
      const response = await expoFetch(`${API_BASE}/api/v1/process`, { method: "POST", body: form });
      const body = await response.json() as ProcessResult | { detail?: string };
      if (!response.ok) throw new Error("detail" in body ? body.detail : `HTTP ${response.status}`);
      const value = body as ProcessResult;
      setRecognitionResult(value);
      setCounts(value.osa.model_visible_counts);
      setPhase("recognition_ready");
    } catch (caught) {
      setError(String(caught));
      setPhase("error");
    }
  }

  async function processStitch() {
    if (photos.length < MIN_STITCH_PHOTOS || photos.length > MAX_STITCH_PHOTOS) return;
    beginProcessing(
      "stitch",
      stitchExecution === "native"
        ? "Đang kiểm tra native stitcher…"
        : "Đang tải ảnh và ghép trên máy chủ…",
    );
    try {
      const files = photos.map((photo) => new File(photo.uri));
      files.forEach((file, index) => assertReadablePhotoFile(file, index));
      if (stitchExecution === "native") {
        const result = await stitchOnDevice(
          photos.map((photo) => photo.uri),
          ({ progress, stage }) => setProgressText(`Ghép trên điện thoại: ${stage} · ${Math.round(progress * 100)}%`),
        );
        const panoramaUri = result.path.startsWith("file://") ? result.path : `file://${result.path}`;
        setStitchResult({
          session_id: `native_${Date.now()}`,
          mode: "stitch_only",
          status: "success",
          input_count: photos.length,
          execution: "native",
          stitching: {
            success: true,
            status: "native_complete",
            message: "Native OpenCV đã dùng đủ ảnh; vẫn cần kiểm tra thủ công seam, ghosting và sản phẩm lặp/mất.",
            panorama_uri: panoramaUri,
            panorama_width: result.width,
            panorama_height: result.height,
            used_indices: result.usedIndices,
          },
        });
      } else {
        const form = buildOrderedPhotoUploadForm(
          files,
          captureMetadata(),
          MIN_STITCH_PHOTOS,
          MAX_STITCH_PHOTOS,
        );
        const response = await expoFetch(`${API_BASE}/api/v1/stitch`, { method: "POST", body: form });
        const body = await response.json() as Omit<StitchOnlyResult, "execution"> | { detail?: string };
        if (!response.ok) throw new Error("detail" in body ? body.detail : `HTTP ${response.status}`);
        setStitchResult({ ...(body as Omit<StitchOnlyResult, "execution">), execution: "server" });
      }
      setPhase("stitch_result");
    } catch (caught) {
      setError(String(caught));
      setPhase("error");
    }
  }

  async function selectNativeExecution() {
    setStitchExecution("native");
    const status = await nativeStitcherAvailability();
    setNativeStatus(status.reason);
  }

  function selectWorkflow(next: Workflow) {
    setWorkflow(next);
  }

  async function retry() {
    if (retryAction === "recognition") await processRecognition();
    if (retryAction === "stitch") await processStitch();
  }

  async function submitCorrections() {
    if (!recognitionResult) return;
    const corrections = Object.entries(counts).map(([item_code, visible_count]) => ({ item_code, visible_count }));
    if (!corrections.length) return;
    try {
      const response = await expoFetch(`${API_BASE}/api/v1/sessions/${recognitionResult.session_id}/corrections`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ actor: "mobile_user", corrections }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      Alert.alert("Đã lưu", "Số lượng chỉnh sửa được lưu riêng với dự đoán mô hình.");
    } catch (caught) {
      Alert.alert("Không lưu được", String(caught));
    }
  }

  function resetSession() {
    setPhotos([]);
    setSweepDirection("right");
    setRecognitionResult(null);
    setStitchResult(null);
    setError(null);
    setPhase("capture");
  }

  if (!permission) return <SafeAreaView style={styles.center}><ActivityIndicator /></SafeAreaView>;
  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.center}>
        <Text style={styles.body}>Cần quyền camera để chụp ảnh kệ.</Text>
        <Button title="Cấp quyền camera" onPress={requestPermission} />
      </SafeAreaView>
    );
  }

  if (phase === "processing") {
    return (
      <SafeAreaView style={styles.center}>
        <ActivityIndicator size="large" />
        <Text style={styles.body}>{progressText}</Text>
        <Text style={styles.muted}>Ảnh gốc vẫn được giữ để thử lại nếu thất bại.</Text>
      </SafeAreaView>
    );
  }

  if (phase === "recognition_ready" && recognitionResult) {
    const summary = summarizeDetections(recognitionResult.recognition?.detections ?? []);
    return (
      <SafeAreaView style={styles.completionScreen}>
        <StatusBar style="light" />
        <View style={styles.completionIcon}>
          <Text style={styles.completionCheck}>✓</Text>
        </View>
        <Text style={styles.completionTitle}>Đã ghép và nhận diện xong</Text>
        <Text style={styles.completionBody}>
          {recognitionResult.recognition
            ? `Model phát hiện ${summary.total} sản phẩm nhìn thấy · ${summary.resolved} sản phẩm đã xác định SKU.`
            : "Ảnh đã ghép xong nhưng kết quả nhận diện chưa khả dụng."}
        </Text>
        <Pressable
          accessibilityRole="button"
          style={({ pressed }) => [styles.resultButton, pressed && styles.buttonPressed]}
          onPress={() => setPhase("recognition_result")}
        >
          <Text style={styles.resultButtonText}>Xem kết quả</Text>
        </Pressable>
        <Pressable
          accessibilityRole="button"
          style={({ pressed }) => [styles.secondaryButton, pressed && styles.buttonPressed]}
          onPress={() => setPhase("capture")}
        >
          <Text style={styles.secondaryButtonText}>Quay lại ảnh đã chụp</Text>
        </Pressable>
        <Pressable accessibilityRole="button" onPress={resetSession}>
          <Text style={styles.textButton}>Chụp phiên mới</Text>
        </Pressable>
      </SafeAreaView>
    );
  }

  if (phase === "stitch_result" && stitchResult) {
    const imageUri = stitchResult.stitching.panorama_uri
      ?? absoluteUrl(stitchResult.stitching.panorama_url);
    return (
      <SafeAreaView style={styles.root}>
        <StatusBar style="light" />
        <ScrollView contentContainerStyle={styles.content}>
          <Text style={styles.title}>Kết quả ghép panorama</Text>
          <Text style={styles.muted}>
            {stitchResult.execution === "native" ? "Xử lý trên điện thoại" : "Xử lý trên máy chủ"}
            {` · ${stitchResult.input_count} ảnh`}
          </Text>
          {stitchResult.stitching.success
            && imageUri
            && stitchResult.stitching.panorama_width
            && stitchResult.stitching.panorama_height ? (
              <StitchPanoramaView
                imageUri={imageUri}
                imageWidth={stitchResult.stitching.panorama_width}
                imageHeight={stitchResult.stitching.panorama_height}
              />
            ) : <Text style={styles.warning}>Cần chụp lại: {stitchResult.stitching.message}</Text>}
          <Text style={styles.warning}>{stitchResult.stitching.message}</Text>
          {imageUri ? <Button title="Lưu panorama vào thư viện" onPress={() => {
            void savePanorama(imageUri)
              .then(() => Alert.alert("Đã lưu", "Panorama đã được lưu vào thư viện ảnh."))
              .catch((caught) => Alert.alert("Không lưu được", String(caught)));
          }} /> : null}
          {imageUri ? <Button title="Chia sẻ panorama" onPress={() => {
            void sharePanorama(imageUri).catch((caught) => Alert.alert("Không chia sẻ được", String(caught)));
          }} /> : null}
          <Button title="Quay lại các ảnh để ghép lại" onPress={() => setPhase("capture")} />
          <Button title="Chụp phiên mới" onPress={resetSession} />
        </ScrollView>
      </SafeAreaView>
    );
  }

  if (phase === "recognition_result" && recognitionResult) {
    const detections = recognitionResult.recognition?.detections ?? [];
    const summary = summarizeDetections(detections);
    return (
      <SafeAreaView style={styles.root}>
        <StatusBar style="light" />
        <ScrollView contentContainerStyle={styles.content}>
          <Pressable accessibilityRole="button" onPress={() => setPhase("recognition_ready")}>
            <Text style={styles.backButtonText}>‹ Quay lại</Text>
          </Pressable>
          <Text style={styles.title}>Kết quả ghép + nhận diện</Text>
          <Text style={styles.section}>Số lượng sản phẩm nhìn thấy</Text>
          <View style={styles.countSummaryRow}>
            <View style={styles.countMetric}>
              <Text style={styles.countMetricValue}>{summary.total}</Text>
              <Text style={styles.countMetricLabel}>Phát hiện</Text>
            </View>
            <View style={styles.countMetric}>
              <Text style={styles.countMetricValue}>{summary.resolved}</Text>
              <Text style={styles.countMetricLabel}>Đã chốt SKU</Text>
            </View>
            <View style={styles.countMetric}>
              <Text style={styles.countMetricValue}>{summary.unresolved}</Text>
              <Text style={styles.countMetricLabel}>Chưa chốt SKU</Text>
            </View>
          </View>
          {summary.byBrand.length ? (
            <View style={styles.brandCountWrap}>
              {summary.byBrand.map(({ brand, count }) => (
                <View key={brand} style={styles.brandCountChip}>
                  <Text style={styles.brandCountName}>{brand}</Text>
                  <Text style={styles.brandCountValue}>{count}</Text>
                </View>
              ))}
            </View>
          ) : <Text style={styles.muted}>Chưa có sản phẩm nào được detector ghi nhận.</Text>}
          {recognitionResult.stitching.success
            && recognitionResult.stitching.panorama_url
            && recognitionResult.stitching.panorama_width
            && recognitionResult.stitching.panorama_height ? (
              <PanoramaResult
                imageUrl={absoluteUrl(recognitionResult.stitching.panorama_url)}
                imageWidth={recognitionResult.stitching.panorama_width}
                imageHeight={recognitionResult.stitching.panorama_height}
                detections={detections}
              />
            ) : <Text style={styles.warning}>Cần chụp lại: {recognitionResult.stitching.message}</Text>}
          {recognitionResult.recognition_error
            ? <Text style={styles.warning}>Nhận diện chưa khả dụng: {recognitionResult.recognition_error}</Text>
            : null}
          {detections.length ? <Text style={styles.section}>Chi tiết phát hiện</Text> : null}
          {detections.map((detection) => (
            <View key={detection.detection_id} style={styles.card}>
              <Text style={styles.cardTitle}>{detection.brand} · {detection.item_code ?? "Chưa có item_code"}</Text>
              <Text style={styles.body}>{detection.item_name ?? formatDetectionReason(detection)}</Text>
              <Text style={styles.muted}>Detector class {detection.detector_class_id}: {detection.brand} · confidence {detection.detector_score.toFixed(3)}</Text>
              <Text style={styles.muted}>BBox: {formatDetectionBBox(detection.bbox)}</Text>
              {detection.similarity_score !== null
                ? <Text style={styles.muted}>Cosine similarity: {detection.similarity_score.toFixed(4)} (không phải xác suất)</Text>
                : null}
            </View>
          ))}
          <Text style={styles.section}>Số lượng theo SKU (có thể sửa)</Text>
          {Object.keys(counts).length ? Object.entries(counts).map(([itemCode, count]) => (
              <View key={itemCode} style={styles.countRow}>
                <Text style={styles.body}>{itemCode}</Text>
                <Pressable style={styles.smallButton} onPress={() => setCounts((value) => ({ ...value, [itemCode]: Math.max(0, count - 1) }))}><Text>−</Text></Pressable>
                <Text style={styles.count}>{count}</Text>
                <Pressable style={styles.smallButton} onPress={() => setCounts((value) => ({ ...value, [itemCode]: count + 1 }))}><Text>+</Text></Pressable>
              </View>
            )) : (
              <Text style={styles.muted}>Chưa có SKU nào được chốt; số đếm theo brand vẫn hiển thị ở phía trên.</Text>
            )}
          <Text style={styles.warning}>Không phát hiện không đồng nghĩa hết hàng. Số lượng nhìn thấy không phải tồn kho ẩn.</Text>
          {Object.keys(counts).length
            ? <Button title="Lưu số lượng chỉnh sửa" onPress={submitCorrections} />
            : null}
          <Button title="Chụp phiên mới" onPress={resetSession} />
        </ScrollView>
      </SafeAreaView>
    );
  }

  if (phase === "error") {
    return (
      <SafeAreaView style={styles.center}>
        <Text style={styles.warning}>Lỗi xử lý: {error}</Text>
        <Button title="Thử lại với các ảnh hiện tại" onPress={() => { void retry(); }} />
        <Button title="Quay lại danh sách ảnh" onPress={() => setPhase("capture")} />
        <Button title="Xóa ảnh và chụp lại" onPress={resetSession} />
      </SafeAreaView>
    );
  }

  const sampleAge = guidance.sampledAt === null ? null : Math.max(0, Date.now() - guidance.sampledAt);
  return (
    <View style={styles.captureScreen}>
      <StatusBar hidden />
      <CameraView
        ref={camera}
        style={StyleSheet.absoluteFill}
        facing="back"
        animateShutter={false}
      />
      {previous ? (
        <>
          <View
            style={[
              styles.previousReferenceWindow,
              sweepDirection === "right" ? styles.previousReferenceLeft : styles.previousReferenceRight,
            ]}
            pointerEvents="none"
          >
            <Image
              source={{ uri: previous.uri }}
              style={[
                styles.previousReferenceImage,
                sweepDirection === "right" ? styles.previousImageRightEdge : styles.previousImageLeftEdge,
              ]}
              resizeMode="cover"
            />
          </View>
          <View
            style={[styles.overlapBand, sweepDirection === "right" ? styles.overlapBandLeft : styles.overlapBandRight]}
            pointerEvents="none"
          />
          <View
            style={[styles.overlapLine, sweepDirection === "right" ? styles.overlapLineLeft : styles.overlapLineRight]}
            pointerEvents="none"
          />
          <Text
            pointerEvents="none"
            style={[styles.directionArrow, sweepDirection === "right" ? styles.arrowRight : styles.arrowLeft]}
          >
            {sweepDirection === "right" ? "→" : "←"}
          </Text>
        </>
      ) : <View style={styles.firstFrameGuide} pointerEvents="none" />}
      <View style={styles.levelLine} pointerEvents="none" />

      <SafeAreaView style={styles.captureSafeArea} edges={["top", "right", "bottom", "left"]} pointerEvents="box-none">
        <View style={styles.captureHeader}>
          <View>
            <Text style={styles.captureTitle}>Chụp panorama</Text>
            <Text style={styles.captureSubtitle}>
              {workflow === "recognition" ? "Ghép + nhận diện" : "Chỉ ghép panorama"}
            </Text>
          </View>
          <Pressable
            accessibilityRole="button"
            style={({ pressed }) => [styles.overlayButton, pressed && styles.buttonPressed]}
            onPress={() => setShowAdvanced((value) => !value)}
          >
            <Text style={styles.overlayButtonText}>{showAdvanced ? "Đóng" : "Tùy chọn"}</Text>
          </Pressable>
        </View>

        {showAdvanced ? (
          <View style={styles.modePanel}>
            <Text style={styles.panelLabel}>Chế độ</Text>
            <View style={styles.modeRow}>
              <Pressable
                accessibilityRole="button"
                style={[styles.optionChip, workflow === "recognition" && styles.optionChipSelected]}
                onPress={() => selectWorkflow("recognition")}
              >
                <Text style={styles.optionChipText}>Ghép + nhận diện</Text>
              </Pressable>
              <Pressable
                accessibilityRole="button"
                style={[styles.optionChip, workflow === "stitch" && styles.optionChipSelected]}
                onPress={() => selectWorkflow("stitch")}
              >
                <Text style={styles.optionChipText}>Chỉ ghép</Text>
              </Pressable>
            </View>
            {workflow === "stitch" ? (
              <>
                <Text style={styles.panelLabel}>Nơi xử lý panorama</Text>
                <View style={styles.modeRow}>
                  <Pressable
                    accessibilityRole="button"
                    style={[styles.optionChip, stitchExecution === "server" && styles.optionChipSelected]}
                    onPress={() => setStitchExecution("server")}
                  >
                    <Text style={styles.optionChipText}>Máy chủ</Text>
                  </Pressable>
                  <Pressable
                    accessibilityRole="button"
                    style={[styles.optionChip, stitchExecution === "native" && styles.optionChipSelected]}
                    onPress={() => { void selectNativeExecution(); }}
                  >
                    <Text style={styles.optionChipText}>Điện thoại</Text>
                  </Pressable>
                </View>
                {stitchExecution === "native" ? <Text style={styles.panelHint}>{nativeStatus}</Text> : null}
              </>
            ) : <Text style={styles.panelHint}>Nhận diện sử dụng detector và gallery trên API server.</Text>}
            {__DEV__ ? (
              <Text style={styles.diagnosticText}>
                Sensor: {guidance.sensorSource ?? "none"} · {guidance.status ?? "unknown"} · motion {guidance.motionDelta?.toFixed(2) ?? "n/a"} · age {sampleAge ?? "n/a"}ms
              </Text>
            ) : null}
          </View>
        ) : null}

        <View style={styles.captureSpacer} pointerEvents="none" />

        <View style={styles.captureBottom}>
          <View style={styles.captureControls}>
            <View style={styles.captureStatusRow}>
              <View style={[styles.guidanceBadge, guidanceReady ? styles.good : styles.advisory]}>
                <Text style={styles.guidanceText}>{guidanceText}</Text>
              </View>
              <Text style={styles.photoCount}>{photos.length}/{captureLimit} ảnh</Text>
            </View>

            <Text style={styles.instructionText}>{instruction}</Text>

            {photos.length === 1 ? (
              <View style={styles.directionRow}>
                <Text style={styles.directionLabel}>Hướng quét:</Text>
                <Pressable
                  accessibilityRole="button"
                  style={[styles.directionButton, sweepDirection === "left" && styles.directionButtonSelected]}
                  onPress={() => setSweepDirection("left")}
                >
                  <Text style={styles.directionButtonText}>← Trái</Text>
                </Pressable>
                <Pressable
                  accessibilityRole="button"
                  style={[styles.directionButton, sweepDirection === "right" && styles.directionButtonSelected]}
                  onPress={() => setSweepDirection("right")}
                >
                  <Text style={styles.directionButtonText}>Phải →</Text>
                </Pressable>
              </View>
            ) : photos.length >= 2 ? (
              <Text style={styles.lockedDirection}>
                Hướng quét: {sweepDirection === "right" ? "sang phải →" : "← sang trái"}
              </Text>
            ) : null}

            {photos.length ? (
              <View>
                <Text style={styles.thumbnailHint}>Chạm vào ảnh để chụp lại từ vị trí đó</Text>
                <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.thumbnails}>
                  {photos.map((photo, index) => (
                    <Pressable
                      accessibilityRole="button"
                      accessibilityLabel={`Chụp lại từ ảnh ${index + 1}`}
                      key={`${photo.uri}-${index}`}
                      style={({ pressed }) => [styles.thumbWrap, pressed && styles.buttonPressed]}
                      onPress={() => setPhotos((value) => value.slice(0, index))}
                    >
                      <Image source={{ uri: photo.uri }} style={styles.thumb} />
                      <View style={styles.thumbIndexBadge}>
                        <Text style={styles.thumbIndexText}>#{index + 1}</Text>
                      </View>
                    </Pressable>
                  ))}
                </ScrollView>
              </View>
            ) : null}
          </View>

          <View style={styles.shutterDock}>
            <View style={styles.shutterSide}>
              {photos.length ? (
                <Pressable
                  accessibilityRole="button"
                  accessibilityLabel="Bỏ ảnh vừa chụp"
                  style={({ pressed }) => [styles.dockAction, pressed && styles.buttonPressed]}
                  onPress={() => setPhotos((value) => value.slice(0, -1))}
                >
                  <Text style={styles.dockActionText}>↶ Bỏ ảnh</Text>
                </Pressable>
              ) : <Text style={styles.dockHint}>Tối đa {captureLimit} ảnh</Text>}
            </View>

            <Pressable
              accessibilityRole="button"
              accessibilityLabel={`Chụp ảnh ${Math.min(photos.length + 1, captureLimit)}`}
              disabled={!captureEnabled}
              style={({ pressed }) => [
                styles.shutterButton,
                !captureEnabled && styles.buttonDisabled,
                pressed && captureEnabled && styles.buttonPressed,
              ]}
              onPress={() => { void capture(); }}
            >
              <View style={styles.shutterButtonInner} />
            </Pressable>

            <View style={styles.shutterSide}>
              {photos.length >= MIN_STITCH_PHOTOS ? (
                <Pressable
                  accessibilityRole="button"
                  accessibilityLabel={primaryAction.title}
                  disabled={primaryAction.disabled}
                  style={({ pressed }) => [
                    styles.processButton,
                    primaryAction.disabled && styles.buttonDisabled,
                    pressed && !primaryAction.disabled && styles.buttonPressed,
                  ]}
                  onPress={runPrimaryAction}
                >
                  <Text style={styles.processButtonText}>
                    {workflow === "recognition" ? "Nhận diện" : "Ghép ảnh"}
                  </Text>
                </Pressable>
              ) : <Text style={styles.dockHint}>Cần ít nhất 2 ảnh</Text>}
            </View>
          </View>
        </View>
      </SafeAreaView>
    </View>
  );
}

export default function App() {
  return (
    <SafeAreaProvider>
      <ShelfCaptureApp />
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: "#0d131d", padding: 12, gap: 8 },
  content: { gap: 12, paddingBottom: 30 },
  center: { flex: 1, backgroundColor: "#0d131d", alignItems: "center", justifyContent: "center", gap: 16, padding: 24 },
  title: { color: "white", fontSize: 24, fontWeight: "700" },
  section: { color: "white", fontSize: 16, fontWeight: "600" },
  body: { color: "#e7edf6", fontSize: 14 },
  muted: { color: "#9fb0c5", fontSize: 12 },
  warning: { color: "#ffbd69", fontSize: 13 },
  captureScreen: { flex: 1, backgroundColor: "#000" },
  captureSafeArea: { flex: 1, justifyContent: "space-between" },
  captureSpacer: { flex: 1 },
  captureHeader: {
    marginHorizontal: 14,
    marginTop: 6,
    paddingHorizontal: 14,
    paddingVertical: 11,
    borderRadius: 18,
    backgroundColor: "rgba(5, 10, 16, 0.72)",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
  },
  captureTitle: { color: "white", fontSize: 21, fontWeight: "800" },
  captureSubtitle: { color: "#c5d2e2", fontSize: 12, marginTop: 2 },
  overlayButton: {
    minHeight: 38,
    paddingHorizontal: 14,
    borderRadius: 19,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.42)",
    backgroundColor: "rgba(255,255,255,0.12)",
    alignItems: "center",
    justifyContent: "center",
  },
  overlayButtonText: { color: "white", fontSize: 13, fontWeight: "700" },
  previousReferenceWindow: {
    position: "absolute", top: 0, bottom: 0, width: "45%", overflow: "hidden", opacity: 0.42,
  },
  previousReferenceLeft: { left: 0 },
  previousReferenceRight: { right: 0 },
  previousReferenceImage: { position: "absolute", top: 0, bottom: 0, width: "222%", height: "100%" },
  previousImageRightEdge: { right: 0 },
  previousImageLeftEdge: { left: 0 },
  overlapBand: {
    position: "absolute", top: 0, bottom: 0, width: "20%", backgroundColor: "rgba(54, 220, 151, 0.18)",
  },
  overlapBandLeft: { left: "35%" },
  overlapBandRight: { right: "35%" },
  overlapLine: {
    position: "absolute", top: 0, bottom: 0, borderColor: "#45e6a5", borderStyle: "dashed",
  },
  overlapLineLeft: { left: "45%", borderLeftWidth: 2 },
  overlapLineRight: { right: "45%", borderRightWidth: 2 },
  directionArrow: {
    position: "absolute", top: "45%", color: "#45e6a5", fontSize: 48, fontWeight: "800",
  },
  arrowRight: { right: 18 },
  arrowLeft: { left: 18 },
  firstFrameGuide: {
    position: "absolute", left: 18, right: 18, top: "18%", bottom: "30%",
    borderWidth: 2, borderColor: "rgba(69,230,165,0.8)", borderRadius: 18,
  },
  levelLine: { position: "absolute", left: "18%", right: "18%", top: "48%", height: 2, backgroundColor: "rgba(255,255,255,0.55)" },
  guidanceBadge: { paddingHorizontal: 11, paddingVertical: 7, borderRadius: 16 },
  good: { backgroundColor: "rgba(25,160,110,0.9)" },
  advisory: { backgroundColor: "rgba(38,69,101,0.92)" },
  guidanceText: { color: "white", fontSize: 12, fontWeight: "700" },
  captureBottom: { paddingHorizontal: 12, paddingBottom: 8, gap: 10 },
  captureControls: {
    padding: 12,
    gap: 9,
    borderRadius: 22,
    backgroundColor: "rgba(5, 9, 14, 0.82)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.14)",
  },
  captureStatusRow: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", gap: 8 },
  photoCount: { color: "white", fontSize: 13, fontWeight: "700" },
  instructionText: { color: "white", textAlign: "center", fontSize: 13, lineHeight: 18, fontWeight: "600" },
  directionRow: { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8 },
  directionLabel: { color: "#c5d2e2", fontSize: 12 },
  directionButton: {
    minHeight: 34,
    paddingHorizontal: 13,
    borderRadius: 17,
    backgroundColor: "rgba(255,255,255,0.1)",
    alignItems: "center",
    justifyContent: "center",
  },
  directionButtonSelected: { backgroundColor: "#16865f" },
  directionButtonText: { color: "white", fontSize: 12, fontWeight: "700" },
  lockedDirection: { color: "#b9c7d8", fontSize: 12, textAlign: "center" },
  thumbnailHint: { color: "#9fb0c5", fontSize: 10, marginBottom: 5 },
  thumbnails: { maxHeight: 54 },
  thumbWrap: {
    width: 62,
    height: 52,
    marginRight: 8,
    borderRadius: 9,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.3)",
  },
  thumb: { width: "100%", height: "100%" },
  thumbIndexBadge: {
    position: "absolute",
    left: 4,
    bottom: 4,
    paddingHorizontal: 5,
    paddingVertical: 2,
    borderRadius: 8,
    backgroundColor: "rgba(0,0,0,0.72)",
  },
  thumbIndexText: { color: "white", fontSize: 10, fontWeight: "800" },
  shutterDock: {
    minHeight: 96,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 28,
    backgroundColor: "rgba(5, 9, 14, 0.9)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.18)",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  shutterSide: { width: 92, alignItems: "center", justifyContent: "center" },
  shutterButton: {
    width: 78,
    height: 78,
    borderRadius: 39,
    padding: 5,
    borderWidth: 4,
    borderColor: "white",
    alignItems: "center",
    justifyContent: "center",
  },
  shutterButtonInner: { width: "100%", height: "100%", borderRadius: 32, backgroundColor: "#22b981" },
  dockAction: {
    minHeight: 42,
    minWidth: 84,
    paddingHorizontal: 9,
    borderRadius: 14,
    backgroundColor: "rgba(255,255,255,0.12)",
    alignItems: "center",
    justifyContent: "center",
  },
  dockActionText: { color: "white", fontSize: 12, fontWeight: "700", textAlign: "center" },
  dockHint: { color: "#9fb0c5", fontSize: 10, lineHeight: 14, textAlign: "center" },
  processButton: {
    minHeight: 46,
    minWidth: 90,
    paddingHorizontal: 10,
    borderRadius: 16,
    backgroundColor: "#22b981",
    alignItems: "center",
    justifyContent: "center",
  },
  processButtonText: { color: "#04130e", fontSize: 12, fontWeight: "800", textAlign: "center" },
  modePanel: {
    marginHorizontal: 14,
    marginTop: 8,
    padding: 12,
    gap: 8,
    borderRadius: 16,
    backgroundColor: "rgba(9, 16, 25, 0.94)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.14)",
  },
  modeRow: { flexDirection: "row", gap: 8 },
  panelLabel: { color: "#e7edf6", fontSize: 12, fontWeight: "700" },
  panelHint: { color: "#9fb0c5", fontSize: 11, lineHeight: 15 },
  optionChip: {
    flex: 1,
    minHeight: 38,
    paddingHorizontal: 10,
    borderRadius: 12,
    backgroundColor: "rgba(255,255,255,0.09)",
    alignItems: "center",
    justifyContent: "center",
  },
  optionChipSelected: { backgroundColor: "#177d5b", borderWidth: 1, borderColor: "#45e6a5" },
  optionChipText: { color: "white", fontSize: 12, fontWeight: "700", textAlign: "center" },
  diagnosticText: { color: "#79d8ff", fontSize: 10 },
  buttonDisabled: { opacity: 0.42 },
  buttonPressed: { opacity: 0.72 },
  completionScreen: {
    flex: 1,
    backgroundColor: "#0d131d",
    alignItems: "center",
    justifyContent: "center",
    padding: 28,
    gap: 16,
  },
  completionIcon: {
    width: 76,
    height: 76,
    borderRadius: 38,
    backgroundColor: "#22b981",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 6,
  },
  completionCheck: { color: "#04130e", fontSize: 40, fontWeight: "900" },
  completionTitle: { color: "white", fontSize: 25, fontWeight: "800", textAlign: "center" },
  completionBody: { color: "#aebed1", fontSize: 14, lineHeight: 20, textAlign: "center", marginBottom: 8 },
  resultButton: {
    width: "100%",
    minHeight: 54,
    borderRadius: 27,
    backgroundColor: "#22b981",
    alignItems: "center",
    justifyContent: "center",
  },
  resultButtonText: { color: "#04130e", fontSize: 16, fontWeight: "800" },
  secondaryButton: {
    width: "100%",
    minHeight: 50,
    borderRadius: 25,
    borderWidth: 1,
    borderColor: "#506177",
    alignItems: "center",
    justifyContent: "center",
  },
  secondaryButtonText: { color: "#e7edf6", fontSize: 14, fontWeight: "700" },
  textButton: { color: "#8fdcbe", fontSize: 14, fontWeight: "700", padding: 8 },
  backButtonText: { color: "#8fdcbe", fontSize: 15, fontWeight: "700", paddingVertical: 4 },
  card: { backgroundColor: "#172231", padding: 12, borderRadius: 10, gap: 3 },
  cardTitle: { color: "#45e6a5", fontSize: 16, fontWeight: "700" },
  countSummaryRow: { flexDirection: "row", gap: 8 },
  countMetric: {
    flex: 1,
    minHeight: 82,
    padding: 10,
    borderRadius: 14,
    backgroundColor: "#172231",
    alignItems: "center",
    justifyContent: "center",
    gap: 4,
  },
  countMetricValue: { color: "#45e6a5", fontSize: 27, fontWeight: "900" },
  countMetricLabel: { color: "#c5d2e2", fontSize: 10, lineHeight: 13, textAlign: "center" },
  brandCountWrap: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  brandCountChip: {
    minHeight: 38,
    paddingLeft: 12,
    paddingRight: 7,
    borderRadius: 19,
    backgroundColor: "#172231",
    borderWidth: 1,
    borderColor: "#2c4056",
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  brandCountName: { color: "#e7edf6", fontSize: 12, fontWeight: "700" },
  brandCountValue: {
    minWidth: 25,
    height: 25,
    borderRadius: 13,
    backgroundColor: "#22b981",
    color: "#04130e",
    fontSize: 12,
    fontWeight: "900",
    textAlign: "center",
    lineHeight: 25,
  },
  countRow: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", gap: 10 },
  smallButton: { width: 34, height: 34, borderRadius: 17, backgroundColor: "#dce8f5", alignItems: "center", justifyContent: "center" },
  count: { color: "white", minWidth: 24, textAlign: "center", fontWeight: "700" },
});
