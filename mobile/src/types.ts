export type CapturePhoto = {
  uri: string;
  width: number;
  height: number;
  capturedAt: string;
  screenOrientation: number;
  guidance: GuidanceState;
  exif?: Record<string, unknown>;
};

export type GuidanceState = {
  available: boolean;
  reason?: string;
  sensorSource?: "device-motion" | "accelerometer" | null;
  status?: "initializing" | "unavailable" | "permission_denied" | "no_samples" | "stale" | "invalid_samples" | "moving" | "stabilizing" | "ready" | "background";
  blockReason?: string | null;
  reading?: { x: number; y: number; z: number } | null;
  sampledAt: number | null;
  tiltDegrees: number | null;
  motionDelta: number | null;
  upright: boolean;
  stable: boolean;
  ready: boolean;
};

export type Detection = {
  detection_id: string;
  bbox: [number, number, number, number];
  bbox_coordinate_system: "panorama_pixels_xyxy";
  panorama_size: { width: number; height: number };
  brand: string;
  detector_class_id: number;
  detector_score: number;
  item_code: string | null;
  item_name: string | null;
  similarity_score: number | null;
  resolved: boolean;
  reason: string | null;
  candidate?: { item_code: string; item_name: string; brand?: string } | null;
  gallery_coverage: {
    supported: boolean;
    detected_brand: string;
    supported_brands: string[];
  };
};

export type ProcessResult = {
  session_id: string;
  input_count: number;
  status: "success" | "retake_required" | "partial_recognition_unavailable";
  stitching: {
    success: boolean;
    reason_code?: string | null;
    message: string;
    panorama_url?: string;
    panorama_width?: number;
    panorama_height?: number;
    match_preview_url?: string | null;
  };
  recognition: null | {
    detections: Detection[];
    unresolved_count: number;
    recognition_policy: {
      minimum_similarity: number | null;
      threshold_calibrated: boolean;
      cosine_is_calibrated_probability: false;
      full_gallery_search: true;
      brand_filter: false;
      gallery_supported_brands: string[];
      coverage_guard: string;
    };
  };
  recognition_error?: string | null;
  osa: {
    detected_visible_count: number;
    brand_visible_counts: Record<string, number>;
    resolved_visible_count: number;
    observed_item_codes: string[];
    model_visible_counts: Record<string, number>;
    unresolved_count: number;
    osa_status: string;
    limitations: string[];
  };
};

export type StitchOnlyResult = {
  session_id: string;
  mode: "stitch_only";
  status: "success" | "retake_required";
  input_count: number;
  execution: "server" | "native";
  stitching: {
    success: boolean;
    status?: string;
    reason_code?: string | null;
    message: string;
    panorama_url?: string;
    panorama_uri?: string;
    panorama_width?: number;
    panorama_height?: number;
    used_indices?: number[];
    report_url?: string;
    match_preview_url?: string | null;
  };
};
