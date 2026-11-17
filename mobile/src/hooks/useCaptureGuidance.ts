import { useEffect, useRef, useState } from "react";
import { AppState } from "react-native";
import {
  Accelerometer,
  AccelerometerMeasurement,
  DeviceMotion,
  DeviceMotionMeasurement,
} from "expo-sensors";
import {
  EMPTY_GUIDANCE_ACCUMULATOR,
  describeGuidanceBlock,
  evaluateGuidanceReading,
  lowPassGravity,
  selectGuidanceSensorSource,
  sensorStaleAfterMs,
} from "../captureGuidance";
import type { GravityVector, GuidanceAccumulator, GuidanceConfig } from "../captureGuidance";
import type { GuidanceState } from "../types";

const STANDARD_GRAVITY_MS2 = 9.80665;

const INITIAL: GuidanceState = {
  available: false,
  reason: "Đang kiểm tra cảm biến…",
  sensorSource: null,
  status: "initializing",
  blockReason: "sensor_initializing",
  reading: null,
  sampledAt: null,
  tiltDegrees: null,
  motionDelta: null,
  upright: false,
  stable: false,
  ready: false,
};

type SensorSubscription = { remove(): void };

async function permissionGranted(sensor: typeof DeviceMotion | typeof Accelerometer): Promise<boolean> {
  try {
    const current = await sensor.getPermissionsAsync();
    if (current.granted) return true;
    if (!current.canAskAgain) return false;
    return (await sensor.requestPermissionsAsync()).granted;
  } catch {
    return true;
  }
}

export function useCaptureGuidance(config: GuidanceConfig): GuidanceState {
  const [state, setState] = useState<GuidanceState>(INITIAL);
  const accumulator = useRef<GuidanceAccumulator>({ ...EMPTY_GUIDANCE_ACCUMULATOR });
  const filteredGravity = useRef<GravityVector | null>(null);
  const lastReadingAt = useRef<number | null>(null);

  useEffect(() => {
    let active = true;
    let sensorSubscription: SensorSubscription | null = null;
    let staleTimer: ReturnType<typeof setInterval> | null = null;
    let appStateSubscription: { remove(): void } | null = null;
    let selectedSource: "device-motion" | "accelerometer" | null = null;
    let sourceStartedAt = 0;
    let accelerometerFallbackReady = false;

    const resetWindow = () => {
      accumulator.current = { ...EMPTY_GUIDANCE_ACCUMULATOR };
      filteredGravity.current = null;
      lastReadingAt.current = null;
    };

    const consumeReading = (raw: GravityVector) => {
      if (!raw.every(Number.isFinite)) {
        resetWindow();
        setState({
          ...INITIAL,
          available: true,
          sensorSource: selectedSource,
          status: "invalid_samples",
          blockReason: "invalid_sensor_vector",
          reason: "Không đọc được cảm biến; bạn vẫn có thể chụp thủ công.",
        });
        return;
      }
      const now = Date.now();
      const gravity = lowPassGravity(
        filteredGravity.current,
        raw,
        config.gravityFilterAlpha ?? 0.2,
      );
      filteredGravity.current = gravity;
      const evaluated = evaluateGuidanceReading(
        gravity,
        now,
        config,
        accumulator.current,
      );
      const description = describeGuidanceBlock(evaluated.guidance);
      lastReadingAt.current = now;
      accumulator.current = evaluated.accumulator;
      setState({
        ...evaluated.guidance,
        ...description,
        sensorSource: selectedSource,
        reading: { x: raw[0], y: raw[1], z: raw[2] },
      });
    };

    const subscribe = (source: "device-motion" | "accelerometer") => {
      sensorSubscription?.remove();
      selectedSource = source;
      sourceStartedAt = Date.now();
      resetWindow();
      setState({
        ...INITIAL,
        available: true,
        sensorSource: source,
        status: "no_samples",
        blockReason: "waiting_for_first_sample",
        reason: "Đang chờ mẫu cảm biến đầu tiên…",
      });
      if (source === "device-motion") {
        DeviceMotion.setUpdateInterval(config.updateIntervalMs);
        sensorSubscription = DeviceMotion.addListener((measurement: DeviceMotionMeasurement) => {
          const value = measurement.accelerationIncludingGravity;
          if (!value || value.x == null || value.y == null || value.z == null) {
            consumeReading([Number.NaN, Number.NaN, Number.NaN]);
            return;
          }
          consumeReading([value.x, value.y, value.z]);
        });
      } else {
        Accelerometer.setUpdateInterval(config.updateIntervalMs);
        sensorSubscription = Accelerometer.addListener((measurement: AccelerometerMeasurement) => {
          consumeReading([
            measurement.x * STANDARD_GRAVITY_MS2,
            measurement.y * STANDARD_GRAVITY_MS2,
            measurement.z * STANDARD_GRAVITY_MS2,
          ]);
        });
      }
    };

    const initialize = async () => {
      resetWindow();
      setState(INITIAL);
      try {
        const deviceMotionAvailable = await DeviceMotion.isAvailableAsync();
        const deviceMotionAllowed = deviceMotionAvailable && await permissionGranted(DeviceMotion);
        const accelerometerAvailable = await Accelerometer.isAvailableAsync();
        const accelerometerAllowed = accelerometerAvailable && await permissionGranted(Accelerometer);
        accelerometerFallbackReady = accelerometerAllowed;
        if (!active) return;
        const source = selectGuidanceSensorSource(deviceMotionAllowed, accelerometerAllowed);
        if (source === null) {
          const permissionDenied = (deviceMotionAvailable && !deviceMotionAllowed)
            || (accelerometerAvailable && !accelerometerAllowed);
          setState({
            ...INITIAL,
            status: permissionDenied ? "permission_denied" : "unavailable",
            blockReason: permissionDenied ? "sensor_permission_denied" : "no_supported_sensor",
            reason: permissionDenied
              ? "Không có quyền cảm biến; bạn vẫn có thể chụp thủ công."
              : "Thiết bị không có cảm biến phù hợp; bạn vẫn có thể chụp thủ công.",
          });
          return;
        }
        subscribe(source);
      } catch (error: unknown) {
        if (!active) return;
        setState({
          ...INITIAL,
          status: "unavailable",
          blockReason: "sensor_initialization_failed",
          reason: `Không khởi tạo được cảm biến: ${String(error)}`,
        });
      }
    };

    void initialize();
    const staleAfter = sensorStaleAfterMs(config);
    staleTimer = setInterval(() => {
      if (!active || selectedSource === null || AppState.currentState !== "active") return;
      const lastReading = lastReadingAt.current;
      if (lastReading === null && Date.now() - sourceStartedAt > Math.max(1500, staleAfter * 2)) {
        if (selectedSource === "device-motion" && accelerometerFallbackReady) {
          subscribe("accelerometer");
          setState((current) => ({
            ...current,
            reason: "DeviceMotion không trả mẫu; đã chuyển sang gia tốc kế…",
            blockReason: "device_motion_first_sample_timeout_fallback",
          }));
        } else {
          sourceStartedAt = Date.now();
          setState((current) => ({
            ...current,
            status: "no_samples",
            blockReason: "first_sample_timeout",
            reason: "Không nhận được mẫu cảm biến; bạn vẫn có thể chụp thủ công.",
            ready: false,
          }));
        }
        return;
      }
      if (lastReading !== null && Date.now() - lastReading > staleAfter) {
        accumulator.current = { ...EMPTY_GUIDANCE_ACCUMULATOR };
        filteredGravity.current = null;
        setState((current) => ({
          ...current,
          status: "stale",
          blockReason: "sensor_sample_stale",
          reason: "Dữ liệu cảm biến đã cũ; bạn vẫn có thể chụp thủ công.",
          stable: false,
          ready: false,
        }));
      }
    }, Math.max(50, Math.min(250, config.updateIntervalMs)));
    appStateSubscription = AppState.addEventListener("change", (nextState) => {
      resetWindow();
      if (nextState === "active") {
        sourceStartedAt = Date.now();
        setState((current) => ({
          ...current,
          status: selectedSource ? "no_samples" : "initializing",
          blockReason: selectedSource ? "waiting_after_foreground" : "sensor_initializing",
          reason: "Đang chờ dữ liệu cảm biến sau khi mở lại ứng dụng…",
          sampledAt: null,
          stable: false,
          ready: false,
        }));
      } else {
        setState((current) => ({
          ...current,
          status: "background",
          blockReason: "app_not_active",
          reason: "Ứng dụng đang ở nền.",
          sampledAt: null,
          stable: false,
          ready: false,
        }));
      }
    });

    return () => {
      active = false;
      sensorSubscription?.remove();
      if (staleTimer !== null) clearInterval(staleTimer);
      appStateSubscription?.remove();
    };
  }, [
    config.gravityFilterAlpha,
    config.maxMotionDelta,
    config.stabilityWindowMs,
    config.updateIntervalMs,
    config.staleAfterMs,
  ]);
  return state;
}
