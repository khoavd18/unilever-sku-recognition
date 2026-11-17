import assert from "node:assert/strict";
import test from "node:test";

import {
  assertReadablePhotoFile,
  buildCaptureUploadForm,
  buildOrderedPhotoUploadForm,
} from "../src/uploadContract.ts";

function supportedFile(contents, name, uri) {
  const file = new Blob([contents], { type: "image/jpeg" });
  Object.defineProperties(file, {
    exists: { configurable: true, value: true },
    name: { value: name },
    uri: { value: uri },
    bytes: { value: async () => new Uint8Array(await file.arrayBuffer()) },
  });
  return file;
}

test("builds repeated photos in capture order and JSON-string metadata", async () => {
  const first = supportedFile("first-photo", "camera-a.jpg", "file:///camera-a.jpg");
  const second = supportedFile("second-photo", "camera-b.jpg", "file:///camera-b.jpg");
  const metadata = {
    capture_order: [0, 1],
    coverage_complete: false,
    platform: "android",
    photos: [{ uri: first.uri }, { uri: second.uri }],
    guidance_config: { maxMotionDelta: 0.18, stabilityWindowMs: 350 },
  };

  const form = buildCaptureUploadForm([first, second], metadata);
  const parts = form.getAll("photos");

  assert.equal(parts.length, 2);
  assert.equal(parts[0].name, "capture_1.jpg");
  assert.equal(parts[1].name, "capture_2.jpg");
  assert.equal(await parts[0].text(), "first-photo");
  assert.equal(await parts[1].text(), "second-photo");
  assert.deepEqual(JSON.parse(form.get("capture_metadata")), metadata);
  assert.equal(form.has("expected_item_codes"), false);
});

test("builds an ordered N-photo stitch-only upload", async () => {
  const files = [1, 2, 3, 4].map((index) => supportedFile(
    `photo-${index}`,
    `camera-${index}.jpg`,
    `file:///camera-${index}.jpg`,
  ));
  const form = buildOrderedPhotoUploadForm(files, { capture_order: [0, 1, 2, 3] }, 2, 6);
  const parts = form.getAll("photos");
  assert.equal(parts.length, 4);
  assert.deepEqual(await Promise.all(parts.map((part) => part.text())), [
    "photo-1", "photo-2", "photo-3", "photo-4",
  ]);
  assert.deepEqual(JSON.parse(form.get("capture_metadata")), { capture_order: [0, 1, 2, 3] });
});

test("builds a 2-6 photo recognition upload without changing part order", async () => {
  const files = [1, 2, 3].map((index) => supportedFile(
    `recognition-${index}`,
    `camera-${index}.jpg`,
    `file:///camera-${index}.jpg`,
  ));
  const metadata = { capture_order: [0, 1, 2], sweep_direction: "left" };
  const form = buildOrderedPhotoUploadForm(files, metadata, 2, 6);

  assert.deepEqual(
    await Promise.all(form.getAll("photos").map((part) => part.text())),
    ["recognition-1", "recognition-2", "recognition-3"],
  );
  assert.deepEqual(JSON.parse(form.get("capture_metadata")), metadata);
});

test("rejects a missing captured file", () => {
  const missing = supportedFile("first-photo", "missing.jpg", "file:///missing.jpg");
  Object.defineProperty(missing, "exists", { value: false });

  assert.throws(
    () => assertReadablePhotoFile(missing, 0),
    /Ảnh #1 không còn tồn tại/,
  );
});

test("rejects an empty captured file", () => {
  const empty = supportedFile("", "empty.jpg", "file:///empty.jpg");

  assert.throws(
    () => assertReadablePhotoFile(empty, 1),
    /Ảnh #2 rỗng hoặc không đọc được/,
  );
});
