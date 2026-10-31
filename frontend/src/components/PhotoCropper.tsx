/**
 * Crop and position a profile picture before it is uploaded.
 *
 * Without this, a doctor who photographed themselves at arm's length, or
 * a patient whose whole family is in the frame, gets the same crop the
 * file happened to have, and every avatar in the product is a different
 * shape. A square crop centred on the face is what an avatar is for, so
 * the crop is the default rather than an extra step.
 *
 * The image is read locally with `createImageBitmap` and cropped to a
 * canvas. Nothing is uploaded until the doctor confirms, so a cancelled
 * crop leaves no object in storage and no row in the database.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import Cropper, { type Area, type Point } from "react-easy-crop";
import "@/components/cropper.css";

/** Square, because every avatar in the product is a circle. */
const ASPECT = 1;
/** Comfortably under the 4 MB server limit after a PNG round-trip. */
const OUTPUT_SIZE = 512;
const OUTPUT_TYPE = "image/jpeg";
const OUTPUT_QUALITY = 0.9;

export interface CropResult {
  blob: Blob;
  previewUrl: string;
}

export default function PhotoCropper({
  file,
  onCancel,
  onConfirm,
}: {
  file: File;
  onCancel: () => void;
  onConfirm: (result: CropResult) => void;
}) {
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);
  // `crop` is the crop rectangle's *position* (a Point); `croppedArea` is
  // the pixel box produced on completion, which is what the canvas needs.
  // They are different shapes and conflating them is a type error.
  const [crop, setCrop] = useState<Point>({ x: 0, y: 0 });
  const [croppedArea, setCroppedArea] = useState<Area | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // react-easy-crop reports the decoded size, but not a drawable image -
  // see the object URL effect below for why a real element is loaded here
  // instead.
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [ready, setReady] = useState(false);

  // The cropper takes a URL, not a File.
  //
  // `onMediaLoaded` is deliberately not used to get a drawable: it reports
  // a `MediaSize` ({width, height, naturalWidth, naturalHeight}), which is
  // a plain object. Passing that to `ctx.drawImage` throws, the canvas
  // silently produces nothing, and every crop fails with "Could not crop
  // that image" - which is exactly the bug. A real HTMLImageElement has
  // to be loaded here, from the same object URL, for the canvas to have
  // something to draw.
  //
  // Revoked on unmount so a cancelled crop does not pin the decoded image
  // in memory until the tab closes.
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => {
    const url = URL.createObjectURL(file);
    setSrc(url);
    const image = new Image();
    image.onload = () => {
      imageRef.current = image;
      setReady(true);
    };
    image.onerror = () => setError("That file could not be read as an image.");
    image.src = url;
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const onCropComplete = useCallback((_: Area, pixels: Area) => {
    setCroppedArea(pixels);
  }, []);

  /** Crop with a canvas and hand back a blob plus a preview URL. */
  const makeCrop = useCallback(async (): Promise<CropResult | null> => {
    const image = imageRef.current;
    if (!image || !croppedArea) return null;
    if (!image.naturalWidth || !image.naturalHeight) return null;

    const canvas = document.createElement("canvas");
    canvas.width = OUTPUT_SIZE;
    canvas.height = OUTPUT_SIZE;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    const { width, height, x, y } = croppedArea;
    if (width < 1 || height < 1) return null;

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(
      image,
      Math.round(x),
      Math.round(y),
      Math.round(width),
      Math.round(height),
      0,
      0,
      OUTPUT_SIZE,
      OUTPUT_SIZE,
    );

    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, OUTPUT_TYPE, OUTPUT_QUALITY),
    );
    if (!blob) return null;
    return { blob, previewUrl: URL.createObjectURL(blob) };
  }, [croppedArea]);

  async function confirm() {
    setBusy(true);
    setError(null);
    try {
      const result = await makeCrop();
      if (!result) {
        setError("Could not crop that image. Try a different one.");
        return;
      }
      onConfirm(result);
    } catch {
      setError("Could not crop that image. Try a different one.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="cropper-overlay" role="dialog" aria-label="Crop picture">
      <div className="cropper-card card">
        <div className="cropper-head">
          <h2 className="dash-h2">Crop picture</h2>
          <p className="dash-sub">
            Drag to reposition, scroll or pinch to zoom. Saved square, so it
            looks the same everywhere.
          </p>
        </div>

        <div className="cropper-frame">
          <Cropper
            image={src ?? ""}
            crop={crop}
            onCropChange={setCrop}
            zoom={zoom}
            rotation={rotation}
            aspect={ASPECT}
            onZoomChange={setZoom}
            onRotationChange={setRotation}
            onCropComplete={onCropComplete}
            showGrid
            objectFit="contain"
            restrictPosition
          />
        </div>

        <div className="cropper-controls">
          <label className="cropper-zoom">
            <span>Zoom</span>
            <input
              type="range"
              min={1}
              max={3}
              step={0.01}
              value={zoom}
              onChange={(e) => setZoom(Number(e.target.value))}
            />
          </label>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={() => setRotation((r) => (r + 90) % 360)}
          >
            Rotate
          </button>
        </div>

        {error && <div className="callout callout--warn">{error}</div>}

        <div className="cropper-actions">
          <button className="btn btn-ghost btn-sm" onClick={onCancel}>
            Cancel
          </button>
          <button
            className="btn btn-primary btn-sm"
            onClick={confirm}
            disabled={busy || !croppedArea || !ready}
          >
            {busy ? <span className="spinner" /> : "Use this picture"}
          </button>
        </div>
      </div>
    </div>
  );
}
