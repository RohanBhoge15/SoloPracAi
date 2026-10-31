/**
 * One avatar, used by the top bar, Settings, the doctor dashboard and the
 * patient portal.
 *
 * It exists because the same three questions were being answered four
 * times, and the answers had drifted apart:
 *
 *   1. Should the honorific be part of the initials? "Dr Rohan Bhoge"
 *      gave "DR", which is not a name and not a doctor. Handled once,
 *      here, rather than at each call site.
 *   2. How does a photo get loaded? The `<img>` cannot send an
 *      Authorization header, so the URL has to be fetched first and the
 *      tag pointed at object storage. Getting that wrong looks exactly
 *      like "the upload did nothing", because the picture is stored
 *      correctly and simply never displays.
 *   3. What happens when there is no picture, or the load fails? The
 *      answer has to be visible initials, never a hole. Hiding a failed
 *      image is what turned the first version of this into a silent bug.
 */

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

/**
 * Honorifics that are titles, not names.
 *
 * "Dr Rohan Bhoge" is a name, a title and a name. Taking the first word
 * gives the title. The optional dot handles "Dr." and the alternation
 * handles "Prof" as well as "Professor".
 */
const HONORIFIC =
  /^(dr|mr|mrs|ms|miss|prof|professor|sr|shri)\.?\s+/i;

/** The person's first name, with any title removed. */
export function firstName(displayName: string | null | undefined): string {
  const name = (displayName ?? "").trim();
  if (!name) return "";
  const withoutTitle = name.replace(HONORIFIC, "").trim();
  return (withoutTitle || name).split(/\s+/)[0];
}

/**
 * Up to two initials from the *name*, never from a title.
 *
 * "Dr Rohan Bhoge" -> "RB". "Sunita Devi" -> "SD". "R. Menon" -> "RM",
 * because a lone initial still counts as a name part.
 */
export function initialsOf(displayName: string | null | undefined): string {
  const name = (displayName ?? "").trim();
  if (!name) return "?";
  const withoutTitle = name.replace(HONORIFIC, "").trim() || name;
  return (
    withoutTitle
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 2)
      .map((p) => p[0]?.toUpperCase() ?? "")
      .join("") || "?"
  );
}

export default function Avatar({
  name,
  photoUrl,
  photoKey,
  className = "avatar",
  size,
}: {
  name: string | null | undefined;
  /** A presigned object-storage URL, or null. Fetched by the caller
   *  because the caller knows whose account it is: the signed-in user,
   *  a patient, or a doctor in "My doctors". */
  photoUrl: string | null;
  /** Changes whenever the underlying picture changes, so a re-upload is
   *  not served from a stale cache. */
  photoKey?: string | number | null;
  className?: string;
  size?: number;
}) {
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    setSrc(photoUrl);
  }, [photoUrl, photoKey]);

  const style = size ? { width: size, height: size } : undefined;

  if (!src) {
    return (
      <span className={`${className} ${className}--initials`} style={style}>
        {initialsOf(name)}
      </span>
    );
  }
  return (
    <img
      className={`${className} ${className}--img`}
      src={src}
      alt=""
      style={style}
      onError={() => setSrc(null)}
    />
  );
}

/**
 * Fetches the signed-in account's own picture.
 *
 * Hooked separately from <Avatar> because only the account holder may ask
 * for it, so a component that renders someone else's avatar must not
 * reach for this by accident.
 */
export function useOwnPhotoUrl(enabled: boolean): string | null {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    if (!enabled) {
      setUrl(null);
      return;
    }
    api
      .photoUrl()
      .then(({ url: u }) => {
        if (live) setUrl(u);
      })
      .catch(() => {
        // No picture is the normal case, not an error to surface.
        if (live) setUrl(null);
      });
    return () => {
      live = false;
    };
  }, [enabled]);

  return url;
}
