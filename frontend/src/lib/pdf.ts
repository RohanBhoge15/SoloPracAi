/**
 * Open a generated PDF in a new tab on desktop, or report that it could not
 * be opened so the caller can show a button instead.
 *
 * Why this is split by platform rather than just "always window.open":
 *
 *   - A PDF cannot be shown in an in-app modal on iOS. Safari does not
 *     render PDFs inside an iframe, so the "big popup" idea is
 *     desktop-only. On iPhone the file is either downloaded or pushed to a
 *     separate viewer either way.
 *
 *   - `window.open` is only exempt from the popup blocker when it runs
 *     during the user's click. The issue flow is click -> await the API ->
 *     get a presigned URL -> open, so by the time we have the URL it is no
 *     longer a user gesture and mobile browsers block the call. Desktop
 *     browsers are more forgiving, but not reliably.
 *
 * So: desktop opens a new tab, mobile gets a button to tap. The tap *is* a
 * user gesture, so it opens cleanly.
 */

/** Phones and tablets, where a new tab cannot be opened programmatically. */
export function isMobile(): boolean {
  if (typeof navigator === "undefined") return false;
  return (
    /Android|iPhone|iPad|iPod|Mobile|Windows Phone/i.test(navigator.userAgent) ||
    window.innerWidth < 768
  );
}

/**
 * Try to open `url` in a new tab.
 *
 * Returns true when a tab was opened (desktop), false when it was blocked
 * or we are on mobile — the caller should then show a "View PDF" button.
 *
 * `noopener,noreferrer` so the new tab cannot reach back into the app
 * window, which matters because the URL is a presigned bearer token.
 */
export function openPdfInNewTab(url: string): boolean {
  if (isMobile()) return false;

  // Deliberately NOT `window.open(url, "_blank", "noopener,noreferrer")`.
  //
  // That call cannot report success. With `noopener` in the feature string
  // the HTML spec *requires* window.open to return null, whether or not a
  // tab actually opened - the null is there to stop the new page reaching
  // back through `window.opener`. So `tab !== null` was always false, on
  // every desktop browser, forever.
  //
  // The consequence was not cosmetic. This function is what decides
  // between "a tab opened, so close the sheet" and "no tab, so keep the
  // sheet up with a View PDF button". Always-false meant every desktop
  // prescription took the phone branch: a PDF opened in a tab nobody was
  // told about, while the sheet sat there claiming it could not open one.
  //
  // Going via about:blank gives back a handle we are allowed to inspect.
  // `null` now means what it should - the popup blocker stopped us - and
  // the caller can fall back to a button. The opener is cleared before we
  // navigate, which gives the same isolation `noopener` would have.
  const tab = window.open("about:blank", "_blank");
  if (!tab) return false;

  try {
    tab.opener = null;
  } catch {
    // Already cross-origin, so there is no opener to hand it. Nothing to do.
  }
  tab.location.replace(url);
  return true;
}
