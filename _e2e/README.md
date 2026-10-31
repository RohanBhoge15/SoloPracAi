# _e2e

Browser e2e scripts for SoloPrac. `BASE` defaults to `http://localhost:8080`; set it if nginx is elsewhere.

Run from this folder:

```
node browser.mjs    # full product walkthrough: doctor + patient
node crop.mjs       # photo crop dialog lands in object storage
node documents.mjs  # documents: prescription + weekly report
node photo.mjs      # SHOULD BE SUPERSEDED: photo now goes through cropper
```

`photo.mjs` is expected to FAIL because the Settings page opens the crop dialog before reading an `<img>` from the blob. Its scenario is covered by `crop.mjs`.
