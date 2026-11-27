# Sync Study frontend

This is a local-only research UI for the FastAPI service, not a deployable public product. It does not provide a detector or predict when the service has no model. The development server proxies `/api` to `127.0.0.1:8000`. Do not expose the Vite server or API publicly. Configuring a model does not establish its provenance, evaluation, calibration, readiness or deployment rights.

Use Node.js 24 for the tested toolchain. With the repository's Python environment and the local API configured as described in the root README, start the API on port 8000. Then:

```sh
cd frontend
npm ci
npm run dev
```

Open the printed `http://127.0.0.1:5173/` address. Upload a consented MP4 up to 100 MB and at most 30 seconds. An interrupted, failed or unavailable model is shown explicitly. The displayed classifier score is uncalibrated; DTW results are experimental and unvalidated. No user media is included in Git. Completed and failed job results can be deleted from the current session; you can recover a saved job after a refresh if you kept its ID. This UI does not yet offer cancellation, full accessibility certification, production reverse-proxy configuration or public deployment security.

`npm run build` performs a TypeScript check and production bundle; `npm test` runs unit tests. A production static deployment would need its own explicitly secured same-origin `/api` proxy; Vite's development proxy does not apply to the bundle.
