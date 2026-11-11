# Sync Study frontend

This is a local-only research UI for the FastAPI service, not a deployable public product. It does not provide a detector or predict when the service has no model. The API token is entered into the page and held in memory; the development server proxies `/api` to `127.0.0.1:8000`. Do not expose the Vite server or API publicly. Configuring a model does not establish its provenance, evaluation, calibration, readiness or deployment rights.

Use Node.js 24 for the tested toolchain. With the repository's Python environment and the local API configured as described in the root README, start the API on port 8000. Then:

```sh
cd frontend
npm ci
npm run dev
```

Open the printed `http://127.0.0.1:5173/` address. Upload a consented MP4 up to 100 MB and at most 30 seconds and use the local bearer token configured for the API. An interrupted, failed or unavailable model is shown explicitly. The displayed classifier score is uncalibrated; DTW results are experimental and unvalidated. No user media or token is included in Git. Completed and failed job results can be deleted from the current session; the UI cannot recover the job ID after a page refresh. This UI does not yet offer cancellation, full accessibility certification, production reverse-proxy configuration or public deployment security.

`npm run build` performs a TypeScript check and production bundle; `npm test` runs unit tests. A production static deployment would need its own explicitly secured same-origin `/api` proxy; Vite's development proxy does not apply to the bundle.
