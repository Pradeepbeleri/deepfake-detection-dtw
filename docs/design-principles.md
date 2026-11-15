# Design principles

> **Status.** These principles are **implemented** by the React/TypeScript frontend in
> `frontend/`.
>
> **Verified in Chromium (15 November 2026):** visible keyboard focus on every tab stop
> (11/11, 3px outline); all sampled text meets the WCAG 2.2 AA contrast minimum (4.5:1 normal,
> 3:1 large); all visible controls meet the 24×24 px minimum target size of SC 2.5.8; no
> horizontal overflow at 390/768/1440 px; one `h1` with section `h2`s; labelled form controls;
> decorative SVGs hidden from assistive technology; a live region for asynchronous status.
>
> **Not verified:** screen-reader walkthrough with an actual assistive technology,
> reduced-motion behaviour, forced-colors mode, and the remaining WCAG 2.2 AA success criteria.
> This is an automated subset, not a full audit or a certification.

- **Clarity before spectacle:** use plain labels, strong hierarchy, generous spacing and measured typography. Show uncertainty, provenance and limitations next to results.
- **Distinctive restraint:** research-grade dark and light surfaces, subtle teal/cyan accent, deliberate type scale and an original identity. Avoid decorative dashboards, invented charts and gratuitous gradients.
- **Accessible by default:** semantic HTML, visible focus, keyboard operation, sufficient contrast, responsive reflow, screen-reader status announcements and reduced-motion support. Target WCAG 2.2 AA and verify it.
- **Motion with purpose:** use short transitions to explain state changes, respecting system preferences; never animate fake progress.
- **Component discipline:** React/TypeScript with CSS custom properties for color, type, spacing, radii and motion; reusable buttons, forms, status messages, progress and evidence views with accessible variants. Keep domain logic separate from presentation — API access and response validation live in `api.ts`, result rendering in `ResultDetails.tsx`.
- **Trust through behavior:** disclose input limits, retention and model availability; support cancellation and deletion. Align every displayed number and visualization with actual API evidence and its semantics.
