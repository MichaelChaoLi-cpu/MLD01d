# Manuscript Revision Changes

Schema: `kila-revision-changes/v1`

## reviewer-1/comment-3

### part-01

- Location: Methods > Variables, paragraph beginning 'The core explanatory variable'
- Reason: Define multi-hazard exposure explicitly as a distinct-hazard-type count and state what the count does not measure.
- Kila decisions: KILA-D-20260826-001
- Mode: `replace`
- Timestamp: 2026-08-26T07:16:01Z
- Author: Kila
- Markup SHA-256 before: `1c65af3ec15ed427d4018fd1209be5c006fc0d6171d1ddef05bbabd45e676d58`
- Markup SHA-256 after: `bf8d1b94cd26c04dfdf5a69505144ffcb7776d7b670b9c6234f9205d3ec8deb6`
- Revision IDs: `1`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260826T161601809601.reviewer-1-comment-3.part-01.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Formula verification: not applicable
- Before:

~~~~text
The core explanatory variable for environmental pressure is the number of natural disaster types experienced. This is a derived count variable that summarizes multi-hazard exposure.
~~~~

- After:

~~~~text
The core explanatory variable for environmental pressure is the number of natural disaster types experienced. This is a derived count variable that summarizes multi-hazard exposure. Specifically, the count is the number of distinct disaster types reported by a household (observed range, 0-15 across 19 survey categories), rather than a measure of disaster frequency, intensity, timing, or co-occurrence.
~~~~

- Minimal tracked fragments:
  1. `insert`
     - Before: ""
     - After: " Specifically, the count is the number of distinct disaster types reported by a household (observed range, 0-15 across 19 survey categories), rather than a measure of disaster frequency, intensity, timing, or co-occurrence."

### metadata-correction-01

- Location: Whole markup document > tracked-change metadata
- Reason: Human requested that the tracked-change author be anonymous rather than Kila.
- Kila decisions: none (non-substantive metadata correction)
- Mode: `metadata-only`
- Timestamp: 2026-08-26T07:46:57Z
- Author: anonymous
- Markup SHA-256 before: `bf8d1b94cd26c04dfdf5a69505144ffcb7776d7b670b9c6234f9205d3ec8deb6`
- Markup SHA-256 after: `aaf96b72cff99baba0392119906194640e9f28230e2b307b7c74238e56ffff18`
- Revision IDs: `1`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260826T164411753146.tracked-author.Kila-to-anonymous.docx`
- Tracked author before: `Kila`
- Tracked author after: `anonymous`
- Author-only XML verification: `true`
- Manuscript content unchanged: `true`
- Paragraph properties preserved: `true`
- Run content and styles preserved: `true`
