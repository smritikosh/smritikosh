; Markdown tags. No official tags.scm exists upstream; written for smritikosh.
;
; Capture heading nodes rather than recursive `section` nodes. The Markdown
; chunker uses consecutive heading offsets as non-overlapping boundaries and
; adds the active heading hierarchy to every window. Capturing `section`
; instead would emit each nested subsection once by itself and again through
; every ancestor.

; ── sections ─────────────────────────────────────────────────────────────────

(atx_heading
  heading_content: (inline) @name) @definition.section

(setext_heading
  heading_content: (paragraph) @name) @definition.section

; ── fenced code blocks ───────────────────────────────────────────────────────
; Only blocks that declare a language, since an unlabelled fence has no name to
; key on and the extractor drops nameless matches anyway.

(fenced_code_block
  (info_string
    (language) @name)) @definition.code_block
