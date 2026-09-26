; YAML tags. No official tags.scm exists upstream; written for smritikosh.
;
; The grammar is tree-sitter-yaml (stream / document / block_mapping_pair),
; already loaded by tree-sitter-language-pack as "yaml".
;
; The capture sits on the pair rather than the key, same as JSON, so the chunk
; carries the value. Three depths, and the section strategy keeps the
; shallowest that fits the embedder window. A short manifest stays one chunk
; per top-level key; a large spec splits along its own keys.
;
; Plain scalars cover Kubernetes manifests. Quoted keys and flow mappings
; (`{a: 1}`) are a different shape and are not captured here.

; ── depth 1: top-level keys (apiVersion, kind, metadata, spec) ───────────────

(document
  (block_node
    (block_mapping
      (block_mapping_pair
        key: (flow_node
          (plain_scalar) @name)) @definition.section)))

; ── depth 2 ──────────────────────────────────────────────────────────────────

(document
  (block_node
    (block_mapping
      (block_mapping_pair
        value: (block_node
          (block_mapping
            (block_mapping_pair
              key: (flow_node
                (plain_scalar) @name)) @definition.subsection))))))

; ── depth 3 ──────────────────────────────────────────────────────────────────

(document
  (block_node
    (block_mapping
      (block_mapping_pair
        value: (block_node
          (block_mapping
            (block_mapping_pair
              value: (block_node
                (block_mapping
                  (block_mapping_pair
                    key: (flow_node
                      (plain_scalar) @name)) @definition.subsubsection)))))))))
