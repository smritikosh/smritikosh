(call function: (identifier) @name) @call

(call
  function: (attribute
    object: (identifier) @receiver
    attribute: (identifier) @name)) @call

(call
  function: (attribute
    object: (call function: (identifier) @receiver)
    attribute: (identifier) @name)) @call

(import_statement name: (dotted_name) @module) @import

(import_statement
  name: (aliased_import
    name: (dotted_name) @module
    alias: (identifier) @alias)) @import

(import_from_statement
  module_name: (_) @module
  name: (dotted_name) @name) @import

(import_from_statement
  module_name: (_) @module
  name: (aliased_import
    name: (dotted_name) @name
    alias: (identifier) @alias)) @import
