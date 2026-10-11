;- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
;
; A page-aligned buffer for the unit tests (weights of the matrix kernel).
;
        .export _wbuf

        .segment "ALIGNED"
_wbuf:  .res 512

;-----------------------------------------------------------------------------
