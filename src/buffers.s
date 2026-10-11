;- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
;
; Activation buffers shared by the modules of the engine.
;
;   _acc    int16[256]  outputs of a layer: 32 values, or the 256 logits
;   _unit   int16[32]   unit of each element of _acc: the value is acc * 2^(unit/16)
;   _x      int16[32]   residual stream, fixed point with R_RES fractional bits
;   _xs     byte[32]    input of a linear layer in log8: sign (bit 7)
;   _xe     byte[32]      and exponent (127 = zero)
;   _kx     int16         and scale: the value is +-2^((kx - e)/16)
;
        .export _acc, _unit, _x, _xs, _xe, _kx

        .bss
_acc:   .res 512
_unit:  .res 64
_x:     .res 64
_xs:    .res 32
_xe:    .res 32
_kx:    .res 2

;-----------------------------------------------------------------------------
