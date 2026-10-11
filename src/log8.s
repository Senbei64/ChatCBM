;- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
;
; Conversions to log8 (sign + 7-bit exponent, 127 = zero):
;
;   to_log8()      _acc[0..31], _unit -> _xs, _xe, _kx
;                  one scale per vector: kx = max L, e = kx - L, with L = ilog16|acc| + unit
;                  (intengine.enc_dynamic): the input of a linear layer
;
;   enc_static()   _acc[0..31], _unit -> out[i * stride] = sign << 7 | e
;                  a fixed scale per head: e = max(K[i / 8] - L, 0) (intengine.enc_static):
;                  q, k and v, the last two stored in the KV cache
;                  parameters: _enc_out (output pointer), _enc_k (pointer to K[4]), _enc_stride
;
; In both, e > 126 becomes zero (127).
;
        .export _to_log8, _enc_static, _enc_out, _enc_k, _enc_stride
        .import _acc, _unit, _xs, _xe, _kx
        .import ilogabs
        .importzp ix, sg, km, t16, pa, pk

ZERO    = 127

        .bss
_enc_out:    .res 2
_enc_k:      .res 2
_enc_stride: .res 1
ltab:   .res 64                 ; L of each element (int16)

        .code
; ---- to_log8 ----------------------------------------------------------------
_to_log8:
        lda #$00
        sta km
        lda #$80
        sta km+1                ; km = -32768: no element yet
        lda #31
        sta ix
tl1:    lda ix
        asl a
        tay
        ldx ix
        lda _acc,y
        ora _acc+1,y
        bne tlnz
        lda #ZERO
        sta _xe,x
        lda #0
        sta _xs,x
        beq tlnext
tlnz:   lda #0
        sta _xe,x
        lda _acc+1,y
        and #$80
        sta _xs,x
        lda _acc,y
        pha
        lda _acc+1,y
        tax
        pla
        jsr ilogabs
        pha
        lda ix
        asl a
        tay
        pla
        clc
        adc _unit,y
        sta ltab,y
        txa
        adc _unit+1,y
        sta ltab+1,y
        lda ltab,y              ; L > km (signed)?
        cmp km
        lda ltab+1,y
        sbc km+1
        bvc :+
        eor #$80
:       bmi tlnext
        lda ltab,y
        sta km
        lda ltab+1,y
        sta km+1
tlnext: dec ix
        bpl tl1
        lda km+1                ; no non-zero element: k = 0
        cmp #$80
        bne :+
        lda km
        bne :+
        sta km+1
:       lda km
        sta _kx
        lda km+1
        sta _kx+1
        ldx #31                 ; e = k - L, > 126 -> zero
tl2:    lda _xe,x
        bne tl2n                ; already zero
        txa
        asl a
        tay
        sec
        lda km
        sbc ltab,y
        sta t16
        lda km+1
        sbc ltab+1,y
        bne tlz
        lda t16
        cmp #ZERO
        bcc :+
tlz:    lda #ZERO
:       sta _xe,x
tl2n:   dex
        bpl tl2
        rts

; ---- enc_static ---------------------------------------------------------------
_enc_static:
        lda _enc_out
        sta pa
        lda _enc_out+1
        sta pa+1
        lda _enc_k
        sta pk
        lda _enc_k+1
        sta pk+1
        lda #0
        sta ix
es1:    lda ix
        asl a
        tay
        lda _acc,y
        ora _acc+1,y
        bne esnz
        lda #ZERO
        bne esst
esnz:   lda _acc,y
        pha
        lda _acc+1,y
        tax
        pla
        jsr ilogabs             ; sg = sign
        pha
        lda ix
        asl a
        tay
        pla
        clc                     ; L = ilog + unit
        adc _unit,y
        sta t16
        txa
        adc _unit+1,y
        sta t16+1
        lda ix                  ; K[i / 8]
        lsr a
        lsr a
        and #$FE
        tay
        sec
        lda (pk),y
        sbc t16
        sta t16
        iny
        lda (pk),y
        sbc t16+1
        bmi es0                 ; e < 0 -> 0
        bne eszero              ; e >= 256
        lda t16
        cmp #ZERO
        bcs eszero
        bcc essg
es0:    lda #0
essg:   sta t16
        lda sg
        and #$80
        ora t16
        jmp esst
eszero: lda #ZERO
esst:   ldy #0
        sta (pa),y
        clc
        lda pa
        adc _enc_stride
        sta pa
        bcc :+
        inc pa+1
:       inc ix
        lda ix
        cmp #32
        bne es1
        rts

;-----------------------------------------------------------------------------
