;- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
;
; Residual connection (intengine: x = sat16(x + to_resid(acc, unit))):
;
;   resid_add()   x[i] = sat16(x[i] + to_fixed(acc[i], unit[i], R_RES)), i = 0..31
;
        .export _resid_add
        .import _acc, _unit, _x
        .import fixabs, addsat
        .importzp ix, sg, t16

        .include "config.inc"

        .code
_resid_add:
        lda #31
        sta ix
ra1:    lda ix
        asl a
        tay
        clc
        lda _unit,y
        adc #<(16 * R_RES)
        sta t16
        lda _unit+1,y
        adc #>(16 * R_RES)
        sta t16+1
        lda _acc,y
        pha
        lda _acc+1,y
        tax
        pla
        jsr fixabs
        beq ranext              ; nothing to add
        bcc rafit
        lda sg                  ; >= 65536: x saturates by the sign
        bmi :+
        lda #$FF
        ldx #$7F
        bne rast
:       lda #0
        ldx #$80
        bne rast                ; always taken (X = $80)
rafit:  lda ix
        asl a
        tay
        lda _x,y
        pha
        lda _x+1,y
        tax
        lda sg
        asl a                   ; carry = sign
        pla
        jsr addsat
rast:   pha
        lda ix
        asl a
        tay
        pla
        sta _x,y
        txa
        sta _x+1,y
ranext: dec ix
        bpl ra1
        rts

;-----------------------------------------------------------------------------
