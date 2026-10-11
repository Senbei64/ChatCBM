;- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
;
; GELU through a table (intengine.IntModel.gelu):
;
;   gelu()   u = to_fixed(acc[i], unit[i], GELU_FRAC)
;            acc[i] = u if u >= top, 0 if u < -top, GELU[u + top] otherwise
;            unit[i] = -16 GELU_FRAC
;
; top = GELU_RANGE << GELU_FRAC: the table covers [-GELU_RANGE, GELU_RANGE) in steps of
; 2^-GELU_FRAC. _gelu_table points to it (2 top int16 entries, in the model image).
;
        .export _gelu, _gelu_table
        .import _acc, _unit
        .import tofix
        .importzp r32, ix, t16, pk

        .include "config.inc"

GELU_TOP = GELU_RANGE << GELU_FRAC

        .bss
_gelu_table: .res 2

        .code
_gelu:  lda #31
        sta ix
gl1:    lda ix
        asl a
        tay
        clc
        lda _unit,y
        adc #<(16 * GELU_FRAC)
        sta t16
        lda _unit+1,y
        adc #>(16 * GELU_FRAC)
        sta t16+1
        lda _acc,y
        pha
        lda _acc+1,y
        tax
        pla
        jsr tofix               ; r32 = u
        lda r32+3
        bmi glneg
        ora r32+2
        ora r32+1
        bne glid                ; u >= 256 >= top: g = u
        lda r32
        cmp #GELU_TOP
        bcs glid
        clc                     ; index = u + top
        adc #GELU_TOP
        jmp gltab
glneg:  lda r32+3               ; u < 0: needs u >= -top
        and r32+2
        and r32+1
        cmp #$FF
        bne glz
        lda r32
        cmp #<(-GELU_TOP)
        bcc glz
        clc
        adc #GELU_TOP           ; u + top (0..top-1)
gltab:  asl a                   ; index * 2 (up to 2 * 2 top - 2 = 510)
        tay
        lda #0
        rol a
        clc
        adc _gelu_table+1
        sta pk+1
        lda _gelu_table
        sta pk
        lda (pk),y
        sta r32
        iny
        lda (pk),y
        sta r32+1
        jmp glst
glz:    lda #0
        sta r32
        sta r32+1
glid:
glst:   lda ix
        asl a
        tay
        lda r32
        sta _acc,y
        lda r32+1
        sta _acc+1,y
        lda #<(-16 * GELU_FRAC)
        sta _unit,y
        lda #>(-16 * GELU_FRAC)
        sta _unit+1,y
        dec ix
        bmi :+
        jmp gl1
:       rts

;-----------------------------------------------------------------------------
