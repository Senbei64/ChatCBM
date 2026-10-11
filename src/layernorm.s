;- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
;
; LayerNorm with integers only (intengine.IntModel.layernorm):
;
;   layernorm()   _x (residual, 32 x int16) -> _acc[0..31], _unit = kg - 16 F_LN
;
;   d   = x - (sum x >> 5)                       mean: a sum and a shift
;   ed  = log8 exponents of d (one scale)        |d| in log8
;   var = max(sum pow2_16(16F - 2 ed) >> 5, 1)   squaring doubles the exponent
;   r   = (16F - ilog16(var) + 1) >> 1           1/sqrt halves it
;   y   = +-pow2_16(16F - (ed + eg - r)) + b     times the gain g (log8), plus the bias
;
; Parameters: _ln_g (pointer to the 32 gains as log8 bytes), _ln_b (pointer to the 32
; biases, int16, already shifted to F_LN fractional bits), _ln_kg (scale of the gains).
;
        .export _layernorm, _ln_g, _ln_b, _ln_kg
        .import _acc, _unit, _x
        .import _ilog16u, _pow2_16, pow32, neg32, clr32, sat32, add16, signext
        .importzp ix, km, t16, pb, pk, s24

        .include "config.inc"

ZERO    = 127

        .bss
_ln_g:  .res 2
_ln_b:  .res 2
_ln_kg: .res 2
ltab:   .res 64                 ; ilog16 |d|
dabs:   .res 64                 ; |d|
dneg:   .res 32                 ; sign of d
edt:    .res 32                 ; exponents of d
rr:     .res 1                  ; r
vsum:   .res 2

        .code
_layernorm:
        lda #0                  ; sum of the residual (24 bits)
        sta s24
        sta s24+1
        sta s24+2
        ldy #62
ls1:    lda _x+1,y
        jsr signext
        tax
        clc
        lda _x,y
        adc s24
        sta s24
        lda _x+1,y
        adc s24+1
        sta s24+1
        txa
        adc s24+2
        sta s24+2
        dey
        dey
        bpl ls1
        ldx #5                  ; mean = sum >> 5 (arithmetic, rounded down)
:       lda s24+2
        cmp #$80
        ror s24+2
        ror s24+1
        ror s24
        dex
        bne :-
        lda #0                  ; km = maximum of ilog16|d|
        sta km
        sta km+1
        lda #31
        sta ix
ld1:    lda ix                  ; d = x - mean (17 bits)
        asl a
        tay
        lda _x+1,y
        jsr signext
        sta t16+1               ; byte 2 of x
        sec
        lda _x,y
        sbc s24
        sta dabs,y
        lda _x+1,y
        sbc s24+1
        sta dabs+1,y
        lda t16+1
        sbc s24+2               ; byte 2: sign of d
        ldx ix
        and #$80
        sta dneg,x
        beq :+
        lda dabs,y              ; |d| (fits 16 bits)
        eor #$FF
        clc
        adc #1
        sta dabs,y
        lda dabs+1,y
        eor #$FF
        adc #0
        sta dabs+1,y
:       lda dabs,y
        ora dabs+1,y
        bne ldnz
        lda #ZERO
        sta edt,x
        bne ldnext
ldnz:   lda #0
        sta edt,x
        lda dabs,y
        pha
        lda dabs+1,y
        tax
        pla
        jsr _ilog16u
        pha
        lda ix
        asl a
        tay
        pla
        sta ltab,y
        txa
        sta ltab+1,y
        lda ltab,y              ; maximum (values >= 0)
        cmp km
        lda ltab+1,y
        sbc km+1
        bcc ldnext
        lda ltab,y
        sta km
        lda ltab+1,y
        sta km+1
ldnext: dec ix
        bmi :+
        jmp ld1
:
        lda #0                  ; ed = k - Ld; v = sum pow2_16(16F - 2 ed)
        sta vsum
        sta vsum+1
        lda #31
        sta ix
le1:    ldx ix
        lda edt,x
        bne lenext              ; d = 0
        txa
        asl a
        tay
        sec
        lda km
        sbc ltab,y
        sta t16
        lda km+1
        sbc ltab+1,y
        bne lez
        lda t16
        cmp #ZERO
        bcc :+
lez:    lda #ZERO
        sta edt,x
        bne lenext
:       sta edt,x
        asl a                   ; 16F - 2 ed (>= 16F - 252, signed)
        sta t16
        lda #<(16 * F_LN)
        sec
        sbc t16
        pha
        lda #>(16 * F_LN)
        sbc #0
        tax
        pla
        jsr _pow2_16
        clc
        adc vsum
        sta vsum
        txa
        adc vsum+1
        sta vsum+1
lenext: dec ix
        bpl le1
        ldx #5                  ; var = max(v >> 5, 1)
:       lsr vsum+1
        ror vsum
        dex
        bne :-
        lda vsum
        ora vsum+1
        bne :+
        inc vsum
:       lda vsum                ; r = (16F - ilog16(var) + 1) >> 1
        ldx vsum+1
        jsr _ilog16u
        sta t16
        stx t16+1
        sec
        lda #<(16 * F_LN + 1)
        sbc t16
        sta t16
        lda #>(16 * F_LN + 1)
        sbc t16+1
        lsr a
        lda t16
        ror a
        sta rr
        lda _ln_g               ; pointers to g and to the shifted biases
        sta pk
        lda _ln_g+1
        sta pk+1
        lda _ln_b
        sta pb
        lda _ln_b+1
        sta pb+1
        lda #31
        sta ix
ly1:    jsr clr32
        ldy ix
        lda (pk),y              ; g
        and #$7F
        cmp #ZERO
        beq lyb
        ldx ix
        ldy edt,x
        cpy #ZERO
        beq lyb
        clc                     ; e = ed + eg - r, then 16F - e
        adc edt,x
        sta t16
        lda #0
        adc #0
        sta t16+1
        sec
        lda t16
        sbc rr
        sta t16
        lda t16+1
        sbc #0
        sta t16+1
        sec
        lda #<(16 * F_LN)
        sbc t16
        pha
        lda #>(16 * F_LN)
        sbc t16+1
        tax
        pla
        jsr pow32
        ldy ix
        lda (pk),y
        ldx ix
        eor dneg,x
        bpl lyb
        jsr neg32
lyb:    lda ix                  ; + shifted bias, saturation
        asl a
        tay
        lda (pb),y
        pha
        iny
        lda (pb),y
        tax
        pla
        jsr add16
        jsr sat32
        pha
        lda ix
        asl a
        tay
        pla
        sta _acc,y
        txa
        sta _acc+1,y
        dec ix
        bpl ly1
        sec                     ; unit = kg - 16 F_LN for every element
        lda _ln_kg
        sbc #<(16 * F_LN)
        sta t16
        lda _ln_kg+1
        sbc #>(16 * F_LN)
        ldy #62
:       sta _unit+1,y
        pha
        lda t16
        sta _unit,y
        pla
        dey
        dey
        bpl :-
        rts

;-----------------------------------------------------------------------------
