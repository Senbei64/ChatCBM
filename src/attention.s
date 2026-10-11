;- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
;
; One attention head over the KV cache (intengine.IntModel.step), in the log domain:
;
;   S_t  = sum_d +-PRODA[qe_d + ke_dt]          Q.K for t = 0..n-1
;   z_t  = S_t * 2^Zs                           arithmetic shift
;   ez_t = max z - z_t;  pw_t = pow2_16(16G - ez_t) if ez_t <= 16G, otherwise 0
;   ea_t = ez_t + ilog16(sum pw) - 16G          attention weight in log8 (127 = zero)
;   y_d  = sum_t +-PRODA[ea_t + ve_td]          A.V for d = 0..7
;
; The softmax constant is folded into the weights and the scales of q and k, so the shifted
; score z is already in "exponent / 16" units: exp() is the pow2_16 table and the weights a
; come out directly as log8 exponents, ready for the A.V products.
;
; Input:  _att_q[8]  log8 bytes of the query (sign << 7 | e)
;         _att_kb    keys, transposed: row d = CONTEXT bytes, one column per position
;         _att_vb    values: row t = 32 bytes, 8 of them used
;         _att_n     number of positions (1..CONTEXT), _att_zs shift (signed)
; Output: _att_y[8]  int16
;
        .export _att_head, _att_q, _att_kb, _att_vb, _att_n, _att_zs, _att_y, _prodatab
        .import _pow2_16, _ilog16u

        .include "config.inc"
        .assert CONTEXT = 64, error, "the key rows are 64 bytes"

ZERO    = 127

        .zeropage
atp:    .res 2                  ; PRODA + e
atpm:   .res 2                  ; PRODA + e - 128
mx:     .res 2                  ; maximum score
sum:    .res 3                  ; sum of the pw (up to 17 bits)
ls:     .res 2                  ; ilog16(sum) - 16G
ad:     .res 1                  ; counter d / t

        .segment "ALIGNED"
_prodatab: .res 256             ; PROD of the attention, page aligned

        .bss
_att_q:  .res 8
_att_kb: .res 2
_att_vb: .res 2
_att_n:  .res 1
_att_zs: .res 1
_att_y:  .res 16
sl:     .res 64                 ; scores, low and high bytes
sh:     .res 64
ea:     .res 64                 ; attention weights (log8 exponents)
yl:     .res 8
yh:     .res 8

        .code
; atp = PRODA + e, atpm = PRODA + e - 128   (A = e)
.macro SETPTR
        sta atp
        ora #$80
        sta atpm
.endmacro

_att_head:
        lda #>_prodatab
        sta atp+1
        sta atpm+1
        dec atpm+1
        ldx _att_n                  ; clear the scores
        dex
        lda #0
:       sta sl,x
        sta sh,x
        dex
        bpl :-

; ---- Q.K ----------------------------------------------------------------------
        lda #0
        sta ad
qkd:    ldx ad
        lda _att_q,x
        and #$7F
        cmp #ZERO
        bne :+
        jmp qknext                  ; zero element of the query
:       SETPTR
        lda ad                      ; base of the keys: kb + 64 d (16 bits)
        lsr a
        lsr a
        sta kpos+2                  ; d >> 2 (for now)
        lda ad
        and #3
        lsr a
        ror a
        ror a                       ; (d & 3) << 6
        clc
        adc _att_kb
        sta kpos+1
        sta kneg+1
        lda kpos+2
        adc _att_kb+1
        sta kpos+2
        sta kneg+2
        ldx ad
        lda _att_q,x
        ldx _att_n
        dex
        and #$80
        bne qneg
qpos:
kpos:   ldy $FFFF,x
        bmi :+
        lda sl,x
        clc
        adc (atp),y
        sta sl,x
        bcc qp2
        inc sh,x
        bcs qp2
:       lda sl,x
        sec
        sbc (atpm),y
        sta sl,x
        bcs qp2
        dec sh,x
qp2:    dex
        bpl qpos
        jmp qknext
qneg:
kneg:   ldy $FFFF,x
        bmi :+
        lda sl,x
        sec
        sbc (atp),y
        sta sl,x
        bcs qn2
        dec sh,x
        bcc qn2
:       lda sl,x
        clc
        adc (atpm),y
        sta sl,x
        bcc qn2
        inc sh,x
qn2:    dex
        bpl qneg
qknext: inc ad
        lda ad
        cmp #8
        beq :+
        jmp qkd
:

; ---- z = S << Zs, maximum -----------------------------------------------------
        ldx _att_n
        dex
zloop:  ldy _att_zs
        beq zdone
        bmi zright
zl:     asl sl,x
        lda sh,x                    ; not "rol sh,x": sim65 2.19 executes ROL abs,X ($3E) wrongly
        rol a
        sta sh,x
        dey
        bne zl
        beq zdone
zright: lda sh,x                    ; arithmetic shift right (rounded down)
        cmp #$80
        ror sh,x
        ror sl,x
        iny
        bne zright
zdone:  dex
        bpl zloop
        lda sl                      ; maximum (signed)
        sta mx
        lda sh
        sta mx+1
        ldx _att_n
        dex
        beq ezloop0
mloop:  lda sl,x
        cmp mx
        lda sh,x
        sbc mx+1
        bvc :+
        eor #$80
:       bmi mnext                   ; z < max
        lda sl,x
        sta mx
        lda sh,x
        sta mx+1
mnext:  dex
        bne mloop

; ---- ez = max - z, sum of pow2_16(16G - ez) -------------------------------------
ezloop0:
        lda #0
        sta sum
        sta sum+1
        sta sum+2
        ldx _att_n
        dex
        stx ad
ezloop: ldx ad
        sec
        lda mx
        sbc sl,x
        sta sl,x
        lda mx+1
        sbc sh,x
        sta sh,x
        bne eznext                  ; ez >= 256 > 16G: pw = 0
        lda #G16
        cmp sl,x
        bcc eznext                  ; ez > 16G
        sec
        sbc sl,x                    ; A = 16G - ez (0..16G)
        ldx #0
        jsr _pow2_16
        clc
        adc sum
        sta sum
        txa
        adc sum+1
        sta sum+1
        bcc eznext
        inc sum+2
eznext: dec ad
        bpl ezloop

; ---- ls = ilog16(sum) - 16G -------------------------------------------------------
        ldy #0                      ; 17-bit sum: halve it and add 16
        lda sum+2
        beq :+
        lsr sum+2
        ror sum+1
        ror sum
        ldy #16
:       sty ls
        lda sum
        ldx sum+1
        jsr _ilog16u
        clc
        adc ls
        bcc :+
        inx
:       sec
        sbc #G16
        sta ls
        txa
        sbc #0
        sta ls+1                    ; >= 0

; ---- ea = ez + ls, > 126 -> zero --------------------------------------------------
        ldx _att_n
        dex
ealoop: lda sh,x
        bne eazero
        lda ls+1
        bne eazero
        clc
        lda sl,x
        adc ls
        bcs eazero
        cmp #ZERO
        bcc :+
eazero: lda #ZERO
:       sta ea,x
        dex
        bpl ealoop

; ---- A.V ----------------------------------------------------------------------------
        lda #0
        ldx #7
:       sta yl,x
        sta yh,x
        dex
        bpl :-
        lda _att_vb                 ; pointer to row t = 0
        sta vrow+1
        lda _att_vb+1
        sta vrow+2
        lda #0
        sta ad
avt:    ldx ad
        lda ea,x
        cmp #ZERO
        beq avnext                  ; zero weight: skip the row
        SETPTR
        ldx #7
vrow:   ldy $FFFF,x
        bmi :+
        lda yl,x
        clc
        adc (atp),y
        sta yl,x
        bcc av2
        inc yh,x
        bcs av2
:       lda yl,x
        sec
        sbc (atpm),y
        sta yl,x
        bcs av2
        dec yh,x
av2:    dex
        bpl vrow
avnext: clc                         ; next row: +32 bytes
        lda vrow+1
        adc #32
        sta vrow+1
        bcc :+
        inc vrow+2
:       inc ad
        lda ad
        cmp _att_n
        bne avt

        ldx #7                      ; output: int[8]
        ldy #14
:       lda yl,x
        sta _att_y,y
        lda yh,x
        sta _att_y+1,y
        dey
        dey
        dex
        bpl :-
        rts

;-----------------------------------------------------------------------------
