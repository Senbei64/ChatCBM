;- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
;
; Integer primitives of the engine (python/chatcbm/intengine.py), bit for bit:
;
;   int  __fastcall__ ilog16u(unsigned v);   v > 0 -> 16*n + LOGM[mantissa - 128]  (~16 log2 v)
;   long __fastcall__ pow2_16(int e);        2^(e/16) rounded down = POW[e & 15] * 2^(e>>4 - 15)
;
; pow2_16 saturates to $7FFFFFFF when the result exceeds 2^31. For 0 <= e < 256 the result
; is below 65536 and comes straight from a table (p2lo/p2hi).
;
; Fixed-point helpers shared by the other modules (A/X = low/high byte of an int16):
;
;   negAX     A/X = -A/X
;   ilogabs   A/X = ilog16(|A/X|), A/X != 0; sg = sign (bit 7)
;   pow32     r32 = pow2_16(A/X)
;   neg32     r32 = -r32
;   clr32     r32 = 0
;   sat32     A/X = sat16(r32)
;   add16     r32 += A/X (signed)
;   tofix     r32 = to_fixed(A/X, t16) = sign * pow2_16(ilog16|a| + t16)
;   fixabs    pw = |to_fixed(A/X, t16)| on 16 bits; sg = sign; Z = 1 if a = 0; C = 1 if >= 65536
;   pow2f     pw = 2^(A/X / 16) on 16 bits; C = 1 if the result is >= 65536
;   addsat    A/X = sat16(A/X +/- pw): carry in = operation (0 add, 1 subtract)
;   signext   A = $FF if A has bit 7 set, 0 otherwise
;   sext8     A = signed byte -> A unchanged, X = sign extension
;
; The zero page scratch below is shared: every routine of the engine runs to completion
; before the next one starts, so they can all use it as temporaries.
;
        .export _ilog16u, _pow2_16, powlo, powhi, p2lo, p2hi
        .export negAX, ilogabs, pow32, neg32, clr32, sat32, add16, tofix, fixabs
        .export pow2f, addsat, signext, sext8
        .exportzp r32, ix, sg, km, t16, tfa, pa, pb, pk, s24, pw
        .importzp sreg, tmp1, tmp2, ptr1

        .zeropage
r32:    .res 4                  ; 32-bit result
ix:     .res 1                  ; element index
sg:     .res 1                  ; sign (bit 7)
km:     .res 2                  ; maximum
t16:    .res 2                  ; 16-bit temporary
tfa:    .res 2
pa:     .res 2                  ; pointers
pb:     .res 2
pk:     .res 2
s24:    .res 3                  ; 24-bit sum
pw:     .res 2                  ; 16-bit power of two

        .rodata
        .include "tables.inc"   ; logm_t (128 bytes), powlo, powhi (16 + 16), p2lo, p2hi (256 + 256)

        .code
; ---- ilog16u ----------------------------------------------------------------
_ilog16u:
        cpx #0
        beq lo_only
        stx tmp2
        ldy #7
hi_loop:
        iny                     ; n++
        lsr tmp2
        ror a
        ldx tmp2
        bne hi_loop             ; until the high byte is empty: A = mantissa 128..255
        beq finish
lo_only:
        ldy #7
lo_loop:
        cmp #$80
        bcs finish
        asl a
        dey
        bpl lo_loop             ; always (v > 0)
finish: tax                     ; mantissa
        tya
        asl a
        asl a
        asl a
        asl a                   ; 16 n (n <= 15)
        clc
        adc logm_t-128,x
        ldx #0
        bcc :+
        inx
:       rts

; ---- pow2_16 ----------------------------------------------------------------
_pow2_16:
        cpx #0                  ; 0 <= e < 256: table (result < 65536)
        bne :+
        tay
        lda #0
        sta sreg
        sta sreg+1
        lda p2lo,y
        ldx p2hi,y
        rts
:       bpl p2slow
        lda #0                  ; e < 0: 2^(e/16) < 1 -> 0
        sta sreg
        sta sreg+1
        tax
        rts
p2slow: sta tmp1
        and #15
        tay
        lda powlo,y
        sta ptr1
        lda powhi,y
        sta ptr1+1
        lda #0
        sta sreg
        sta sreg+1
        txa                     ; e >> 4, arithmetic (rounded down)
        .repeat 4
        cmp #$80
        ror a
        ror tmp1
        .endrepeat
        tax
        lda tmp1
        sec
        sbc #15
        sta tmp1                ; n = (e >> 4) - 15, low byte
        txa
        sbc #0                  ; high byte
        bmi right
        bne big                 ; n >= 256
        ldy tmp1
        beq done                ; n = 0
        cpy #16
        bcs big
left:   asl ptr1                ; 32 bits << n
        rol ptr1+1
        rol sreg
        rol sreg+1
        dey
        bne left
        beq done
right:  cmp #$FF
        bne zero                ; n < -256
        lda tmp1
        cmp #$F0                ; n >= -16 ?
        bcc zero
        eor #$FF
        tay
        iny                     ; Y = -n (1..16)
rloop:  lsr ptr1+1
        ror ptr1
        dey
        bne rloop
done:   lda ptr1
        ldx ptr1+1
        rts
zero:   lda #0
        tax
        rts
big:    lda #$FF
        sta sreg
        lda #$7F
        sta sreg+1
        lda #$FF
        tax
        rts

; ---- fixed-point helpers ------------------------------------------------------
negAX:  eor #$FF
        clc
        adc #1
        pha
        txa
        eor #$FF
        adc #0
        tax
        pla
        rts

ilogabs:
        stx sg
        cpx #$80
        bcc :+
        jsr negAX
:       jmp _ilog16u

pow32:  jsr _pow2_16
        sta r32
        stx r32+1
        lda sreg
        sta r32+2
        lda sreg+1
        sta r32+3
        rts

neg32:  ldx #0
        sec
        .repeat 4, I
        txa
        sbc r32+I
        sta r32+I
        .endrepeat
        rts

clr32:  lda #0
        sta r32
        sta r32+1
        sta r32+2
        sta r32+3
        rts

sat32:  lda r32+3
        bmi sneg
        ora r32+2
        bne smax
        lda r32+1
        bmi smax
        tax
        lda r32
        rts
sneg:   cmp #$FF
        bne smin
        lda r32+2
        cmp #$FF
        bne smin
        lda r32+1
        bpl smin
        tax
        lda r32
        rts
smax:   lda #$FF
        ldx #$7F
        rts
smin:   lda #0
        ldx #$80
        rts

add16:  pha
        txa
        and #$80
        beq :+
        lda #$FF
:       sta sg                  ; sign extension (sg is free here)
        pla
        clc
        adc r32
        sta r32
        txa
        adc r32+1
        sta r32+1
        lda sg
        adc r32+2
        sta r32+2
        lda sg
        adc r32+3
        sta r32+3
        rts

tofix:  sta tfa
        stx tfa+1
        ora tfa+1
        bne :+
        jmp clr32
:       lda tfa
        jsr ilogabs
        clc
        adc t16
        pha
        txa
        adc t16+1
        tax
        pla
        jsr pow32
        lda tfa+1
        bpl :+
        jmp neg32
:       rts

fixabs: sta tfa
        stx tfa+1
        ora tfa+1
        beq fa0
        lda tfa
        jsr ilogabs
        clc
        adc t16
        pha
        txa
        adc t16+1
        tax
        pla
        jsr pow2f               ; carry = overflow
        lda #1                  ; Z = 0
        rts
fa0:    sta pw
        sta pw+1
        clc
        rts

pow2f:  cpx #0                  ; 0 <= e < 256: table
        bne :+
        tay
        lda p2lo,y
        sta pw
        lda p2hi,y
        sta pw+1
        clc
        rts
:       bpl :+
        lda #0                  ; e < 0: zero
        sta pw
        sta pw+1
        clc
        rts
:       sec                     ; e >= 256: >= 65536
        rts

addsat: php                     ; carry = operation
        pha
        ldy #0
        txa
        bpl :+
        dey
:       sty s24+2
        pla
        plp
        bcs as_sub
        clc
        adc pw
        sta s24
        txa
        adc pw+1
        sta s24+1
        lda s24+2
        adc #0
        sta s24+2
        jmp as_sat
as_sub: sec
        sbc pw
        sta s24
        txa
        sbc pw+1
        sta s24+1
        lda s24+2
        sbc #0
        sta s24+2
as_sat: lda s24+2
        beq as_pos
        cmp #$FF
        bne as_ov
        lda s24+1               ; negative: bit 7 must be set
        bpl as_min
        tax
        lda s24
        rts
as_pos: lda s24+1
        bmi as_max
        tax
        lda s24
        rts
as_ov:  bmi as_min
as_max: lda #$FF
        ldx #$7F
        rts
as_min: lda #0
        ldx #$80
        rts

signext:
        and #$80
        beq :+
        lda #$FF
:       rts

sext8:  ldx #0
        cmp #$80
        bcc :+
        dex
:       rts

;-----------------------------------------------------------------------------
