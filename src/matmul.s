;- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
;
; Matrix kernel: 32 inputs times a 32 x 32 block of packed nf4 weights.
;
;   void __fastcall__ matmul(const unsigned char *w);     w page aligned
;
;   acc[j] = sum_i +-PROD[xe_i + we_ij]       j = 0..31, in _acc_lo/_acc_hi (zero page)
;
; Multiplying two log8 values means adding their exponents; PROD[e] = 2^(F - e/16) turns the
; sum back into a linear product, so a dot product is lookups and additions only. The trick
; (from the author's 2018 notes, "exp with bias"): for a whole input row the table pointer tp
; is moved by the input exponent, tp = PROD + xe_i, so product and conversion are a single
; ADC/SBC (tp),Y with Y = weight exponent.
;
; - The weight sign is bit 7 of its log8 byte: LDY sets the N flag, BMI reads it for free.
;   Negative weights use tpm = PROD + xe_i - 128, so Y >= 128 points to the same entry.
; - The input sign is decided once per row: two copies of the loop, ADC and SBC swapped.
; - No tests for zero: zero is exponent 127 and PROD[e >= 127] = 0. A zero input skips its row.
; - Packed nf4: a 512-byte block, column j = 16 bytes at w + 16 j, byte k holds row 2k in the
;   low nibble and row 2k+1 in the high one. Even rows go through lutlo, odd rows through
;   luthi: 256-entry tables that return the log8 byte of the weight. With w page aligned the
;   low bytes of the operands are fixed, and only the 128 high bytes are patched per call.
;
; About 30 cycles per product (24 with one byte per weight, at twice the memory).
; Before use: prodtab = PROD, lutlo[b] = log8(b & 15), luthi[b] = log8(b >> 4) (model.s).
;
        .export _matmul, _prodtab, _lutlo, _luthi
        .exportzp _acc_lo, _acc_hi
        .import _xs, _xe

ZERO    = 127

        .segment "ALIGNED"
_prodtab: .res 256              ; PROD, page aligned
_lutlo:   .res 256              ; low nibble -> log8 byte
_luthi:   .res 256              ; high nibble -> log8 byte

        .zeropage
tp:     .res 2                  ; PROD + xe
tpm:    .res 2                  ; PROD + xe - 128
kk:     .res 1                  ; byte (row pair) index
_acc_lo: .res 32
_acc_hi: .res 32

        .code
; one product: column J, nibble table LUT, ADDP = 1 for a positive input
.macro MAC J, LUT, ADDP, BODY
        .local nw, done
        .ident(.sprintf("%s%d", BODY, J)):
        ldy $FF00 + ((16 * J) & $FF),x
        lda LUT,y
        tay
        bmi nw
        lda _acc_lo+J
    .if ADDP
        clc
        adc (tp),y
    .else
        sec
        sbc (tp),y
    .endif
        sta _acc_lo+J
    .if ADDP
        bcc done
        inc _acc_hi+J
        bcs done
    .else
        bcs done
        dec _acc_hi+J
        bcc done
    .endif
nw:     lda _acc_lo+J
    .if ADDP
        sec
        sbc (tpm),y
        sta _acc_lo+J
        bcs done
        dec _acc_hi+J
    .else
        clc
        adc (tpm),y
        sta _acc_lo+J
        bcc done
        inc _acc_hi+J
    .endif
done:
.endmacro

; input row in Y: skip it if zero, otherwise set the pointers and jump to the loop for its sign
.macro ROWSETUP POSLAB, NEGLAB, SKIPLAB
        lda _xe,y
        cmp #ZERO
        beq SKIPLAB
        sta tp
        ora #$80
        sta tpm
        ldx kk
        lda _xs,y
        bmi :+
        jmp POSLAB
:       jmp NEGLAB
.endmacro

_matmul:
        txa                     ; high byte of the block: columns 0..15, then +1 for 16..31
        .repeat 16, J
        sta .ident(.sprintf("op%d", J)) + 2
        sta .ident(.sprintf("on%d", J)) + 2
        sta .ident(.sprintf("ep%d", J)) + 2
        sta .ident(.sprintf("en%d", J)) + 2
        .endrepeat
        clc
        adc #1
        .repeat 16, J
        sta .ident(.sprintf("op%d", J + 16)) + 2
        sta .ident(.sprintf("on%d", J + 16)) + 2
        sta .ident(.sprintf("ep%d", J + 16)) + 2
        sta .ident(.sprintf("en%d", J + 16)) + 2
        .endrepeat

        lda #0                  ; clear the accumulators
        ldx #31
:       sta _acc_lo,x
        sta _acc_hi,x
        dex
        bpl :-
        lda #>_prodtab
        sta tp+1
        sta tpm+1
        dec tpm+1
        ldx #15
rowk:   stx kk
        txa                     ; odd row 2k+1
        asl a
        ora #1
        tay
        ROWSETUP oddpos, oddneg, even
even:   lda kk                  ; even row 2k
        asl a
        tay
        ROWSETUP evenpos, evenneg, next
next:   ldx kk
        dex
        bpl rowk
        rts

oddpos: .repeat 32, J
        MAC J, _luthi, 1, "op"
        .endrepeat
        jmp even
oddneg: .repeat 32, J
        MAC J, _luthi, 0, "on"
        .endrepeat
        jmp even
evenpos:
        .repeat 32, J
        MAC J, _lutlo, 1, "ep"
        .endrepeat
        jmp next
evenneg:
        .repeat 32, J
        MAC J, _lutlo, 0, "en"
        .endrepeat
        jmp next

;-----------------------------------------------------------------------------
