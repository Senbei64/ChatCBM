;- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
;
; End of a linear layer: kernel sums plus bias.
;
;   lin_post()   acc[j0 + j] = sat16(kernel_j +- pow2_16(L'_j - kx)), j = 0..31
;                unit[j] = kx + kw[j] - 16 F_LIN          (only if _lin_unit = 1)
;
; The kernel sums come from matmul (_acc_lo/_acc_hi). Parameters:
;   _lin_bias   pointer to the biases prepared by python/export.py (prep_bias):
;               2 (L' + 1024) + sign, with L' = ilog16|q| + 16 (F - s); $7FFF = no bias
;   _lin_kw     pointer to the weight scales k (int8, one per output)
;   _lin_j0     first output: 0 for a 32-output layer, 32 b for block b of the logits
;   _lin_unit   1: write _unit too (32-output layers)
; _kx is the scale of the input vector (log8.s).
;
        .export _lin_post, _lin_bias, _lin_kw, _lin_j0, _lin_unit
        .import _acc, _unit, _kx
        .import pow2f, addsat, sext8
        .importzp _acc_lo, _acc_hi, ix, sg, t16, pa, pb, pk

        .include "config.inc"

        .bss
_lin_bias: .res 2
_lin_kw:   .res 2
_lin_j0:   .res 2
_lin_unit: .res 1

        .code
_lin_post:
        lda _lin_j0             ; pa = &_acc[j0], pb = &bias[j0], pk = &kw[j0]
        asl a
        tax
        lda _lin_j0+1
        rol a
        tay
        txa
        clc
        adc #<_acc
        sta pa
        tya
        adc #>_acc
        sta pa+1
        txa
        clc
        adc _lin_bias
        sta pb
        tya
        adc _lin_bias+1
        sta pb+1
        clc
        lda _lin_j0
        adc _lin_kw
        sta pk
        lda _lin_j0+1
        adc _lin_kw+1
        sta pk+1
        lda #0
        sta ix
lp1:    lda ix
        asl a
        tay
        lda (pb),y              ; prepared bias
        sta t16
        iny
        lda (pb),y
        sta t16+1
        cmp #$7F
        bne lpb
        lda t16
        cmp #$FF
        bne lpb
        ldx ix                  ; no bias: the kernel value (|a| <= 8160)
        lda _acc_lo,x
        pha
        lda _acc_hi,x
        tax
        pla
        jmp lpst
lpb:    lsr t16+1               ; e = (t16 >> 1) - 1024 - kx; carry = bias sign
        ror t16
        lda #0
        rol a
        sta sg
        sec
        lda t16
        sbc _kx
        sta t16
        lda t16+1
        sbc _kx+1
        sta t16+1
        sec
        lda t16
        sbc #<1024
        pha
        lda t16+1
        sbc #>1024
        tax
        pla
        jsr pow2f
        bcc lpfit
        lda sg                  ; |bias| >= 65536: saturate by its sign
        bne :+
        lda #$FF
        ldx #$7F
        jmp lpst
:       lda #0
        ldx #$80
        jmp lpst
lpfit:  ldx ix
        lda _acc_lo,x
        pha
        lda _acc_hi,x
        tax
        lda sg
        lsr a                   ; carry = sign
        pla
        jsr addsat
lpst:   pha
        lda ix
        asl a
        tay
        pla
        sta (pa),y
        txa
        iny
        sta (pa),y
        lda _lin_unit           ; unit[j] = kx + kw[j] - 16F
        beq lpnext
        ldy ix
        lda (pk),y
        jsr sext8
        clc
        adc _kx
        sta t16
        txa
        adc _kx+1
        sta t16+1
        lda ix
        asl a
        tay
        sec
        lda t16
        sbc #<(16 * F_LIN)
        sta _unit,y
        lda t16+1
        sbc #>(16 * F_LIN)
        sta _unit+1,y
lpnext: inc ix
        lda ix
        cmp #32
        beq :+
        jmp lp1
:       rts

;-----------------------------------------------------------------------------
