;- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
;
; Logits as integers (intengine: z = to_fixed(to_log(acc, u + Zl), 0)):
;
;   logits_fix()   acc[j] = to_fixed(acc[j], kx + kw[j] - 16 F_LIN + Z_L, 0), j = 0..255
;
; z = logit * 16 log2(e): the unit of the sampler (sampler.s) and of the argmax (model.s).
; _logit_kw points to the scales of the output weights (wte, one int8 per token).
;
        .export _logits_fix, _logit_kw
        .import _acc, _kx
        .import fixabs, sext8
        .importzp ix, sg, t16, pa, pk, pw

        .include "config.inc"

        .bss
_logit_kw: .res 2

        .code
_logits_fix:
        lda #<_acc
        sta pa
        lda #>_acc
        sta pa+1
        lda _logit_kw
        sta pk
        lda _logit_kw+1
        sta pk+1
        lda #0
        sta ix
lf1:    ldy ix                  ; t16 = kx + kw[j] - 16F + Zl
        lda (pk),y
        jsr sext8
        clc
        adc _kx
        sta t16
        txa
        adc _kx+1
        sta t16+1
        sec
        lda t16
        sbc #<(16 * F_LIN - Z_L)
        sta t16
        lda t16+1
        sbc #>(16 * F_LIN - Z_L)
        sta t16+1
        ldy #0
        lda (pa),y
        pha
        iny
        lda (pa),y
        tax
        pla
        jsr fixabs              ; z = sign * pw (the logits fit 16 bits)
        lda sg
        bpl :+
        sec
        lda #0
        sbc pw
        sta pw
        lda #0
        sbc pw+1
        sta pw+1
:       ldy #0
        lda pw
        sta (pa),y
        iny
        lda pw+1
        sta (pa),y
        clc
        lda pa
        adc #2
        sta pa
        bcc :+
        inc pa+1
:       inc ix
        bne lf1
        rts

;-----------------------------------------------------------------------------
