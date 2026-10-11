;- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
;
; The model: one autoregressive step of the transformer (intengine.IntModel.step).
;
;   void model_init(void);                  aligned tables from the image (call once)
;   void __fastcall__ model_step(unsigned);  A = token, X = position; output: _acc = logits z
;   unsigned char argmax(void);              token with the highest logit (not PAD, UNK, BOT)
;
; Step: embeddings (token + position), then for every layer
;   LayerNorm -> log8 -> q, k, v -> k and v into the KV cache -> 4 attention heads -> log8
;   -> wo -> residual -> LayerNorm -> log8 -> w1 -> GELU -> log8 -> w2 -> residual
; and at the end LayerNorm -> log8 -> 256 logits with the embedding matrix (weight tying).
;
; With _kv_only = 1 (the prompt tokens except the last) the step stops as soon as k and v of
; the last layer are in the cache: q, attention and FFN of the last layer and the logits
; would only produce the logits of this position, which are not used. The cache is the same
; as after a full step.
;
; _layer_hook is called with A = layer at the end of every layer (except the last one of a
; _kv_only step): the machine can use it as a sign of life. It points to an RTS by default.
;
; The pointers of each layer are in layer_desc (desc.inc, written by python/export.py),
; DESC_W words per layer in the order of the D_* constants; the final ones in final_desc.
;
        .export _model_init, _model_step, _argmax, _kv_only, _layer_hook, _img
        .import _matmul, _lin_post, _to_log8, _resid_add, _logits_fix, _layernorm
        .import _enc_static, _gelu, _att_head, _pow2_16
        .import _lin_bias, _lin_kw, _lin_j0, _lin_unit, _ln_g, _ln_b, _ln_kg, _logit_kw
        .import _enc_out, _enc_k, _enc_stride, _gelu_table
        .import _att_q, _att_kb, _att_vb, _att_n, _att_zs, _att_y
        .import _acc, _unit, _x, _lutlo, _luthi, _prodtab, _prodatab

        .include "config.inc"

ZERO    = 127
C       = 32                    ; d_model
HS      = 8                     ; head size
CTX_SHIFT = 6
        .assert CONTEXT = 1 << CTX_SHIFT, error, "the KV cache layout assumes a 64-token context"

; words of a layer descriptor
D_WQ = 0
D_WK = 3
D_WV = 6
D_WO = 9
D_W1 = 12
D_W2 = 15                       ; each matrix: W, kw, bias
D_LN1 = 18                      ; each LayerNorm: g, kg (value), b
D_LN2 = 21
D_KQ = 24                       ; static scales (int[4]) of q, k, v and the shift Zs
D_KK = 25
D_KV = 26
D_ZS = 27
D_KC = 28                       ; kcache[L] (K transposed), vcache[L]
D_VC = 29
DESC_W = 30
; final descriptor
F_LNF = 0                       ; g, kg, b
F_WTE = 3                       ; W, kw, bias
F_WPE = 6                       ; W, kw

        .zeropage
dp:     .res 2                  ; descriptor of the current layer
sp2:    .res 2                  ; work pointer
lw:     .res 2
li:     .res 1                  ; layer
hh:     .res 1                  ; head / counter
ee:     .res 2

        .segment "IMAGE"
_img:   .incbin "model.bin"     ; the model image, page aligned

        .data
_layer_hook: .word nohook

        .bss
st_tok: .res 1
st_pos: .res 1
_kv_only: .res 1
qb:     .res C                  ; q of the current position (log8 bytes)

        .segment "KVCACHE"
kcache: .res N_LAYER * C * CONTEXT  ; KV cache in log8: K transposed (row = dimension)
vcache: .res N_LAYER * CONTEXT * C  ;   V by position

        .rodata
        .include "desc.inc"     ; layer_desc, final_desc

        .code
; copy word n of the descriptor to dst (2 bytes)
.macro DCOPY n, dst
        ldy #2 * (n)
        lda (dp),y
        sta dst
        iny
        lda (dp),y
        sta dst+1
.endmacro

; 32-output linear layer: matrix at word n of the descriptor
.macro LIN32 n
        DCOPY n + 1, _lin_kw
        DCOPY n + 2, _lin_bias
        DCOPY n, lw
        jsr lin32
.endmacro

lin32:  lda #1
        sta _lin_unit
        lda lw
        ldx lw+1
        jsr _matmul
        lda #0
        sta _lin_j0
        sta _lin_j0+1
        jmp _lin_post

; LayerNorm: g, kg, b at words n, n+1, n+2 of the descriptor
.macro LNORM n
        DCOPY n, _ln_g
        DCOPY n + 1, _ln_kg
        DCOPY n + 2, _ln_b
        jsr _layernorm
.endmacro

; enc_static: output already in _enc_out, K = word nk, stride
.macro ENC nk, stride
        DCOPY nk, _enc_k
        lda #stride
        sta _enc_stride
        jsr _enc_static
.endmacro

nohook: rts

; x[i] += embedding row (packed nf4, 16 bytes) at sp2, scale k in A
embadd: sta ee                  ; k (signed) extended to 16 bits
        ldx #0
        cmp #$80
        bcc :+
        dex
:       stx ee+1
        clc                     ; ee = k + 16 R
        lda ee
        adc #<(16 * R_RES)
        sta ee
        lda ee+1
        adc #>(16 * R_RES)
        sta ee+1
        lda #0
        sta hh
ea1:    lda hh                  ; byte = row[i >> 1], nibble by i & 1
        lsr a
        tay
        lda (sp2),y
        tay
        lda hh
        and #1
        beq :+
        lda _luthi,y
        jmp :++
:       lda _lutlo,y
:       pha                     ; log8 byte of the weight
        and #$7F
        cmp #ZERO
        bne :+
        pla
        jmp eanext
:       sta lw                  ; e
        sec                     ; exponent = k - e + 16 R
        lda ee
        sbc lw
        pha
        lda ee+1
        sbc #0
        tax
        pla
        jsr _pow2_16            ; A/X = m (low word)
        sta lw
        stx lw+1
        pla
        bpl :+
        lda #0                  ; negative
        sec
        sbc lw
        sta lw
        lda #0
        sbc lw+1
        sta lw+1
:       lda hh
        asl a
        tay
        clc
        lda _x,y
        adc lw
        sta _x,y
        lda _x+1,y
        adc lw+1
        sta _x+1,y
eanext: inc hh
        lda hh
        cmp #C
        bne ea1
        rts

; ---- attention: 4 heads -------------------------------------------------------
attention:
        ldx st_pos
        inx
        stx _att_n
        lda #0
        sta hh
at1:    lda hh                  ; q of head h
        asl a
        asl a
        asl a
        tax
        ldy #0
:       lda qb,x
        sta _att_q,y
        inx
        iny
        cpy #HS
        bne :-
        lda hh                  ; kb = kcache[L] + 8 h * CONTEXT
        asl a
        asl a
        asl a
        sta lw
        lda #0
        sta lw+1
        ldx #0                  ; * CONTEXT (a power of 2)
:       asl lw
        rol lw+1
        inx
        cpx #CTX_SHIFT
        bne :-
        ldy #2 * D_KC
        lda (dp),y
        clc
        adc lw
        sta _att_kb
        iny
        lda (dp),y
        adc lw+1
        sta _att_kb+1
        ldy #2 * D_VC           ; vb = vcache[L] + 8 h
        lda hh
        asl a
        asl a
        asl a
        clc
        adc (dp),y
        sta _att_vb
        iny
        lda (dp),y
        adc #0
        sta _att_vb+1
        ldy #2 * D_ZS           ; zs = Zs[h]
        lda (dp),y
        sta sp2
        iny
        lda (dp),y
        sta sp2+1
        lda hh
        asl a
        tay
        lda (sp2),y
        sta _att_zs
        jsr _att_head
        ldy #2 * D_KV           ; unit = Kv[h] - 16 F_ATT
        lda (dp),y
        sta sp2
        iny
        lda (dp),y
        sta sp2+1
        lda hh
        asl a
        tay
        sec
        lda (sp2),y
        sbc #<(16 * F_ATT)
        sta ee
        iny
        lda (sp2),y
        sbc #>(16 * F_ATT)
        sta ee+1
        lda hh                  ; acc[8h + d] = y[d], unit
        asl a
        asl a
        asl a
        asl a
        tax                     ; X = 16 h (byte offset)
        ldy #0
:       lda _att_y,y
        sta _acc,x
        lda ee
        sta _unit,x
        inx
        iny
        lda _att_y,y
        sta _acc,x
        lda ee+1
        sta _unit,x
        inx
        iny
        cpy #2 * HS
        bne :-
        inc hh
        lda hh
        cmp #4
        beq :+
        jmp at1
:       rts

; ---- initialization ------------------------------------------------------------
_model_init:
        ldx #0                  ; PROD tables on aligned pages, nibble -> log8
:       lda _img + OFF_T_PROD,x
        sta _prodtab,x
        lda _img + OFF_T_PRODA,x
        sta _prodatab,x
        txa
        and #15
        tay
        lda _img + OFF_NF4LOG,y
        sta _lutlo,x
        txa
        lsr a
        lsr a
        lsr a
        lsr a
        tay
        lda _img + OFF_NF4LOG,y
        sta _luthi,x
        inx
        bne :-
        lda #<(_img + OFF_T_GELU)
        sta _gelu_table
        lda #>(_img + OFF_T_GELU)
        sta _gelu_table+1
        rts

; ---- argmax: the highest logit; on ties the first one (as np.argmax) ----------------
_argmax:
        lda #<_acc
        sta sp2
        lda #>_acc
        sta sp2+1
        lda #$00
        sta ee
        lda #$80
        sta ee+1                ; maximum = -32768
        lda #0
        sta lw                  ; best
        sta hh                  ; j
pk1:    lda hh
        cmp #TOK_PAD
        beq pknext
        cmp #TOK_UNK
        beq pknext
        cmp #TOK_BOT
        beq pknext
        ldy #0
        lda (sp2),y             ; z > maximum (signed)?
        cmp ee
        iny
        lda (sp2),y
        sbc ee+1
        bvc :+
        eor #$80
:       bmi pknext
        beq pkeq                ; it may be equal: compare the low bytes
pkset:  ldy #0
        lda (sp2),y
        sta ee
        iny
        lda (sp2),y
        sta ee+1
        lda hh
        sta lw
        jmp pknext
pkeq:   ldy #0                  ; high difference 0: z - max in 0..255, z = max only if
        lda (sp2),y             ; the low bytes are equal too
        cmp ee
        bne pkset
pknext: clc
        lda sp2
        adc #2
        sta sp2
        bcc :+
        inc sp2+1
:       inc hh
        bne pk1
        lda lw
        ldx #0
        rts

; Z = 1 if _kv_only and this is the last layer
kvlast: lda _kv_only
        beq :+
        lda li
        cmp #N_LAYER - 1
        rts
:       lda #1
        rts

; ---- the step --------------------------------------------------------------------
_model_step:
        sta st_tok
        stx st_pos
        ldy #2 * C - 1          ; x = 0
        lda #0
:       sta _x,y
        dey
        bpl :-
        lda final_desc + 2 * F_WTE  ; x += wte[tok] (16 bytes per token)
        sta sp2
        lda final_desc + 2 * F_WTE + 1
        sta sp2+1
        lda st_tok              ; sp2 += tok * 16
        sta lw
        lda #0
        sta lw+1
        ldx #4
:       asl lw
        rol lw+1
        dex
        bne :-
        clc
        lda sp2
        adc lw
        sta sp2
        lda sp2+1
        adc lw+1
        sta sp2+1
        lda final_desc + 2 * (F_WTE + 1)    ; kw[tok]
        sta lw
        lda final_desc + 2 * (F_WTE + 1) + 1
        sta lw+1
        ldy st_tok
        lda (lw),y
        jsr embadd
        lda final_desc + 2 * F_WPE  ; x += wpe[pos]
        sta sp2
        lda final_desc + 2 * F_WPE + 1
        sta sp2+1
        lda st_pos
        sta lw
        lda #0
        sta lw+1
        ldx #4
:       asl lw
        rol lw+1
        dex
        bne :-
        clc
        lda sp2
        adc lw
        sta sp2
        lda sp2+1
        adc lw+1
        sta sp2+1
        lda final_desc + 2 * (F_WPE + 1)
        sta lw
        lda final_desc + 2 * (F_WPE + 1) + 1
        sta lw+1
        ldy st_pos
        lda (lw),y
        jsr embadd

        lda #<layer_desc
        sta dp
        lda #>layer_desc
        sta dp+1
        lda #0
        sta li
layer:  LNORM D_LN1
        jsr _to_log8
        jsr kvlast              ; prompt, last layer: no q
        beq lwk
        LIN32 D_WQ
        lda #<qb
        sta _enc_out
        lda #>qb
        sta _enc_out+1
        ENC D_KQ, 1
lwk:    LIN32 D_WK
        ldy #2 * D_KC           ; out = kcache[L] + pos (the column of the position)
        lda (dp),y
        clc
        adc st_pos
        sta _enc_out
        iny
        lda (dp),y
        adc #0
        sta _enc_out+1
        ENC D_KK, CONTEXT
        LIN32 D_WV
        lda st_pos              ; out = vcache[L] + 32 pos
        sta lw
        lda #0
        sta lw+1
        ldx #5
:       asl lw
        rol lw+1
        dex
        bne :-
        ldy #2 * D_VC
        lda (dp),y
        clc
        adc lw
        sta _enc_out
        iny
        lda (dp),y
        adc lw+1
        sta _enc_out+1
        ENC D_KV, 1
        jsr kvlast              ; prompt, last layer: k and v are in the cache, done
        bne :+
        rts
:       jsr attention
        jsr _to_log8
        LIN32 D_WO
        jsr _resid_add
        LNORM D_LN2
        jsr _to_log8
        LIN32 D_W1
        jsr _gelu
        jsr _to_log8
        LIN32 D_W2
        jsr _resid_add
        lda li                  ; end of the layer: hook for the machine (A = layer)
        jsr callhook
        clc                     ; next layer
        lda dp
        adc #2 * DESC_W
        sta dp
        bcc :+
        inc dp+1
:       inc li
        lda li
        cmp #N_LAYER
        beq :+
        jmp layer
:
        lda #<final_desc        ; final LayerNorm and logits
        sta dp
        lda #>final_desc
        sta dp+1
        LNORM F_LNF
        jsr _to_log8
        DCOPY F_WTE + 1, _lin_kw
        DCOPY F_WTE + 2, _lin_bias
        DCOPY F_WTE, lw
        lda #0
        sta _lin_unit
        sta _lin_j0
        sta _lin_j0+1
        sta hh
lg1:    lda lw
        ldx lw+1
        jsr _matmul
        jsr _lin_post
        clc                     ; next block: +512 bytes, +32 outputs
        lda lw+1
        adc #2
        sta lw+1
        clc
        lda _lin_j0
        adc #32
        sta _lin_j0
        bcc :+
        inc _lin_j0+1
:       inc hh
        lda hh
        cmp #VOCAB / 32
        bne lg1
        DCOPY F_WTE + 1, _logit_kw
        jmp _logits_fix

callhook:
        jmp (_layer_hook)

;-----------------------------------------------------------------------------
