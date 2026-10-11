;- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
;
; Choosing the next token from the integer logits z in _acc (intengine.IntSampler):
;
;   unsigned char argmax(void);   the highest logit; on ties the first one (as np.argmax)
;   unsigned char sample(void);   temperature and top-p with integers only
;
; PAD, UNK and BOT are never chosen.
;
; sample(): the weight of a token is p = samp[min(zmax - z, SAMP_L - 1)], a table of
; 2^(-e / (16 T)) (sampler.inc; banned tokens get the last entry, 0). Python sorts the tokens
; by decreasing weight, ties by token: here the exponents with the same weight form
; contiguous groups (the table decreases), so the groups are scanned in order and the tokens
; of each group by index. The scan stops at the first sum >= threshold (top-p), then
; r = xorshift32 & mask until r < sum, and the first token with cumulative sum > r wins.
;
        .export _argmax, _sample, _rng_state
        .import _acc

        .include "config.inc"

        .rodata
        .include "sampler.inc"  ; SAMP_L, TOP_P, samp_lo, samp_hi
        .assert SAMP_L <= 256, error, "temperature table too long"

        .zeropage
sp:     .res 2                  ; pointer into the logits
zm:     .res 2                  ; maximum logit
tot:    .res 3                  ; sum of the weights
thr:    .res 4                  ; tot * TOP_P: the threshold is bytes 1..3
cum:    .res 3                  ; cumulative sum of the chosen tokens
msk:    .res 3
rr:     .res 3
vv:     .res 2                  ; weight of the current group
e0:     .res 1                  ; group of exponents [e0, e1)
e1:     .res 1
cnt:    .res 1
tt:     .res 4
best:   .res 1
jj:     .res 1

        .data
_rng_state: .byte 1, 0, 0, 0    ; xorshift32, never zero

        .bss
ebuf:   .res 256                ; exponent of the weight of each token
ltok:   .res 256                ; tokens kept by top-p, in order
lc0:    .res 256                ; cumulative sums (24 bits)
lc1:    .res 256
lc2:    .res 256

        .code
; ---- argmax ------------------------------------------------------------------------
_argmax:
        lda #<_acc
        sta sp
        lda #>_acc
        sta sp+1
        lda #$00
        sta zm
        lda #$80
        sta zm+1                ; maximum = -32768
        lda #0
        sta best
        sta jj
pk1:    lda jj
        cmp #TOK_PAD
        beq pknext
        cmp #TOK_UNK
        beq pknext
        cmp #TOK_BOT
        beq pknext
        ldy #0
        lda (sp),y              ; z > maximum (signed)?
        cmp zm
        iny
        lda (sp),y
        sbc zm+1
        bvc :+
        eor #$80
:       bmi pknext
        beq pkeq                ; it may be equal: compare the low bytes
pkset:  ldy #0
        lda (sp),y
        sta zm
        iny
        lda (sp),y
        sta zm+1
        lda jj
        sta best
        jmp pknext
pkeq:   ldy #0                  ; high difference 0: z - max in 0..255, z = max only if
        lda (sp),y              ; the low bytes are equal too
        cmp zm
        bne pkset
pknext: clc
        lda sp
        adc #2
        sta sp
        bcc :+
        inc sp+1
:       inc jj
        bne pk1
        lda best
        ldx #0
        rts

; ---- sample ------------------------------------------------------------------------
_sample:
        jsr _argmax             ; zm = maximum of the allowed tokens
        lda #<_acc
        sta sp
        lda #>_acc
        sta sp+1
        lda #0
        sta tot
        sta tot+1
        sta tot+2
        tax                     ; X = token
seloop: cpx #TOK_PAD
        beq seban
        cpx #TOK_UNK
        beq seban
        cpx #TOK_BOT
        beq seban
        ldy #0                  ; d = zmax - z (>= 0, at most 16 bits unsigned)
        sec
        lda zm
        sbc (sp),y
        sta tt
        iny
        lda zm+1
        sbc (sp),y
        bne seban
        lda tt
        cmp #SAMP_L - 1
        bcc seset
seban:  lda #SAMP_L - 1
seset:  sta ebuf,x
        tay
        clc
        lda tot
        adc samp_lo,y
        sta tot
        lda tot+1
        adc samp_hi,y
        sta tot+1
        bcc :+
        inc tot+2
:       clc
        lda sp
        adc #2
        sta sp
        bcc :+
        inc sp+1
:       inx
        bne seloop

        lda #0                  ; thr = tot * TOP_P (32 bits), threshold = thr >> 8
        sta thr
        sta thr+1
        sta thr+2
        sta thr+3
        lda #TOP_P
        sta tt
        ldx #8
smul:   asl thr
        rol thr+1
        rol thr+2
        rol thr+3
        asl tt
        bcc :+
        clc
        lda thr
        adc tot
        sta thr
        lda thr+1
        adc tot+1
        sta thr+1
        lda thr+2
        adc tot+2
        sta thr+2
        bcc :+
        inc thr+3
:       dex
        bne smul

        lda #0
        sta cum
        sta cum+1
        sta cum+2
        sta cnt
        sta e0
sgroup: ldx e0
        lda samp_lo,x
        sta vv
        lda samp_hi,x
        sta vv+1
        ora vv
        beq scut                ; zero weight: never reached before the threshold
:       inx                     ; e1 = first exponent with a different weight
        cpx #SAMP_L
        bcs :+
        lda samp_lo,x
        cmp vv
        bne :+
        lda samp_hi,x
        cmp vv+1
        beq :-
:       stx e1
        ldy #0
stok:   lda ebuf,y
        cmp e0
        bcc snext
        cmp e1
        bcs snext
        clc                     ; cum += weight; ltok/lc[cnt] = token, cum
        lda cum
        adc vv
        sta cum
        lda cum+1
        adc vv+1
        sta cum+1
        bcc :+
        inc cum+2
:       ldx cnt
        tya
        sta ltok,x
        lda cum
        sta lc0,x
        lda cum+1
        sta lc1,x
        lda cum+2
        sta lc2,x
        inc cnt
        lda cum                 ; cum >= threshold (thr bytes 1..3)?
        cmp thr+1
        lda cum+1
        sbc thr+2
        lda cum+2
        sbc thr+3
        bcs scut
snext:  iny
        bne stok
        lda e1
        sta e0
        cmp #SAMP_L
        bcc sgroup
scut:   lda cum+2               ; mask = 2^bitlen(cum) - 1
        beq :+
        jsr smear
        sta msk+2
        lda #$FF
        sta msk+1
        sta msk
        bne srand
:       sta msk+2
        lda cum+1
        beq :+
        jsr smear
        sta msk+1
        lda #$FF
        sta msk
        bne srand
:       sta msk+1
        lda cum
        jsr smear
        sta msk
srand:  jsr xorshift
        lda _rng_state
        and msk
        sta rr
        lda _rng_state+1
        and msk+1
        sta rr+1
        lda _rng_state+2
        and msk+2
        sta rr+2
        lda rr                  ; r < cum ?
        cmp cum
        lda rr+1
        sbc cum+1
        lda rr+2
        sbc cum+2
        bcs srand
        ldx #0
sfind:  lda rr                  ; first i with lc[i] > r, that is r - lc[i] < 0
        cmp lc0,x
        lda rr+1
        sbc lc1,x
        lda rr+2
        sbc lc2,x
        bcc :+
        inx
        bne sfind
:       lda ltok,x
        ldx #0
        rts

; A -> A with every bit below the highest one set
smear:  sta tt
        lsr a
        ora tt
        sta tt
        lsr a
        lsr a
        ora tt
        sta tt
        lsr a
        lsr a
        lsr a
        lsr a
        ora tt
        rts

; xorshift32: x ^= x << 13; x ^= x >> 17; x ^= x << 5 (little endian state)
xorshift:
        lda #0                  ; tt = x << 8, then << 5
        sta tt
        lda _rng_state
        sta tt+1
        lda _rng_state+1
        sta tt+2
        lda _rng_state+2
        sta tt+3
        ldx #5
:       asl tt+1
        rol tt+2
        rol tt+3
        dex
        bne :-
        lda _rng_state+1
        eor tt+1
        sta _rng_state+1
        lda _rng_state+2
        eor tt+2
        sta _rng_state+2
        lda _rng_state+3
        eor tt+3
        sta _rng_state+3
        lda _rng_state+3        ; x >> 17 = (x >> 16) >> 1
        lsr a
        sta tt+1
        lda _rng_state+2
        ror a
        eor _rng_state
        sta _rng_state
        lda tt+1
        eor _rng_state+1
        sta _rng_state+1
        ldx #3                  ; tt = x << 5
:       lda _rng_state,x
        sta tt,x
        dex
        bpl :-
        ldx #5
:       asl tt
        rol tt+1
        rol tt+2
        rol tt+3
        dex
        bne :-
        ldx #3
:       lda _rng_state,x
        eor tt,x
        sta _rng_state,x
        dex
        bpl :-
        rts

;-----------------------------------------------------------------------------
