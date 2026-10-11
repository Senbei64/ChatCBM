;- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
;
; BPE tokenizer of the question (python/chatcbm/tokenizer.py and generate.chat_prompt):
;
;   unsigned char __fastcall__ tokenize(const char *text);
;       ASCII text ending with 0 -> _prompt_ids[_prompt_len] = <user> encode(" " + strip(text)) <bot>
;       returns 0, or 1 if the question does not fit (normalized text or prompt too long)
;
; On the normalized text (lower case, whitespace runs -> one space, " " in front):
;   chunks  " ?letters+'?" | " ?digits+" | " ?punctuation+" | " "    ("_" is skipped)
;   chunk   characters -> ids (chartab, unknown = <unk>), then the BPE merges in rank order:
;           merge r joins the pairs (merge_a[r], merge_b[r]) into token TOK_BASE + r.
;           Applying each merge to the whole chunk in rank order gives the same result as
;           merging the lowest-ranked pair each time (checked by python/export.py): a pair
;           made by merge r can only be merged by a later merge.
;
; The vocabulary strings (voc_lo, voc_hi, voc_str) are also exported, for the output.
;
        .export _tokenize, _prompt_ids, _prompt_len, voc_lo, voc_hi

        .include "config.inc"

        .rodata
        .include "tokenizer.inc"        ; TOK_BASE, N_MERGES, chartab, merges, vocabulary

NMAX    = 120                   ; characters of the normalized text
APOS    = $27
CL_SPC  = 0
CL_LET  = 1
CL_DIG  = 2
CL_PUN  = 3
CL_UND  = 4
        .assert TOK_BASE + N_MERGES <= 256, error, "tokens must fit a byte"

        .zeropage
cp:     .res 2                  ; text
ci:     .res 1                  ; start of the chunk
cj:     .res 1                  ; end of the chunk
cn:     .res 1                  ; length of the normalized text
ck:     .res 1                  ; tokens of the chunk
cr:     .res 1                  ; merge rank
ca:     .res 1
cb:     .res 1
cy:     .res 1
pend:   .res 1                  ; pending space (normalization)

        .bss
nbuf:   .res NMAX               ; normalized text
cids:   .res NMAX               ; tokens of the chunk
_prompt_ids: .res CONTEXT
_prompt_len: .res 1

        .code
; A: character -> C = 1 if it is whitespace for Python (\t \n \v \f \r, \x1c..\x1f, ' ')
isws:   cmp #' ' + 1
        bcs wsno
        cmp #28
        bcs wsyes
        cmp #9
        bcc wsno
        cmp #14
        bcs wsno
wsyes:  sec
        rts
wsno:   clc
        rts

; A: normalized character -> A = class (X and Y unchanged)
cclass: cmp #' '
        beq ccspc
        cmp #'_'
        beq ccund
        cmp #'0'
        bcc ccpun
        cmp #'9' + 1
        bcc ccdig
        cmp #'a'
        bcc ccpun
        cmp #'z' + 1
        bcc cclet
ccpun:  lda #CL_PUN
        rts
ccspc:  lda #CL_SPC
        rts
ccund:  lda #CL_UND
        rts
ccdig:  lda #CL_DIG
        rts
cclet:  lda #CL_LET
        rts

toolong:
        lda #1
        ldx #0
        rts

_tokenize:
        sta cp
        stx cp+1
        lda #' '                ; " " + strip(text), lower case, single spaces
        sta nbuf
        ldx #1
        ldy #0
nskip:  lda (cp),y
        beq ndone
        jsr isws
        bcc ngo
        iny
        bne nskip
        beq toolong
ngo:    lda #0
        sta pend
nloop:  lda (cp),y
        beq ndone
        jsr isws
        bcs nws
        pha
        lda pend
        beq :+
        lda #0
        sta pend
        lda #' '
        cpx #NMAX
        bcs nlong
        sta nbuf,x
        inx
:       pla
        cmp #'A'
        bcc :+
        cmp #'Z' + 1
        bcs :+
        ora #$20
:       cpx #NMAX
        bcs toolong
        sta nbuf,x
        inx
        iny
        bne nloop
        beq toolong
nlong:  pla
        jmp toolong
nws:    lda #1
        sta pend
        iny
        bne nloop
        beq toolong
ndone:  stx cn

        lda #TOK_USER
        sta _prompt_ids
        lda #1
        sta _prompt_len
        ldx #0
chloop: cpx cn
        bcs chend
        stx ci
        lda nbuf,x
        cmp #' '
        bne :+
        inx                     ; optional leading space
:       cpx cn
        bcs chnone
        lda nbuf,x
        jsr cclass
        cmp #CL_LET
        beq chlet
        cmp #CL_DIG
        beq chdig
        cmp #CL_PUN
        beq chpun
chnone: ldx ci                  ; no letter, digit or punctuation after the space
        lda nbuf,x
        cmp #' '
        bne chskip
        inx                     ; "\s+": the space alone
        jmp chemit
chskip: inx                     ; "_": no alternative matches, findall skips it
        jmp chloop
chlet:  inx
        cpx cn
        bcs chemit
        lda nbuf,x
        jsr cclass
        cmp #CL_LET
        beq chlet
        lda nbuf,x              ; optional trailing apostrophe
        cmp #APOS
        bne chemit
        inx
        jmp chemit
chdig:  inx
        cpx cn
        bcs chemit
        lda nbuf,x
        jsr cclass
        cmp #CL_DIG
        beq chdig
        bne chemit
chpun:  inx
        cpx cn
        bcs chemit
        lda nbuf,x
        jsr cclass
        cmp #CL_PUN
        beq chpun
chemit: stx cj
        jsr encode
        bcs chlong
        ldx cj
        jmp chloop
chlong: jmp toolong
chend:  ldx _prompt_len
        cpx #CONTEXT - 1        ; at least one position left to generate
        bcs chlong
        lda #TOK_BOT
        sta _prompt_ids,x
        inx
        stx _prompt_len
        lda #0
        tax
        rts

; chunk nbuf[ci..cj) -> tokens appended to _prompt_ids; C = 1 if there is no room
encode: ldx ci
        ldy #0
ecopy:  lda nbuf,x
        stx cy
        tax
        lda chartab,x
        sta cids,y
        ldx cy
        inx
        iny
        cpx cj
        bcc ecopy
        sty ck
        cpy #2
        bcc eappend
        lda #0
        sta cr
emerge: ldx cr
        lda merge_a,x
        sta ca
        lda merge_b,x
        sta cb
        ldy #0
escan:  iny                     ; pair (cids[y-1], cids[y])
        cpy ck
        bcs enext
        lda cids-1,y
        cmp ca
        bne escan
        lda cids,y
        cmp cb
        bne escan
        lda cr                  ; merge: cids[y-1] = TOK_BASE + r, drop cids[y]
        clc
        adc #TOK_BASE
        sta cids-1,y
        sty cy
eshift: iny
        cpy ck
        bcs :+
        lda cids,y
        sta cids-1,y
        jmp eshift
:       dec ck
        lda ck
        cmp #2
        bcc eappend             ; a single token: done
        ldy cy                  ; go on with the pair (cids[y], cids[y+1])
        jmp escan
enext:  inc cr
        lda cr
        cmp #N_MERGES
        bcc emerge
eappend:
        lda _prompt_len
        clc
        adc ck
        bcs efull
        cmp #CONTEXT
        bcs efull
        ldx _prompt_len
        ldy #0
:       lda cids,y
        sta _prompt_ids,x
        inx
        iny
        cpy ck
        bcc :-
        stx _prompt_len
        clc
        rts
efull:  sec
        rts

;-----------------------------------------------------------------------------
