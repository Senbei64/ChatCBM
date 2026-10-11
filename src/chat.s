;- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
;
; One chat turn, as python/chat.py:
;
;   void chat_init(void);                                    k and v of <user> at position 0
;   unsigned char __fastcall__ chat_turn(const char *text);  0 = ok, 1 = question too long
;
; The prompt is <user> question <bot> (tokenizer.s). Every turn starts from position 0, as
; chat.py keeps no history; position 0 is always <user>, so chat_init puts its k and v in the
; KV cache once and each turn starts from position 1. The prompt tokens except the last run
; the reduced step (_kv_only, model.s). Then tokens are generated until <eos>, <user>, a full
; context or MAX_NEW tokens: greedy if _temp_on = 0, sampled otherwise (sampler.s).
; The generated tokens are also left in _out_ids[_out_len].
;
; Machine dependent, provided by the main program:
;   _io_putc    A = character of a token (ASCII)
;   _io_flush   end of a token
;   _io_tick    after every model step (a sign of life)
; The leading spaces of the first token are not printed, as in chat.py.
;
        .export _chat_init, _chat_turn, _temp_on, _out_ids, _out_len
        .import _tokenize, _prompt_ids, _prompt_len, voc_lo, voc_hi
        .import _model_step, _kv_only, _argmax, _sample
        .import _io_putc, _io_flush, _io_tick

        .include "config.inc"

        .zeropage
cp:     .res 2                  ; string to print
cy:     .res 1
tpos:   .res 1                  ; position in the context
first:  .res 1                  ; first token of the answer (no leading spaces)

        .data
_temp_on: .byte 1

        .bss
_out_ids: .res MAX_NEW
_out_len: .res 1

        .code
_chat_init:
        lda #1
        sta _kv_only
        lda #TOK_USER
        ldx #0
        jsr _model_step
        lda #0
        sta _kv_only
        rts

_chat_turn:
        jsr _tokenize
        cmp #0
        beq :+
        rts
:       sta _out_len
        lda #1
        sta tpos                ; position 0 (<user>) is already in the cache
        sta first
tprompt:
        ldx tpos
        inx
        cpx _prompt_len         ; not the last prompt token: k and v are enough
        lda #0
        bcs :+
        lda #1
:       sta _kv_only
        ldx tpos
        lda _prompt_ids,x
        jsr _model_step         ; A = token, X = position
        jsr _io_tick
        inc tpos
        lda tpos
        cmp _prompt_len
        bcc tprompt
tgen:   lda _out_len
        cmp #MAX_NEW
        bcs tend
        lda _temp_on
        beq :+
        jsr _sample
        jmp tgot
:       jsr _argmax
tgot:   cmp #TOK_EOS
        beq tend
        cmp #TOK_USER
        beq tend
        ldx _out_len
        sta _out_ids,x
        inc _out_len
        jsr ptok
        lda tpos                ; full context: chat.py would slide the window
        cmp #CONTEXT
        bcs tend
        ldx _out_len
        lda _out_ids-1,x
        ldx tpos
        jsr _model_step
        jsr _io_tick
        inc tpos
        jmp tgen
tend:   lda #0
        tax
        rts

; print token A with _io_putc (the first one without leading spaces)
ptok:   tax
        lda voc_lo,x
        sta cp
        lda voc_hi,x
        sta cp+1
        ldy #0
        lda first
        beq pchar
        sty first
:       lda (cp),y
        cmp #' '
        bne pchar
        iny
        bne :-
pchar:  lda (cp),y
        beq pdone
        sty cy
        jsr _io_putc
        ldy cy
        iny
        bne pchar
pdone:  jmp _io_flush

;-----------------------------------------------------------------------------
