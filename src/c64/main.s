;- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
;
; ChatCBM on the Commodore 64: the main program, with input and output through the kernal.
;
; Load with LOAD"CHATCBM",8 and start with RUN: the BASIC line is
;   2026 SYS2083 REM CHATCBM: A TINY LLM
;
; Memory: all 64 KB. Two modes:
;   kernal    $01 = $36: BASIC banked out, I/O and kernal visible, interrupts on. Only to read
;             the question and to print the tokens.
;   compute   $01 = $34: all RAM, interrupts off. The model runs here; the KV cache (16 KB with
;             4 layers) is at $C000-$FFFF, partly under I/O and kernal.
; Zero page $02-$FF belongs to the program while computing: entering the kernal mode swaps
; $90-$FF with the kernal's copy (kzsave), leaving it swaps them back. The RESTORE key is
; disabled (a CIA2 NMI that is never acknowledged keeps the edge-triggered NMI line low), so
; in compute mode no NMI can go through the vectors at $FFFA, which are KV cache RAM.
;
; Commands: an empty line does nothing, /f toggles the fast mode, /q resets the machine.
; The border changes color at every model step. Fast mode (/f): the screen is blanked at the
; end of the first layer of every step and turned on again to print a token, so for most of
; the computation the VIC-II steals no badline cycles (~5%), without ever waiting.
;
; Built with C64_TEST defined, the program answers the questions of the file QUESTIONS on
; drive 8 instead of the keyboard (tests/e2e.py).
;
        .exportzp sreg, tmp1, tmp2, ptr1
        .export _io_putc, _io_flush, _io_tick
        .import _model_init, _layer_hook, _chat_init, _chat_turn, _temp_on, _rng_state
        .import __BSS_RUN__, __BSS_SIZE__, __ALIGNED_RUN__, __ALIGNED_SIZE__

CHRIN   = $FFCF
CHROUT  = $FFD2
SETLFS  = $FFBA
SETNAM  = $FFBD
OPEN    = $FFC0
CLOSE   = $FFC3
CHKIN   = $FFC6
CHKOUT  = $FFC9
CLRCHN  = $FFCC
READST  = $FFB7
NMIVEC  = $0318
BORDER  = $D020
VICCR1  = $D011                 ; bit 4: screen on (off: no badlines)
CI2ICR  = $DD0D
CI2TAL  = $DD04
CI2CRA  = $DD0E

LINEMAX = 80
COLS    = 40
KZ      = $90                   ; start of the kernal zero page
TMAX    = 16                    ; longest token
WMAX    = 32                    ; longest word kept for wrapping
BLANK_AFTER = 0                 ; fast mode: the screen goes off at the end of this layer

        .zeropage
sreg:   .res 2                  ; registers that the engine takes from the cc65 runtime
tmp1:   .res 1
tmp2:   .res 1
ptr1:   .res 2
        .assert ptr1 < KZ, lderror, "link main.o first: its zero page is used in kernal mode too"

        .segment "LOADADDR"
        .word $0801

        .segment "STARTUP"
SYS_ADDR = 2083
        .word stubend, 2026     ; 2026 SYS2083 REM CHATCBM: A TINY LLM
        .byte $9E, .sprintf("%d", SYS_ADDR), " ", $8F, " CHATCBM: A TINY LLM", 0
stubend:
        .word 0
start:  .assert start = SYS_ADDR, lderror, "update SYS_ADDR in the BASIC line"
        sei
        cld
        ldx #$FF
        txs
        lda #<nmirti            ; RESTORE off: a CIA2 NMI, never acknowledged,
        sta NMIVEC              ; keeps the NMI line (edge triggered) low
        lda #>nmirti
        sta NMIVEC+1
        lda #0
        sta CI2CRA
        sta CI2TAL
        sta CI2TAL+1
        lda #$81
        sta CI2ICR
        lda #$19                ; start timer A (one shot, load the latch)
        sta CI2CRA
        ldx #20
:       dex
        bne :-
        lda #<__BSS_RUN__       ; clear BSS and the aligned tables
        ldx #>__BSS_RUN__
        ldy #>(__BSS_SIZE__ + 255)
        jsr zero
        lda #<__ALIGNED_RUN__
        ldx #>__ALIGNED_RUN__
        ldy #>__ALIGNED_SIZE__
        jsr zero
        ldx #KZ_N - 1           ; the initial values "of the program" in $90-$FF
        lda #0
:       sta kzsave,x
        dex
        bpl :-
        lda BORDER
        sta border
        lda $01
        and #$F8
        sta port
        ora #$06                ; BASIC out, I/O and kernal visible
        sta $01
        lda #<blank             ; the end of every layer can blank the screen
        sta _layer_hook
        lda #>blank
        sta _layer_hook+1
        cli
        jsr compute
        jsr _model_init         ; aligned tables from the image
        jsr _chat_init          ; k and v of <user> at position 0
        jsr kernal
        lda #<banner
        ldx #>banner
        jsr puts
.ifdef C64_TEST
        jmp ftest
.endif

loop:   lda #<you
        ldx #>you
        jsr puts
        jsr getline
        lda line
        beq loop
        cmp #'/'
        bne :+
        lda line+1
        cmp #'q'
        bne :+
        lda CI2ICR              ; RESTORE on again, then reset
        jmp ($FFFC)
:       lda line                ; /f: fast mode (screen off while computing)
        cmp #'/'
        bne :++
        lda line+1
        cmp #'f'
        bne :++
        lda fast
        eor #1
        sta fast
        lda #<fastoff
        ldx #>fastoff
        ldy fast
        beq :+
        lda #<faston
        ldx #>faston
:       jsr puts
        jmp loop
:       lda seeded
        bne :+
        jsr seed
:       lda #<bot
        ldx #>bot
        jsr puts
        lda #5
        sta col
        lda #0
        sta wlen
        jsr compute
        lda #<line
        ldx #>line
        jsr _chat_turn
        pha
        jsr kernal
        pla
        beq :+
        lda #<toolong
        ldx #>toolong
        jsr puts
:       lda #13
        jsr CHROUT
        jmp loop

nmirti: rti

; clear Y pages from A/X
zero:   sta ptr1
        stx ptr1+1
        cpy #0
        beq zdone
        lda #0
        tax
:       sta (ptr1,x)
        inc ptr1
        bne :-
        inc ptr1+1
        dey
        bne :-
zdone:  rts

; ---- modes ------------------------------------------------------------------------
; compute mode: interrupts off, all RAM, the program's zero page
compute:
        sei
        jsr zswap
        lda port
        ora #$04                ; $34: all RAM
        sta $01
        rts

; kernal mode: kernal and I/O visible, the kernal's zero page, interrupts on
kernal: lda port
        ora #$06                ; $36
        sta $01
        lda border
        sta BORDER
        lda VICCR1              ; screen on
        ora #$10
        sta VICCR1
        jsr zswap
        cli
        rts

zswap:  ldx #KZ_N - 1
:       lda KZ,x
        ldy kzsave,x
        sta kzsave,x
        tya
        sta KZ,x
        dex
        bpl :-
        rts
KZ_N    = $100 - KZ

; ---- I/O (kernal mode) ---------------------------------------------------------------
; print the ASCII string A/X ending with 0
puts:   sta ptr1
        stx ptr1+1
        ldy #0
:       lda (ptr1),y
        beq :+
        jsr petscii
        jsr CHROUT
        iny
        bne :-
:       rts

; read a line from the keyboard (screen editor) into line, as ASCII ending with 0
getline:
        ldx #0
:       stx tmp1
        jsr CHRIN
        ldx tmp1
        cmp #13
        beq :+
        jsr ascii
        cpx #LINEMAX
        bcs :-
        sta line,x
        inx
        bne :-
:       lda #0
        sta line,x
        lda #13
        jmp CHROUT

; xorshift32 seed from the jiffy clock and the timers, never zero
seed:   lda $A2
        eor $D012
        sta _rng_state
        lda $A1
        eor $DC04
        sta _rng_state+1
        lda $A0
        eor $DC05
        sta _rng_state+2
        lda #1
        sta _rng_state+3
        sta seeded
        rts

; ASCII -> PETSCII (upper/lower case set): a-z -> $41-$5A, A-Z -> $C1-$DA
petscii:
        cmp #'a'
        bcc :+
        cmp #'z' + 1
        bcs pdone
        and #$DF
        rts
:       cmp #'A'
        bcc pdone
        cmp #'Z' + 1
        bcs pdone
        ora #$80
pdone:  rts

; PETSCII -> ASCII: $41-$5A -> a-z, $61-$7A and $C1-$DA -> A-Z
ascii:  cmp #$41
        bcc adone
        cmp #$5B
        bcs :+
        ora #$20
        rts
:       cmp #$61
        bcc adone
        cmp #$7B
        bcs :+
        and #$DF
        rts
:       cmp #$C1
        bcc adone
        cmp #$DB
        bcs adone
        and #$7F
adone:  rts

; ---- called by the engine (compute mode) -----------------------------------------------
_io_putc:
        ldx tlen
        cpx #TMAX
        bcs :+
        sta tbuf,x
        inc tlen
:       rts

; print the collected token, wrapping before a word that does not fit the line
_io_flush:
        jsr kernal
        ldx #0
        lda tlen
        bne :+
        jmp fdone
:       lda raw                 ; test: the token as it is, ASCII, to the open file
        beq :+
fraw:   lda tbuf,x
        jsr CHROUT
        inx
        cpx tlen
        bcc fraw
        jsr CLRCHN              ; and then to the screen as well
        ldx #0
:       lda tbuf                ; on screen, wrapping between words
        cmp #' '
        bne fcont
        lda #0                  ; a new word
        sta wlen
        lda tlen
        clc
        adc col
        cmp #COLS               ; keep the last column free for punctuation
        bcs fwrap
        lda #' '
        jsr CHROUT
        inc col
        inx
        bne fprint
fwrap:  lda #13                 ; a new line instead of the space
        jsr CHROUT
        lda #0
        sta col
        inx
        bne fprint
fcont:  ldy #COLS               ; rest of a word: if it does not fit the line, the whole
        lda tbuf                ; word moves to the next one (deleted with DEL and printed
        cmp #'a'                ; again). Letters keep the last column free, punctuation
        bcc :+                  ; can use it
        cmp #'z' + 1
        bcc :++
:       iny
:       sty tt0
        lda tlen
        clc
        adc col
        cmp tt0
        bcc fprint
        lda wlen
        beq fprint
        clc
        adc tlen
        cmp #COLS
        bcs fprint
        ldy wlen
:       lda #$14                ; DEL
        jsr CHROUT
        dey
        bne :-
        lda #13
        jsr CHROUT
:       lda wbuf,y
        jsr petscii
        jsr CHROUT
        iny
        cpy wlen
        bcc :-
        sty col
fprint: cpx tlen
        bcs fdone
        lda tbuf,x
        ldy wlen
        cpy #WMAX
        bcs :+
        sta wbuf,y
        inc wlen
:       jsr petscii
        jsr CHROUT
        inc col
        lda col
        cmp #COLS
        bcc :+
        lda #0                  ; the cursor already moved to the next line
        sta col
        sta wlen
:       inx
        bne fprint
fdone:  lda raw
        beq :+
        ldx #3                  ; back to the file
        jsr CHKOUT
:       lda #0
        sta tlen
        jmp compute

; end of a layer (A = layer): in fast mode, after layer BLANK_AFTER the screen goes off until
; the next token, so the VIC-II steals no badline cycles. The screen is turned on again when
; the kernal mode prints: it stays on, without waiting, for the first part of the next step.
blank:  ldx fast
        beq :+
        cmp #BLANK_AFTER
        bne :+
        lda port
        ora #$05                ; $35: I/O visible, RAM elsewhere
        sta $01
        lda VICCR1
        and #$EF
        sta VICCR1
        lda port
        ora #$04
        sta $01
:       rts

; a sign of life at every model step: the border changes color
_io_tick:
        lda port
        ora #$05                ; $35: I/O visible, RAM elsewhere
        sta $01
        inc BORDER
        lda port
        ora #$04
        sta $01
        rts

.ifdef C64_TEST
; ---- test with files -------------------------------------------------------------------
; Read the questions, one per line, from the file QUESTIONS on drive 8 (the host directory in
; VICE) and write to ANSWERS the greedy answer and the answer sampled with seed 2026, one per
; line. A JAM instruction at the end: VICE with -jamaction 5 quits.
TEST_SEED = 2026

ftest:  lda #fqname_end - fqname
        ldx #<fqname
        ldy #>fqname
        jsr SETNAM
        lda #2
        ldx #8
        ldy #2
        jsr SETLFS
        jsr OPEN
        lda #faname_end - faname
        ldx #<faname
        ldy #>faname
        jsr SETNAM
        lda #3
        ldx #8
        ldy #3
        jsr SETLFS
        jsr OPEN
        lda #1
        sta raw
fnext:  ldx #2                  ; one line of the file (ending with $0A or $0D)
        jsr CHKIN
        ldx #0
:       stx tmp1
        jsr READST
        bne feof
        jsr CHRIN
        ldx tmp1
        cmp #$0A
        beq fline
        cmp #$0D
        beq fline
        cpx #LINEMAX
        bcs :-
        sta line,x
        inx
        bne :-
feof:   ldx tmp1
fline:  lda #0
        sta line,x
        stx tmp2
        jsr CLRCHN
        lda tmp2
        beq fend
        lda #0                  ; greedy
        jsr fturn
        lda #1                  ; sampled, fixed seed
        jsr fturn
        jmp fnext
fend:   lda #3
        jsr CLOSE
        lda #2
        jsr CLOSE
        .byte $02               ; JAM

; one turn on the line just read, answer to ANSWERS: A = temperature on
fturn:  sta _temp_on
        lda #<you               ; on screen too, to look at it
        ldx #>you
        jsr puts
        lda #<line
        ldx #>line
        jsr puts
        lda #13
        jsr CHROUT
        lda #<bot
        ldx #>bot
        jsr puts
        lda #5
        sta col
        lda #0
        sta wlen
        lda #<TEST_SEED
        sta _rng_state
        lda #>TEST_SEED
        sta _rng_state+1
        lda #0
        sta _rng_state+2
        sta _rng_state+3
        ldx #3
        jsr CHKOUT
        jsr compute
        lda #<line
        ldx #>line
        jsr _chat_turn
        jsr kernal
        lda #$0A
        jsr CHROUT
        jsr CLRCHN
        lda #13
        jmp CHROUT

fqname: .byte "QUESTIONS,S,R"
fqname_end:
faname: .byte "@:ANSWERS,S,W"
faname_end:
.endif

        .rodata
banner: .byte 14, 8, 147        ; lower case, no shift+C= switch, clear the screen
        .byte "ChatCBM: a tiny language model.", 13
        .byte "Ask about the animals or a story.", 13
        .byte "Each word piece takes about 2 seconds.", 13, 0
you:    .byte 13, "you> ", 0
bot:    .byte "bot> ", 0
toolong: .byte "(too long)", 0
faston: .byte "fast mode: screen off while thinking", 0
fastoff: .byte "fast mode off", 0

        .bss
kzsave: .res KZ_N
line:   .res LINEMAX + 1
tbuf:   .res TMAX
tlen:   .res 1
wbuf:   .res WMAX               ; the word being printed (for wrapping)
wlen:   .res 1
col:    .res 1
port:   .res 1
border: .res 1
seeded: .res 1
fast:   .res 1                  ; /f: screen off while computing
tt0:    .res 1
raw:    .res 1                  ; 1: tokens as ASCII to a file (test)

;-----------------------------------------------------------------------------
