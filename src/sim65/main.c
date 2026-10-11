/*- ChatCBM - Copyright 2026 Fabio Carignano -------------------------------*/

/* ChatCBM on sim65, the 6502 simulator of cc65.

     sim65 chatcbm.prg                      chat: one question per line from stdin
     sim65 chatcbm.prg QUESTIONS ANSWERS    test: for every line of QUESTIONS write the
                                            greedy answer and the answer sampled with
                                            seed 2026, one per line, to ANSWERS

   Everything runs in the 6502 modules of src/; this file only reads lines and writes
   characters, the machine-dependent part. The test protocol is the same as on the C64
   (src/c64/main.s), checked by tests/e2e.py. */

#include <stdio.h>
#include <string.h>

#define TEST_SEED 2026

void model_init(void);                               /* model.s */
void chat_init(void);                                /* chat.s */
unsigned char __fastcall__ chat_turn(const char *text);
extern unsigned char temp_on;
extern unsigned long rng_state;                      /* sampler.s */

static FILE *out;
static char line[128];

void __fastcall__ io_putc(char c)                    /* chat.s: a character of a token */
{
    fputc(c, out);
}

void io_flush(void)                                  /* chat.s: end of a token */
{
}

void io_tick(void)                                   /* chat.s: after every model step */
{
}

static void strip_newline(void)
{
    line[strcspn(line, "\r\n")] = 0;
}

static int test(const char *questions, const char *answers)
{
    FILE *in = fopen(questions, "r");
    out = fopen(answers, "w");
    if (!in || !out) {
        puts("cannot open the files");
        return 1;
    }
    while (fgets(line, sizeof line, in)) {
        strip_newline();
        if (!line[0])
            continue;
        temp_on = 0;
        chat_turn(line);
        fputc('\n', out);
        temp_on = 1;
        rng_state = TEST_SEED;
        chat_turn(line);
        fputc('\n', out);
    }
    fclose(in);
    fclose(out);
    return 0;
}

int main(int argc, char **argv)
{
    model_init();
    chat_init();
    if (argc == 3)
        return test(argv[1], argv[2]);
    out = stdout;
    rng_state = TEST_SEED;
    while (fgets(line, sizeof line, stdin)) {
        strip_newline();
        if (!line[0])
            continue;
        printf("you> %s\nbot> ", line);
        if (chat_turn(line))
            printf("(too long)");
        printf("\n");
    }
    return 0;
}

/*--------------------------------------------------------------------------*/
