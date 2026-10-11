/*- ChatCBM - Copyright 2026 Fabio Carignano -------------------------------*/

/* Unit test of src/model.s: the logits after a real prompt, with full steps and with
   the reduced steps of the prompt (k and v only), then a dozen greedy tokens. */

#include "check.h"
#include "vec_model.h"

extern int acc[256];                                  /* buffers.s */
extern unsigned char kv_only;                         /* model.s */
void model_init(void);
void __fastcall__ model_step(unsigned tok_pos);       /* A = token, X = position */
unsigned char argmax(void);
#define step(tok, pos) model_step((tok) | ((unsigned)(pos) << 8))

static void run_prompt(unsigned char reduced)
{
    unsigned char i;
    for (i = 0; i < PROMPT_LEN; ++i) {
        kv_only = reduced && i + 1 < PROMPT_LEN;
        step(prompt[i], i);
    }
    kv_only = 0;
}

int main(void)
{
    unsigned j;
    unsigned char pos, tok;
    model_init();
    run_prompt(0);
    for (j = 0; j < 256; ++j)
        check("z full", j, acc[j], out_z[j]);
    run_prompt(1);
    for (j = 0; j < 256; ++j)
        check("z reduced", j, acc[j], out_z[j]);
    pos = PROMPT_LEN;
    for (j = 0; j < GREEDY_LEN; ++j) {
        tok = argmax();
        check("greedy", j, tok, out_greedy[j]);
        step(tok, pos++);
    }
    return report("model");
}

/*--------------------------------------------------------------------------*/
