/*- ChatCBM - Copyright 2026 Fabio Carignano -------------------------------*/

/* Unit test of src/logits.s: the 256 output sums of the model turned into integer
   logits z = logit * 16 log2(e). */

#include <string.h>
#include "check.h"
#include "vec_logits.h"

extern int acc[256], kx;                              /* buffers.s */
extern const signed char *logit_kw;                   /* logits.s */
void logits_fix(void);

int main(void)
{
    unsigned c, j;
    logit_kw = kw;
    for (c = 0; c < CASES; ++c) {
        memcpy(acc, in_acc + 256 * c, 512);
        kx = in_kx[c];
        logits_fix();
        for (j = 0; j < 256; ++j)
            check("z", 256 * c + j, acc[j], out_z[256 * c + j]);
    }
    return report("logits");
}

/*--------------------------------------------------------------------------*/
