/*- ChatCBM - Copyright 2026 Fabio Carignano -------------------------------*/

/* Unit test of src/gelu.s: FFN outputs of the model and values inside and outside the
   range of the table. */

#include <string.h>
#include "check.h"
#include "vec_gelu.h"

extern int acc[256], unit[32];                        /* buffers.s */
extern const int *gelu_table;                         /* gelu.s */
void gelu(void);

int main(void)
{
    unsigned c, i;
    gelu_table = table;
    for (c = 0; c < CASES; ++c) {
        memcpy(acc, in_acc + 32 * c, 64);
        memcpy(unit, in_unit + 32 * c, 64);
        gelu();
        for (i = 0; i < 32; ++i)
            check("acc", 32 * c + i, acc[i], out_acc[32 * c + i]);
    }
    return report("gelu");
}

/*--------------------------------------------------------------------------*/
