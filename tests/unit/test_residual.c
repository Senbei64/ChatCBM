/*- ChatCBM - Copyright 2026 Fabio Carignano -------------------------------*/

/* Unit test of src/residual.s: a layer output added to the residual stream, with
   random values and with saturation. */

#include <string.h>
#include "check.h"
#include "vec_residual.h"

extern int acc[256], unit[32], x[32];                 /* buffers.s */
void resid_add(void);

int main(void)
{
    unsigned c, i;
    for (c = 0; c < CASES; ++c) {
        memcpy(x, in_x + 32 * c, 64);
        memcpy(acc, in_acc + 32 * c, 64);
        memcpy(unit, in_unit + 32 * c, 64);
        resid_add();
        for (i = 0; i < 32; ++i)
            check("x", 32 * c + i, x[i], out_x[32 * c + i]);
    }
    return report("residual");
}

/*--------------------------------------------------------------------------*/
