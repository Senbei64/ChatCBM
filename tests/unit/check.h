/*- ChatCBM - Copyright 2026 Fabio Carignano -------------------------------*/

/* Minimal checks for the unit tests: count the errors, print the first ones, and
   make the program exit with an error code so that make stops. */

#include <stdio.h>

static unsigned errors;

static void check(const char *what, unsigned i, long got, long want)
{
    if (got == want)
        return;
    if (errors < 8)
        printf("  %s[%u] = %ld, expected %ld\n", what, i, got, want);
    ++errors;
}

static int report(const char *name)
{
    printf("%s: %s\n", name, errors ? "FAILED" : "ok");
    return errors != 0;
}

/*--------------------------------------------------------------------------*/
