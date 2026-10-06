/*
 * P4-139 criterion 4 (#478) C oracle: which scaling factors upstream's
 * tj3SetScalingFactor accepts, and what tj3GetScalingFactors reports.
 *
 * The Rust `ScalingFactor` became a validated type whose `try_new` accepts
 * exactly upstream's set, and `tj3SetScalingFactor` is now a thin layer over
 * it. The rule is a field-by-field lookup in the `sf` table
 * (`turbojpeg.c:199-217`, `:2053-2058`), so a factor *equal in value* to a
 * supported one but not written the same way (4/8, 2/2) is refused. That is
 * easy to get wrong by normalising first, which is why it is traced here
 * rather than transcribed.
 *
 * Output, one line per case:
 *   `sf <index> <num>/<denom>` for every tj3GetScalingFactors entry, then
 *   `set <num>/<denom> <rc> kind=<k>` over num, denom in -1..=20.
 * `kind` is `none` on success, `unsupported` when the message carries
 * upstream's "Unsupported scaling factor" text, and `other` otherwise. The
 * `function():` prefix is not compared: this port writes `function:`, as it
 * does for every TurboJPEG error.
 *
 * Usage: scaling_factor_oracle
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <turbojpeg.h>

int main(void)
{
  int count = 0, i, num, denom;
  tjscalingfactor *table = tj3GetScalingFactors(&count);
  tjhandle handle;

  if (!table) { fprintf(stderr, "tj3GetScalingFactors\n"); return 2; }
  for (i = 0; i < count; i++)
    printf("sf %d %d/%d\n", i, table[i].num, table[i].denom);

  handle = tj3Init(TJINIT_DECOMPRESS);
  if (!handle) { fprintf(stderr, "tj3Init\n"); return 2; }

  for (num = -1; num <= 20; num++) {
    for (denom = -1; denom <= 20; denom++) {
      tjscalingfactor factor;
      int rc;
      const char *kind;

      factor.num = num;
      factor.denom = denom;
      rc = tj3SetScalingFactor(handle, factor);
      if (rc == 0)
        kind = "none";
      else if (strstr(tj3GetErrorStr(handle), "Unsupported scaling factor"))
        kind = "unsupported";
      else
        kind = "other";
      printf("set %d/%d %d kind=%s\n", num, denom, rc, kind);
    }
  }

  tj3Destroy(handle);
  return 0;
}
