All subfolders in this folder are based on the make_brown_dwarf MESA test suite example, hewing as close as possible to the settings enumerated in Paxton et al. 2013.

Unfortunately, in the current version of MESA (r24.03.01), setting initial radius equal to 5 R_J (as Paxton et al. 2013 did) doesn't work for any masses smaller than 20 M_J. Frank Timmes says "this is a known issue...some preliminary mesa-dev investigations suggested the phenomena may be due to changes since 2013 how the low temperature eos is handled". He suggested the workaround from https://lists.mesastar.org/pipermail/mesa-users/2024-April/015133.html, of reducing the initial radius and (if it becomes too small and starts throwing a "failed in kap_get get_kap_from_rhoT temp too low in integration" error, setting kap_lowT_prefix equal to "lowT_fa05_gs98" instead of "lowT_freedman11" in the create step, but not the evolve step).

The other important changes from Paxton et al. 2013 are:
eosDT_file_prefix = 'mesa' #replaces deprecated eos_file_prefix='mesa'

Zbase = 0.02d0
convergence_ignore_equL_residuals = .true.
