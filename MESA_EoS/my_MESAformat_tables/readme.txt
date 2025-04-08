H EoS data comes from Mazevet 2022 blended with CMS19.
    log10T ranges from 2 to 8 in steps of 0.05 (121 entries)
    log10rho ranges from -8 to 6 in steps of 0.05 (281 entries)
    ** Note that some of the columns calculated by finite difference--particularly dE_drho_T and dS_drho_T--have numerical problems for log10rho <= -5. Don't use these tables in that density range.

He EoS data comes from CMS19, columns 0-4 inclusive (others are calculated by finite difference).
    log10T, log10rho ranges are the same as above.
    ** Note that some of the columns calculated by finite difference--particularly chiRho, dlS_dlrho_T, dlE_dlrho_T, chiT, and dlS_dlT_rho (and therefore dS_drho_T, dE_drho_T, and dS_dT_rho) have numerical weirdness in certain places. I have smoothed over the boundaries of the QMD box at log10T = 6 and log10rho = 2, but there is still bad behavior at log10T \lesssim 2.3, so don't  use these tables at extremely cold temperatures.
    ** Note that even after the linear smoothing there is still a little numerical weirdness in the above quantities at large values of log10Q = log10rho - 2log10T + 12. Values of log10Q <= 4.25 look okay, just judging by eye from the plots.

Mixed-composition H/He tables come from the two tables above, combined using the additive volume law, plus Howard & Guillot 2023 mixture terms.

Mesa-h2o_100z00x.data is a blended water EoS. It comes from AQUA where possible; SESAME H2O where not; extrapolated with linear spline to the full CMS parameter space in (rho, T).

Mesa-rock_100z00x.data is a blended rock EoS. It comes from ANEOS forsterite where possible; SESAME basalt where not; extrapolated with linear spline to the full CMS parameter space in (rho, T).

Mesa-rock_water_100z00x.data is a 50/50 blend of the above.

Mesa-iron_100z00x.data is a blended iron EoS. It comes from ANEOS iron where possible, SESAME iron where not, extrapolated to the full CMS parameter space in (rho, T). (Note that the SESAME table has no specific entropy information, so the ANEOS specific entropy is extrapolated farther than the other quantities.)


Mesa-planetBlend_ZZzXXx.data blends the Mazevet/CMS H, CMS col 0-4 He, 50/50 rock_water EoSs in the proportions specified in the table name. Howard & Guillot 2023 terms are added appropriately for the H/He proportions. Combination with Z is done only with the additive volume law (no interaction terms.)