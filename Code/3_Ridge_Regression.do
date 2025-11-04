clear all
set more off

* 1) Download wage1.dta locally (HTTP often avoids r(677) SSL issues)
local url http://fmwww.bc.edu/ec-p/data/wooldridge/wage1.dta
cap copy "`url'" "wage1.dta", replace
if _rc {
    di as error "Download failed (r(677)). Check proxy/VPN or use a different mirror."
    exit _rc
}

* 2) Load dataset
use "wage1.dta", clear

* Quick look
summarize wage educ exper tenure

*--------------------*
* 3) OLS regression  *
*--------------------*
reg wage educ exper tenure
estimates store OLS

*----------------------------*
* 4) Ridge regression (L2)   *
*    Using Stata's lasso suite
*    alpha(0) => pure Ridge
*    selection(cv) => K-fold CV to pick lambda
*----------------------------*
lasso linear wage educ exper tenure, selection(cv) alpha(0)

* Selected lambda (ridge penalty)
display as txt "Selected lambda (ridge): " %9.6f e(lambda)

* Show ridge coefficients
lassocoef, display(coef, standardized)

* Store ridge for comparison
estimates store Ridge

*-----------------------*
* 5) Compare estimates  *
*-----------------------*
estimates table OLS Ridge, b(%9.4f) se(%9.4f) star stats(N r2) varwidth(18)

* (Optional) Show the CV curve and the chosen lambda
estat cvplot


********************************************************************************
*--------------------------------------------------------------*
* STATA 15: Ridge via user-written command (ridgereg, SSC)     *
* - Downloads Wooldridge wage1; if it fails, falls back to auto *
* - Runs OLS and Ridge                                          *
* - (Optional) simple hold-out CV to choose k                   *
*--------------------------------------------------------------*

clear all
set more off
set seed 12345

* 1) Try to download Wooldridge wage1 (HTTP to avoid SSL r(677))
cap copy "http://fmwww.bc.edu/ec-p/data/wooldridge/wage1.dta" "wage1.dta", replace
if _rc {
    di as error "Could not fetch wage1.dta; falling back to built-in auto.dta."
    sysuse auto, clear
    keep if !missing(price, mpg, weight, length)
    local dep   price
    local indep mpg weight length
}
else {
    use "wage1.dta", clear
    summarize wage educ exper tenure
    local dep   wage
    local indep educ exper tenure
}

* 2) Install user-written ridge command (Stata 15 compatible)
cap which ridgereg
if _rc ssc install ridgereg, replace

* 3) OLS for reference
reg `dep' `indep'
estimates store OLS

* 4) Ridge with a chosen k (ridge parameter)
*    (You can adjust k below or let the CV block pick it)
local k 0.10
ridgereg `dep' `indep', k(`k')
estimates store Ridge_k`k'

* 5) Compare OLS vs Ridge (fixed k)
estimates table OLS Ridge_k`k', b(%9.4f) varwidth(18) stats(N r2, fmt(%9.3f))

*----------------------------*
* 6) Optional: pick k by CV  *
*    Simple hold-out split   *
*----------------------------*
preserve
tempvar u valid
gen double `u' = runiform()
gen byte `valid' = (`u' > .70)    // 30% validation, 70% training

* Grid of k values to try
local kgrid 0.001 0.005 0.01 0.02 0.05 0.10 0.20 0.50 1 2

tempname results
mata: st_matrix("`results'", J(0,2,.))   // will store: k , RMSE_valid

quietly {
    foreach k of local kgrid {
        * Fit on training sample only
        ridgereg `dep' `indep' if !`valid', k(`k')

        * Predict on validation sample
        tempvar yhat
        predict double `yhat' if `valid', xb

        * Compute validation RMSE
        quietly summarize `dep' if `valid'
        scalar Nval = r(N)
        tempvar se
        gen double `se' = (`dep' - `yhat')^2 if `valid'
        quietly summarize `se'
        scalar RMSE = sqrt(r(mean))

        * Append (k, RMSE) to results matrix
        mata: st_matrix("`results'", (st_matrix("`results'") \ ( `k' , st_numscalar("RMSE") )))
        drop `yhat' `se'
    }
}

* Find best k (minimum RMSE)
mata: st_view(KRM=., ., "`results'")
mata: st_numscalar("k_best", KRM[ colmin(KRM[.,2]), 1 ])
mata: st_numscalar("rmse_best", KRM[ colmin(KRM[.,2]), 2 ])

di as txt "Best k from hold-out CV = " %9.6f scalar(k_best) "  (RMSE = " %9.6f scalar(rmse_best) ")"

* Refit ridge on full sample with k_best
restore
ridgereg `dep' `indep', k(`=scalar(k_best)')
estimates store Ridge_CV

* Compare OLS vs Ridge (CV-chosen k)
estimates table OLS Ridge_CV, b(%9.4f) varwidth(18) stats(N r2, fmt(%9.3f))

* (Optional) list the CV results table
mat colnames `results' = k rmse_valid
matlist `results', format(%9.6g)

*--------------------------------------------------------------*
* End
*--------------------------------------------------------------*
