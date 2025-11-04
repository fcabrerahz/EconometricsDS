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


