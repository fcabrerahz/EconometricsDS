*--------------------------------------------------------------*
* OLS vs Ridge Regression Example using Wooldridge's wage1 data *
*--------------------------------------------------------------*

* Try plain HTTP (avoids SSL problems that trigger r(677))
local url http://fmwww.bc.edu/ec-p/data/wooldridge/wage1.dta

* Download to a local file, then use it
cap copy "`url'" "wage1.dta", replace
if _rc {
    di as error "Download failed (r(677)). Check proxy/VPN and try Option B/C below."
    exit _rc
}
use "wage1.dta", clear

* OLS
reg wage educ exper tenure
estimates store OLS

* Install lassopack if needed (for ridge via elastic net with alpha=0)
ssc install lassopack, replace

* Ridge with 10-fold CV, alpha(0)=Ridge
cvelasticnet price mpg weight length, alpha(0) kfolds(10) seed(12345)

* Display the selected lambda (ridge penalty)
display "Selected lambda (ridge): " e(lambda_sel)

* List ridge coefficients
matrix list e(b)
estimates store Ridge

* 5. Compare OLS vs Ridge estimates side by side
estimates table OLS Ridge, b(%9.4f) stats(N r2) varwidth(18)
